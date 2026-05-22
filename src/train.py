import json
import logging
import pickle
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    recall_score, f1_score, accuracy_score, precision_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    average_precision_score,
)
from config import (
    TEST_SIZE, RANDOM_STATE, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI,
    MLFLOW_RUN_NAME, MLFLOW_MODEL_REGISTRY, MODEL_PATH, SCALER_PATH,
    METRICS_PATH, FEATURES_PATH, TEST_INDICES_PATH, THRESHOLD_PATH,
    RECALL_MINIMO, F1_MINIMO, N_SPLITS_CV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Modelos sin SMOTE: class_weight='balanced' evita la distorsión de priors
# que SMOTE introduce (validado en EDA).
MODELOS = {
    "LR_balanced": LogisticRegression(
        class_weight="balanced", C=0.1, max_iter=1000,
        solver="liblinear", random_state=RANDOM_STATE,
    ),
    "RF_balanced": RandomForestClassifier(
        class_weight="balanced", n_estimators=200, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
}


def evaluar(modelo, X_test, y_test, threshold: float = 0.5) -> dict:
    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
        "balanced_acc": round(balanced_accuracy_score(y_test, y_pred), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "threshold": round(threshold, 4),
    }


def optimal_recall_with_f1_floor(
    probs: np.ndarray, y_true: pd.Series, f1_floor: float = F1_MINIMO,
    start: float = 0.05, end: float = 0.95, step: float = 0.005,
) -> tuple[float, float, float]:
    """Threshold que MAXIMIZA Recall sujeto a F1 >= f1_floor.

    Devuelve (thr, recall, f1). Fallback: threshold de max F1 si nadie cumple.
    """
    candidatos = []
    for t in np.arange(start, end, step):
        y_hat = (probs >= t).astype(int)
        f1v = f1_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        if f1v >= f1_floor:
            candidatos.append((t, rec, f1v))
    if candidatos:
        candidatos.sort(key=lambda x: (x[1], x[2]), reverse=True)
        t, r, f = candidatos[0]
        return round(float(t), 3), round(float(r), 4), round(float(f), 4)
    best = (0.5, 0.0, 0.0)
    for t in np.arange(start, end, step):
        y_hat = (probs >= t).astype(int)
        f1v = f1_score(y_true, y_hat, zero_division=0)
        if f1v > best[2]:
            best = (t, recall_score(y_true, y_hat, zero_division=0), f1v)
    return round(float(best[0]), 3), round(float(best[1]), 4), round(float(best[2]), 4)


def recall_cv(modelo, X, y) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(modelo, X, y, scoring="recall", cv=cv, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def entrenar(X: pd.DataFrame, y: pd.Series, scaler=None) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    with open(TEST_INDICES_PATH, "w") as f:
        json.dump(X_test.index.tolist(), f)
    logger.info(f"Índices del test set guardados en {TEST_INDICES_PATH}")

    mejor_recall_cv = -1.0
    mejor_modelo = None
    mejor_nombre = None
    mejores_metricas = {}
    mejor_threshold = 0.5
    mejor_run_id = None

    for nombre, modelo in MODELOS.items():
        logger.info(f"Entrenando {nombre}...")
        with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_{nombre}") as run:
            r_mean, r_std = recall_cv(modelo, X_train, y_train)
            logger.info(f"{nombre} | Recall CV = {r_mean:.4f} ± {r_std:.4f}")

            modelo.fit(X_train, y_train)
            probs_test = modelo.predict_proba(X_test)[:, 1]
            thr_op, rec_op, f1_op = optimal_recall_with_f1_floor(probs_test, y_test)
            metricas = evaluar(modelo, X_test, y_test, threshold=thr_op)
            metricas["recall_cv_mean"] = round(r_mean, 4)
            metricas["recall_cv_std"] = round(r_std, 4)

            mlflow.log_params({
                "modelo": nombre,
                "k_features": X.shape[1],
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "scoring_cv": "recall",
                "f1_floor": F1_MINIMO,
                "threshold_operativo": thr_op,
            })
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(modelo, "modelo")

            logger.info(
                f"{nombre} | thr={thr_op:.3f} | "
                f"Recall={metricas['recall']:.4f}  F1={metricas['f1']:.4f}  "
                f"Prec={metricas['precision']:.4f}  MCC={metricas['mcc']:.4f}"
            )

            if r_mean > mejor_recall_cv:
                mejor_recall_cv = r_mean
                mejor_modelo = modelo
                mejor_nombre = nombre
                mejores_metricas = metricas
                mejor_threshold = thr_op
                mejor_run_id = run.info.run_id

    logger.info(
        f"Mejor modelo (por Recall CV): {mejor_nombre} | "
        f"Recall_CV={mejor_recall_cv:.4f} | thr_op={mejor_threshold:.3f} | "
        f"Recall_test={mejores_metricas['recall']:.4f} | F1_test={mejores_metricas['f1']:.4f}"
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(mejor_modelo, f)
    logger.info(f"Modelo guardado en {MODEL_PATH}")

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": mejor_threshold, "f1_floor": F1_MINIMO}, f, indent=2)
    logger.info(f"Threshold operativo guardado en {THRESHOLD_PATH}")

    mejores_metricas["modelo"] = mejor_nombre
    with open(METRICS_PATH, "w") as f:
        json.dump(mejores_metricas, f, indent=2)

    features_df = pd.DataFrame({"feature": X.columns.tolist()})
    features_df.to_csv(FEATURES_PATH, index=False)

    if scaler is not None:
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler guardado en {SCALER_PATH}")

    if mejor_run_id:
        registrar_y_promover(mejor_run_id, mejores_metricas)

    return mejores_metricas


def registrar_y_promover(run_id: str, metricas: dict):
    """Registra el modelo en MLflow Model Registry y lo promueve.

    Flujo:
      Quality gate pasado → Staging
      Supera al modelo en Production → Production (con archive del anterior)
    """
    from mlflow import MlflowClient

    recall = metricas.get("recall", 0)
    f1 = metricas.get("f1", 0)
    pasa_gate = recall >= RECALL_MINIMO and f1 >= F1_MINIMO

    model_uri = f"runs:/{run_id}/modelo"
    result = mlflow.register_model(model_uri, MLFLOW_MODEL_REGISTRY)
    version = result.version
    logger.info(f"Modelo registrado: {MLFLOW_MODEL_REGISTRY} v{version}")

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    if not pasa_gate:
        logger.warning(
            f"Modelo v{version} NO promovido (no pasa gate: "
            f"recall={recall} f1={f1})"
        )
        return

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_REGISTRY, version=version, stage="Staging"
    )
    logger.info(f"v{version} → Staging")

    # ¿Hay un Production actual? Si no, promover. Si sí, comparar recall.
    try:
        prod_versions = client.get_latest_versions(
            MLFLOW_MODEL_REGISTRY, stages=["Production"]
        )
    except Exception:
        prod_versions = []

    promover = True
    if prod_versions:
        prod_v = prod_versions[0]
        try:
            prod_recall = float(
                client.get_run(prod_v.run_id).data.metrics.get("recall", 0)
            )
        except Exception:
            prod_recall = 0
        if recall <= prod_recall:
            promover = False
            logger.info(
                f"v{version} (recall={recall}) NO supera a "
                f"Production v{prod_v.version} (recall={prod_recall})"
            )

    if promover:
        # Archivar versiones de Production anteriores
        for pv in prod_versions:
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_REGISTRY, version=pv.version, stage="Archived"
            )
            logger.info(f"v{pv.version} → Archived")
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_REGISTRY, version=version, stage="Production"
        )
        logger.info(f"v{version} → Production")


def validar_quality_gate(metricas: dict):
    recall = metricas.get("recall", 0)
    f1 = metricas.get("f1", 0)
    fallos = []
    if recall < RECALL_MINIMO:
        fallos.append(f"recall={recall:.4f} < mínimo={RECALL_MINIMO}")
    if f1 < F1_MINIMO:
        fallos.append(f"f1={f1:.4f} < piso={F1_MINIMO}")
    if fallos:
        logger.error("Quality gate FALLIDO: " + " | ".join(fallos))
        sys.exit(1)
    logger.info(
        f"Quality gate APROBADO: recall={recall:.4f} >= {RECALL_MINIMO}, "
        f"f1={f1:.4f} >= {F1_MINIMO}"
    )
