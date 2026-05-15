import json
import logging
import pickle
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config import (
    TEST_SIZE, RANDOM_STATE, TECNICA_BALANCEO, MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI, MLFLOW_RUN_NAME, MODEL_PATH, SCALER_PATH,
    METRICS_PATH, FEATURES_PATH, RECALL_MINIMO,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELOS = {
    "LogisticRegression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE
    ),
}


def balancear(X_train, y_train):
    logger.info(f"Balanceo con técnica: {TECNICA_BALANCEO}")
    if TECNICA_BALANCEO == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif TECNICA_BALANCEO == "undersampling":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    else:
        return X_train, y_train
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    logger.info(f"Distribución post-balanceo: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def evaluar(modelo, X_test, y_test) -> dict:
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else y_pred
    return {
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def entrenar(X: pd.DataFrame, y: pd.Series, scaler=None) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train_bal, y_train_bal = balancear(X_train, y_train)

    mejor_recall = -1
    mejor_modelo = None
    mejor_nombre = None
    mejores_metricas = {}

    for nombre, modelo in MODELOS.items():
        logger.info(f"Entrenando {nombre}...")
        with mlflow.start_run(run_name=f"{MLFLOW_RUN_NAME}_{nombre}"):
            modelo.fit(X_train_bal, y_train_bal)
            metricas = evaluar(modelo, X_test, y_test)

            mlflow.log_params({
                "modelo": nombre,
                "balanceo": TECNICA_BALANCEO,
                "k_features": X.shape[1],
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            })
            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(modelo, "modelo")

            logger.info(f"{nombre}: {metricas}")

            if metricas["recall"] > mejor_recall:
                mejor_recall = metricas["recall"]
                mejor_modelo = modelo
                mejor_nombre = nombre
                mejores_metricas = metricas

    logger.info(f"Mejor modelo: {mejor_nombre} | Recall={mejor_recall:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(mejor_modelo, f)
    logger.info(f"Modelo guardado en {MODEL_PATH}")

    mejores_metricas["modelo"] = mejor_nombre
    with open(METRICS_PATH, "w") as f:
        json.dump(mejores_metricas, f, indent=2)

    features_df = pd.DataFrame({"feature": X.columns.tolist()})
    features_df.to_csv(FEATURES_PATH, index=False)

    if scaler is not None:
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler guardado en {SCALER_PATH}")

    return mejores_metricas


def validar_quality_gate(metricas: dict):
    recall = metricas.get("recall", 0)
    if recall < RECALL_MINIMO:
        logger.error(
            f"Quality gate FALLIDO: recall={recall:.4f} < mínimo={RECALL_MINIMO}"
        )
        sys.exit(1)
    logger.info(f"Quality gate APROBADO: recall={recall:.4f} >= {RECALL_MINIMO}")
