import logging
import sys
import json
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.cf_matrix import make_confusion_matrix
from config import (
    MODEL_PATH, REPORTES_DIR, THRESHOLD_PATH,
    RECALL_MINIMO, F1_MINIMO, TEST_INDICES_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | EVALUATE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _cargar_threshold() -> float:
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            return float(json.load(f)["threshold"])
    logger.warning("threshold.json no encontrado, usando 0.5")
    return 0.5


def ejecutar(X, y, metricas: dict):
    logger.info("=== Paso 4: Evaluación + Quality Gate ===")

    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)

    with open(TEST_INDICES_PATH) as f:
        test_idx = json.load(f)
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]
    logger.info(f"Test set cargado desde índices guardados: {len(X_test)} filas")

    threshold = _cargar_threshold()
    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cf = confusion_matrix(y_test, y_pred)

    fig_path = REPORTES_DIR / "confusion_matrix.png"
    make_confusion_matrix(
        cf,
        group_names=["TN", "FP", "FN", "TP"],
        categories=["Sin Infarto", "Infarto"],
        figsize=(8, 6),
        title=f"Matriz de Confusión — Prevención de Infartos (thr={threshold:.3f})",
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Matriz de confusión guardada en {fig_path}")

    recall = metricas.get("recall", 0)
    f1 = metricas.get("f1", 0)
    fallos = []
    if recall < RECALL_MINIMO:
        fallos.append(f"recall={recall:.4f} < mínimo={RECALL_MINIMO}")
    if f1 < F1_MINIMO:
        fallos.append(f"f1={f1:.4f} < piso={F1_MINIMO}")
    if fallos:
        logger.error("QUALITY GATE FALLIDO: " + " | ".join(fallos))
        sys.exit(1)

    logger.info(
        f"QUALITY GATE APROBADO: recall={recall:.4f} >= {RECALL_MINIMO} | "
        f"f1={f1:.4f} >= {F1_MINIMO}"
    )
    logger.info(f"Métricas finales: {metricas}")
    return metricas
