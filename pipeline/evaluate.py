import logging
import sys
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.cf_matrix import make_confusion_matrix
from config import (
    MODEL_PATH, METRICS_PATH, REPORTES_DIR,
    RECALL_MINIMO, TARGET, TEST_SIZE, RANDOM_STATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | EVALUATE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ejecutar(X, y, metricas: dict):
    logger.info("=== Paso 4: Evaluación + Quality Gate ===")

    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    y_pred = modelo.predict(X_test)
    cf = confusion_matrix(y_test, y_pred)

    fig_path = REPORTES_DIR / "confusion_matrix.png"
    make_confusion_matrix(
        cf,
        group_names=["TN", "FP", "FN", "TP"],
        categories=["Sin Infarto", "Infarto"],
        figsize=(8, 6),
        title="Matriz de Confusión — Prevención de Infartos",
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Matriz de confusión guardada en {fig_path}")

    recall = metricas.get("recall", 0)
    if recall < RECALL_MINIMO:
        logger.error(
            f"QUALITY GATE FALLIDO: recall={recall:.4f} < mínimo={RECALL_MINIMO}"
        )
        sys.exit(1)

    logger.info(f"QUALITY GATE APROBADO: recall={recall:.4f} >= {RECALL_MINIMO}")
    logger.info(f"Métricas finales: {metricas}")
    return metricas
