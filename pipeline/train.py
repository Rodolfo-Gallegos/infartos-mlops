import logging
from src.train import entrenar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN_PIPELINE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ejecutar(X, y, scaler=None):
    logger.info("=== Paso 3: Entrenamiento + MLflow ===")
    metricas = entrenar(X, y, scaler=scaler)
    logger.info(f"Entrenamiento completado. Métricas: {metricas}")
    return metricas
