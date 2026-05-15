import logging
from src.features import construir_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | FEATURES_PIPELINE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ejecutar(df):
    logger.info("=== Paso 2: Feature Engineering ===")
    X, y, scaler, selector = construir_features(df, fit=True)
    logger.info(f"Features finales: {X.shape[1]} | Ejemplos: {X.shape[0]}")
    return X, y, scaler, selector
