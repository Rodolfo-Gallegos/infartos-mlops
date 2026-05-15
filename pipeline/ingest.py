import logging
from src.data_loader import cargar_datos, eliminar_id
from src.preprocessing import preprocesar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | INGEST | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ejecutar(path=None):
    logger.info("=== Paso 1: Ingest + Preprocesamiento ===")
    df = cargar_datos(path)
    df, mediana_imc = preprocesar(df)
    logger.info(f"Dataset limpio: {df.shape}")
    return df, mediana_imc
