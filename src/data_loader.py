import logging
import pandas as pd
from config import RAW_DATA, SEP, COLUMNAS_REQUERIDAS, ID_COL, TARGET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | DATA_LOADER | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cargar_datos(path=None) -> pd.DataFrame:
    ruta = path or RAW_DATA
    logger.info(f"Cargando dataset desde {ruta}")
    df = pd.read_csv(ruta, sep=SEP)
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el dataset: {faltantes}")

    logger.info(f"Target '{TARGET}': {df[TARGET].value_counts().to_dict()}")
    return df


def eliminar_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[ID_COL], errors="ignore")
