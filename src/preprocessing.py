import logging
import numpy as np
import pandas as pd
from scipy.stats import mstats
from config import (
    COLUMNAS_WINSORIZAR, P_WINSOR_LOW, P_WINSOR_HIGH, TARGET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PREPROCESSING | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_imc_mediana = None


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Edad"] = df["Edad"].astype(int)

    # Flag_fumador: NaN → categoría "Desconocido" (30% nulos, no imputar)
    nulos_fumador = df["Flag_fumador"].isna().sum()
    df["Flag_fumador"] = df["Flag_fumador"].fillna("Desconocido")
    logger.info(f"Flag_fumador: {nulos_fumador} nulos → 'Desconocido'")

    return df


def winsorizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUMNAS_WINSORIZAR:
        if col in df.columns:
            lo = df[col].quantile(P_WINSOR_LOW)
            hi = df[col].quantile(P_WINSOR_HIGH)
            df[col] = df[col].clip(lower=lo, upper=hi)
            logger.info(f"Winsorización aplicada a '{col}' [{lo:.2f}, {hi:.2f}]")
    return df


def imputar_imc(df: pd.DataFrame, mediana: float = None) -> tuple[pd.DataFrame, float]:
    global _imc_mediana
    df = df.copy()
    nulos = df["IMC"].isna().sum()
    if nulos == 0:
        med = mediana or df["IMC"].median()
        return df, med

    if mediana is not None:
        med = mediana
    else:
        med = df["IMC"].median()
        _imc_mediana = med

    df["IMC"] = df["IMC"].fillna(med)
    logger.info(f"IMC: {nulos} nulos imputados con mediana={med:.2f}")
    return df, med


def preprocesar(df: pd.DataFrame, mediana_imc: float = None) -> tuple[pd.DataFrame, float]:
    df = limpiar_datos(df)
    df, med = imputar_imc(df, mediana_imc)
    df = winsorizar(df)
    logger.info("Preprocesamiento completado")
    return df, med
