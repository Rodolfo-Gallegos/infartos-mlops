import logging
import pandas as pd
from config import (
    COLUMNAS_WINSORIZAR, P_WINSOR_LOW, P_WINSOR_HIGH,
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

    nulos_fumador = df["Flag_fumador"].isna().sum()
    df["Flag_fumador"] = df["Flag_fumador"].fillna("Desconocido")
    logger.info(f"Flag_fumador: {nulos_fumador} nulos → 'Desconocido'")

    # Genero='Other': 0 infartos en todo el dataset → no aporta señal predictiva
    n_other = (df["Genero"] == "Other").sum()
    if n_other > 0:
        df = df[df["Genero"] != "Other"].reset_index(drop=True)
        logger.info(f"Genero='Other': {n_other} filas excluidas (0 infartos)")

    # Menores de 18: prevalencia de infarto ~0.03% (2/7539) → ruido clínico
    n_menores = (df["Edad"] < 18).sum()
    if n_menores > 0:
        df = df[df["Edad"] >= 18].reset_index(drop=True)
        logger.info(f"Edad < 18: {n_menores} filas excluidas (ruido clínico)")

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
