import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from config import (
    TARGET, ID_COL, COLUMNAS_NUMERICAS, K_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | FEATURES | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MAPA_BINARIO = {
    "Genero": {"Hombre": 0, "Mujer": 1},
    "Estados_civil": {"No": 0, "Si": 1},
}
# Zona_residencia excluida: no es predictor cardiovascular establecido (EDA)


def codificar_binarias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapa in MAPA_BINARIO.items():
        if col in df.columns:
            mapeado = df[col].map(mapa)
            nulos_post = mapeado.isna().sum()
            if nulos_post > 0:
                logger.warning(f"'{col}': {nulos_post} valores no mapeados → 0")
            df[col] = mapeado.fillna(0).astype(int)
            logger.info(f"'{col}' codificada con mapa binario")
    return df


def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_ohe = ["Tipo_trabajo", "Flag_fumador"]
    cols_presentes = [c for c in cols_ohe if c in df.columns]
    df = pd.get_dummies(df, columns=cols_presentes, drop_first=False)
    logger.info(f"OHE aplicado a: {cols_presentes}. Shape resultante: {df.shape}")
    return df


def escalar(df: pd.DataFrame, scaler: StandardScaler = None) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    cols = [c for c in COLUMNAS_NUMERICAS if c in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
        logger.info(f"StandardScaler fit+transform en: {cols}")
    else:
        df[cols] = scaler.transform(df[cols])
        logger.info(f"StandardScaler transform (pre-fit) en: {cols}")
    return df, scaler


def seleccionar_features(
    X: pd.DataFrame,
    y: pd.Series,
    selector: SelectKBest = None,
) -> tuple[pd.DataFrame, SelectKBest]:
    if selector is None:
        selector = SelectKBest(f_classif, k=K_FEATURES)
        X_sel = selector.fit_transform(X, y)
        features = X.columns[selector.get_support()].tolist()
        logger.info(f"SelectKBest seleccionó {K_FEATURES} features: {features}")
    else:
        X_sel = selector.transform(X)
        features = X.columns[selector.get_support()].tolist()

    return pd.DataFrame(X_sel, columns=features, index=X.index), selector


def construir_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    selector: SelectKBest = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler, SelectKBest]:
    df = codificar_binarias(df)
    df = codificar_categoricas(df)

    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL, "Zona_residencia"], errors="ignore")
    X = X.select_dtypes(exclude=["object"])

    X, scaler = escalar(X, scaler if not fit else None)

    if fit:
        X, selector = seleccionar_features(X, y, None)
    else:
        X, selector = seleccionar_features(X, y, selector)

    return X, y, scaler, selector
