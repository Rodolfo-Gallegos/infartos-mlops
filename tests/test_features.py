import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest
from src.features import codificar_binarias, codificar_categoricas, construir_features
from config import K_FEATURES


@pytest.fixture
def df_limpio():
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 5] * 4,
        "Genero": ["Hombre", "Mujer", "Hombre", "Mujer", "Hombre"] * 4,
        "Edad": [30, 45, 60, 25, 70] * 4,
        "Flag_hipertension": [0, 1, 0, 0, 1] * 4,
        "Flag_problem_cardiaco": [0, 0, 1, 0, 1] * 4,
        "Estados_civil": ["Si", "No", "Si", "No", "Si"] * 4,
        "Tipo_trabajo": ["Empresa_privada", "Emprendedor", "Gobierno",
                         "cuidar_ninos", "Empresa_privada"] * 4,
        "Zona_residencia": ["Urbano", "Rural", "Urbano", "Rural", "Urbano"] * 4,
        "Promedio_nivel_glucosa": [80.0, 150.0, 200.0, 90.0, 220.0] * 4,
        "IMC": [22.0, 30.0, 35.0, 18.0, 28.0] * 4,
        "Flag_fumador": ["Nunca_fuma", "antes_fumaba", "fuma", "Desconocido", "Nunca_fuma"] * 4,
        "Ataque_cardiaco": [0, 0, 1, 0, 1] * 4,
    })


def test_codificar_binarias_genero(df_limpio):
    df = codificar_binarias(df_limpio)
    assert set(df["Genero"].unique()).issubset({0, 1})


def test_zona_residencia_excluida_de_features(df_limpio):
    X, y, scaler, selector = construir_features(df_limpio, fit=True)
    assert "Zona_residencia" not in X.columns


def test_codificar_categoricas_crea_dummies(df_limpio):
    df = codificar_binarias(df_limpio)
    df = codificar_categoricas(df)
    assert "Tipo_trabajo" not in df.columns
    assert any("Tipo_trabajo_" in c for c in df.columns)


def test_construir_features_shape(df_limpio):
    X, y, scaler, selector = construir_features(df_limpio, fit=True)
    assert X.shape[1] == K_FEATURES
    assert len(y) == len(X)


def test_construir_features_sin_target_ni_id(df_limpio):
    X, y, scaler, selector = construir_features(df_limpio, fit=True)
    assert "Ataque_cardiaco" not in X.columns
    assert "ID" not in X.columns
