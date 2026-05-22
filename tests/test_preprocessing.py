import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import limpiar_datos, imputar_imc, winsorizar


@pytest.fixture
def df_base():
    return pd.DataFrame({
        "Edad": [35.0, 58.0, 62.0, 45.0],
        "Flag_hipertension": [0, 1, 0, 1],
        "Flag_problem_cardiaco": [0, 0, 0, 1],
        "Estados_civil": ["No", "Si", "No", "Si"],
        "Tipo_trabajo": ["cuidar_ninos", "Empresa_privada", "Empresa_privada", "Gobierno"],
        "Zona_residencia": ["Rural", "Urbano", "Urbano", "Rural"],
        "Promedio_nivel_glucosa": [95.12, 87.96, 110.89, 234.0],
        "IMC": [18.0, 39.2, np.nan, 28.5],
        "Flag_fumador": [None, "Nunca_fuma", None, "antes_fumaba"],
        "Genero": ["Hombre", "Hombre", "Mujer", "Mujer"],
        "Ataque_cardiaco": [0, 0, 0, 1],
        "ID": [1, 2, 3, 4],
    })


def test_limpiar_datos_edad_a_int(df_base):
    df = limpiar_datos(df_base)
    assert df["Edad"].dtype == int


def test_limpiar_datos_fumador_nulos_a_desconocido(df_base):
    df = limpiar_datos(df_base)
    assert df["Flag_fumador"].isna().sum() == 0
    assert "Desconocido" in df["Flag_fumador"].values


def test_imputar_imc_rellena_nulos(df_base):
    df, med = imputar_imc(df_base)
    assert df["IMC"].isna().sum() == 0
    assert med > 0


def test_imputar_imc_usa_mediana_externa(df_base):
    df, _ = imputar_imc(df_base, mediana=99.0)
    fila_nula = df_base["IMC"].isna()
    assert (df.loc[fila_nula, "IMC"] == 99.0).all()


def test_imputar_imc_sin_nulos_no_cambia(df_base):
    df_base["IMC"] = df_base["IMC"].fillna(25.0)
    df, med = imputar_imc(df_base)
    assert df["IMC"].equals(df_base["IMC"])


def test_winsorizar_no_cambia_columnas_no_incluidas(df_base):
    df_original = df_base.copy()
    df = winsorizar(df_base)
    assert df["Flag_hipertension"].equals(df_original["Flag_hipertension"])


def test_limpiar_datos_excluye_menores():
    df = pd.DataFrame({
        "Edad": [15.0, 30.0, 17.0, 50.0],
        "Flag_fumador": ["Nunca_fuma"] * 4,
        "Genero": ["Hombre"] * 4,
        "Ataque_cardiaco": [0, 1, 0, 1],
    })
    resultado = limpiar_datos(df)
    assert (resultado["Edad"] >= 18).all()
    assert len(resultado) == 2


def test_limpiar_datos_excluye_genero_other():
    df = pd.DataFrame({
        "Edad": [30.0, 45.0, 60.0],
        "Flag_fumador": ["Nunca_fuma"] * 3,
        "Genero": ["Hombre", "Other", "Mujer"],
        "Ataque_cardiaco": [0, 0, 1],
    })
    resultado = limpiar_datos(df)
    assert "Other" not in resultado["Genero"].values
    assert len(resultado) == 2
