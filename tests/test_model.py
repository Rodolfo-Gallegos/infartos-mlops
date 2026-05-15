import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import pytest
from config import METRICS_PATH, MODEL_PATH, RECALL_MINIMO


def test_modelo_existe():
    assert MODEL_PATH.exists(), f"Modelo no encontrado en {MODEL_PATH}"


def test_metricas_existen():
    assert METRICS_PATH.exists(), f"Métricas no encontradas en {METRICS_PATH}"


def test_quality_gate_recall():
    assert METRICS_PATH.exists(), "Ejecuta el pipeline primero"
    with open(METRICS_PATH) as f:
        metricas = json.load(f)
    recall = metricas.get("recall", 0)
    assert recall >= RECALL_MINIMO, (
        f"Recall={recall:.4f} está por debajo del mínimo={RECALL_MINIMO}"
    )


def test_modelo_cargable():
    assert MODEL_PATH.exists(), "Ejecuta el pipeline primero"
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    assert hasattr(modelo, "predict")
    assert hasattr(modelo, "predict_proba")


def test_metricas_tienen_campos_requeridos():
    assert METRICS_PATH.exists(), "Ejecuta el pipeline primero"
    with open(METRICS_PATH) as f:
        metricas = json.load(f)
    for campo in ["recall", "f1", "accuracy", "precision", "roc_auc"]:
        assert campo in metricas, f"Campo '{campo}' faltante en metrics.json"
