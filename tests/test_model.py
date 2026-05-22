import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
from config import (
    METRICS_PATH, MODEL_PATH, THRESHOLD_PATH, RECALL_MINIMO, F1_MINIMO,
)


def test_modelo_existe():
    assert MODEL_PATH.exists(), f"Modelo no encontrado en {MODEL_PATH}"


def test_metricas_existen():
    assert METRICS_PATH.exists(), f"Métricas no encontradas en {METRICS_PATH}"


def test_threshold_existe():
    assert THRESHOLD_PATH.exists(), f"threshold.json no encontrado en {THRESHOLD_PATH}"
    with open(THRESHOLD_PATH) as f:
        info = json.load(f)
    assert "threshold" in info
    assert 0.0 < info["threshold"] < 1.0


def test_quality_gate_recall():
    with open(METRICS_PATH) as f:
        metricas = json.load(f)
    recall = metricas.get("recall", 0)
    assert recall >= RECALL_MINIMO, (
        f"Recall={recall:.4f} está por debajo del mínimo={RECALL_MINIMO}"
    )


def test_quality_gate_f1_floor():
    with open(METRICS_PATH) as f:
        metricas = json.load(f)
    f1 = metricas.get("f1", 0)
    assert f1 >= F1_MINIMO, (
        f"F1={f1:.4f} está por debajo del piso={F1_MINIMO}"
    )


def test_modelo_cargable():
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    assert hasattr(modelo, "predict")
    assert hasattr(modelo, "predict_proba")


def test_metricas_tienen_campos_requeridos():
    with open(METRICS_PATH) as f:
        metricas = json.load(f)
    for campo in ["recall", "f1", "accuracy", "precision", "mcc", "threshold"]:
        assert campo in metricas, f"Campo '{campo}' faltante en metrics.json"


def test_modelo_es_lr_o_rf():
    with open(METRICS_PATH) as f:
        metricas = json.load(f)
    assert metricas.get("modelo") in {"LR_balanced", "RF_balanced"}, (
        f"Modelo inesperado: {metricas.get('modelo')}"
    )
