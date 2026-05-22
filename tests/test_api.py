import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from api.app import app

PACIENTE_ALTO_RIESGO = {
    "Genero": "Hombre",
    "Edad": 75,
    "Flag_hipertension": 1,
    "Flag_problem_cardiaco": 1,
    "Estados_civil": "Si",
    "Tipo_trabajo": "Empresa_privada",
    "Zona_residencia": "Urbano",
    "Promedio_nivel_glucosa": 250.0,
    "IMC": 38.5,
    "Flag_fumador": "fuma",
}

PACIENTE_BAJO_RIESGO = {
    "Genero": "Mujer",
    "Edad": 28,
    "Flag_hipertension": 0,
    "Flag_problem_cardiaco": 0,
    "Estados_civil": "No",
    "Tipo_trabajo": "Empresa_privada",
    "Zona_residencia": "Urbano",
    "Promedio_nivel_glucosa": 82.0,
    "IMC": 21.0,
    "Flag_fumador": "Nunca_fuma",
}


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "recall" in data
    assert "f1" in data
    assert "precision" in data
    assert "threshold" in data
    assert "modelo" in data
    assert 0.0 < data["threshold"] < 1.0
    assert data["modelo"] in {"LR_balanced", "RF_balanced"}


def test_info_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "prediccion" in r.json()


def test_prediccion_estructura(client):
    r = client.post("/predecir", json=PACIENTE_ALTO_RIESGO)
    assert r.status_code == 200
    data = r.json()
    assert "probabilidad" in data
    assert "decision" in data
    assert "nivel_riesgo" in data


def test_prediccion_probabilidad_rango(client):
    r = client.post("/predecir", json=PACIENTE_ALTO_RIESGO)
    prob = r.json()["probabilidad"]
    assert 0.0 <= prob <= 1.0


def test_prediccion_decision_valida(client):
    r = client.post("/predecir", json=PACIENTE_ALTO_RIESGO)
    assert r.json()["decision"] in {"ALTO_RIESGO", "REVISAR", "BAJO_RIESGO"}


def test_prediccion_determinismo(client):
    r1 = client.post("/predecir", json=PACIENTE_ALTO_RIESGO)
    r2 = client.post("/predecir", json=PACIENTE_ALTO_RIESGO)
    assert r1.json()["probabilidad"] == r2.json()["probabilidad"]


def test_validacion_edad_invalida(client):
    payload = {**PACIENTE_BAJO_RIESGO, "Edad": -5}
    r = client.post("/predecir", json=payload)
    assert r.status_code == 422


def test_validacion_imc_invalido(client):
    payload = {**PACIENTE_BAJO_RIESGO, "IMC": 200}
    r = client.post("/predecir", json=payload)
    assert r.status_code == 422


def test_validacion_genero_invalido(client):
    payload = {**PACIENTE_BAJO_RIESGO, "Genero": "X"}
    r = client.post("/predecir", json=payload)
    assert r.status_code == 422


def test_fumador_none_es_valido(client):
    payload = {**PACIENTE_BAJO_RIESGO, "Flag_fumador": None}
    r = client.post("/predecir", json=payload)
    assert r.status_code == 200


def test_swagger_disponible(client):
    r = client.get("/docs")
    assert r.status_code == 200
