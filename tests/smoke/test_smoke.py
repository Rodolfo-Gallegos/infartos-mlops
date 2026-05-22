import httpx

BASE_URL = "http://localhost:8000"

PACIENTE_TEST = {
    "Genero": "Hombre",
    "Edad": 65,
    "Flag_hipertension": 1,
    "Flag_problem_cardiaco": 1,
    "Estados_civil": "Si",
    "Tipo_trabajo": "Empresa_privada",
    "Zona_residencia": "Urbano",
    "Promedio_nivel_glucosa": 220.5,
    "IMC": 32.1,
    "Flag_fumador": "antes_fumaba",
}


def test_health_smoke():
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_prediccion_smoke():
    r = httpx.post(f"{BASE_URL}/predecir", json=PACIENTE_TEST, timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "probabilidad" in data
    assert data["decision"] in {"ALTO_RIESGO", "REVISAR", "BAJO_RIESGO"}


def test_docs_smoke():
    r = httpx.get(f"{BASE_URL}/docs", timeout=10)
    assert r.status_code == 200
