import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTES_DIR = ROOT_DIR / "reportes"

RAW_DATA = DATA_DIR / "dataset.csv"
MODEL_PATH = ARTIFACTS_DIR / "modelo.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURES_PATH = ARTIFACTS_DIR / "features.csv"

SEP = ";"
ID_COL = "ID"
TARGET = "Ataque_cardiaco"
COLUMNAS_REQUERIDAS = [
    ID_COL, "Genero", "Edad", "Flag_hipertension",
    "Flag_problem_cardiaco", "Estados_civil", "Tipo_trabajo",
    "Zona_residencia", "Promedio_nivel_glucosa", "IMC",
    "Flag_fumador", TARGET,
]

COLUMNAS_WINSORIZAR = ["Promedio_nivel_glucosa", "IMC", "Edad"]
P_WINSOR_LOW = 0.01
P_WINSOR_HIGH = 0.99
COLUMNAS_NUMERICAS = ["Edad", "Promedio_nivel_glucosa", "IMC"]
COLUMNAS_CATEGORICAS = [
    "Genero", "Estados_civil", "Tipo_trabajo",
    "Zona_residencia", "Flag_fumador",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
TECNICA_BALANCEO = "smote"
K_FEATURES = 10

RECALL_MINIMO = 0.70

MLFLOW_EXPERIMENT = "infartos-prevencion"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(ROOT_DIR / "mlruns"))
MLFLOW_RUN_NAME = os.getenv("PIPELINE_VERSION", "local")

UMBRAL_ALTO_RIESGO = 0.40
UMBRAL_MEDIO_RIESGO = 0.20

for d in [DATA_DIR, ARTIFACTS_DIR, REPORTES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
