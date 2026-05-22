import json
import logging
import pickle
import pandas as pd
from config import (
    MODEL_PATH, SCALER_PATH, METRICS_PATH, FEATURES_PATH, THRESHOLD_PATH,
    UMBRAL_ALTO_RIESGO, UMBRAL_MEDIO_RIESGO, COLUMNAS_NUMERICAS,
)
from src.features import codificar_binarias, codificar_categoricas

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.metricas = {}
        self.features = []
        self.umbral_alto = UMBRAL_ALTO_RIESGO
        self.umbral_medio = UMBRAL_MEDIO_RIESGO
        self._cargado = False

    def cargar(self):
        with open(MODEL_PATH, "rb") as f:
            self.modelo = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(METRICS_PATH) as f:
            self.metricas = json.load(f)
        self.features = pd.read_csv(FEATURES_PATH)["feature"].tolist()

        # Threshold operativo aprendido en entrenamiento (max Recall con piso F1).
        # REVISAR queda en [umbral_alto/2, umbral_alto); si quedara por encima del
        # default fijo, prevalece el default fijo como piso de "medio".
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH) as f:
                thr_info = json.load(f)
            self.umbral_alto = float(thr_info["threshold"])
            self.umbral_medio = min(UMBRAL_MEDIO_RIESGO, self.umbral_alto * 0.5)

        self._cargado = True
        logger.info(
            f"Modelo '{self.metricas.get('modelo')}' cargado | "
            f"Recall={self.metricas.get('recall')} F1={self.metricas.get('f1')} | "
            f"thr_alto={self.umbral_alto:.3f} thr_medio={self.umbral_medio:.3f} | "
            f"Features={len(self.features)}"
        )

    def predecir(self, datos: dict) -> dict:
        if not self._cargado:
            raise RuntimeError("Modelo no cargado. Llama a cargar() primero.")

        df = pd.DataFrame([datos])
        df["Flag_fumador"] = df["Flag_fumador"].fillna("Desconocido")
        df["Edad"] = df["Edad"].astype(int)

        df = codificar_binarias(df)
        df = codificar_categoricas(df)

        cols_num = [c for c in COLUMNAS_NUMERICAS if c in df.columns]
        df[cols_num] = self.scaler.transform(df[cols_num])

        df = df.reindex(columns=self.features, fill_value=0)

        proba = float(self.modelo.predict_proba(df)[0][1])

        if proba >= self.umbral_alto:
            decision, nivel = "ALTO_RIESGO", "alto"
        elif proba >= self.umbral_medio:
            decision, nivel = "REVISAR", "medio"
        else:
            decision, nivel = "BAJO_RIESGO", "bajo"

        return {
            "probabilidad": round(proba, 4),
            "decision": decision,
            "nivel_riesgo": nivel,
        }


predictor = Predictor()
