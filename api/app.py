import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.predictor import predictor
from api.schemas import PacienteInput, PrediccionOutput, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | API | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VERSION = os.getenv("PIPELINE_VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando API - cargando modelo...")
    predictor.cargar()
    logger.info("Modelo cargado. API lista.")
    yield
    logger.info("Apagando API.")


app = FastAPI(
    title="API Prevención de Infartos",
    description="Predice el riesgo de infarto cardíaco en asegurados.",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def info():
    return {
        "servicio": "API Prevención de Infartos",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
        "prediccion": "/predecir",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    if not predictor._cargado:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    m = predictor.metricas
    return HealthResponse(
        status="ok",
        modelo=m.get("modelo", "desconocido"),
        recall=m.get("recall", 0),
        f1=m.get("f1", 0),
        precision=m.get("precision", 0),
        threshold=predictor.umbral_alto,
        version=VERSION,
    )


@app.post("/predecir", response_model=PrediccionOutput, tags=["Predicción"])
def predecir(paciente: PacienteInput):
    try:
        resultado = predictor.predecir(paciente.model_dump())
        logger.info(
            f"Predicción: prob={resultado['probabilidad']} | "
            f"decisión={resultado['decision']}"
        )
        return PrediccionOutput(**resultado)
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))
