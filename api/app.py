import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from api.predictor import predictor
from api.schemas import PacienteInput, PrediccionOutput, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | API | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VERSION = os.getenv("PIPELINE_VERSION", "1.0.0")

# CORS: lista exacta + regex. Por defecto solo localhost y subdominios *.hf.space.
# Override con env vars (coma-separadas para ALLOWED_ORIGINS).
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX", r"^https://[a-z0-9\-]+\.hf\.space$"
)

PREDICT_RATE_LIMIT = os.getenv("PREDICT_RATE_LIMIT", "30/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[])


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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
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
@limiter.limit(PREDICT_RATE_LIMIT)
def predecir(request: Request, paciente: PacienteInput):
    try:
        resultado = predictor.predecir(paciente.model_dump())
        logger.info(
            f"Predicción: prob={resultado['probabilidad']} | "
            f"decisión={resultado['decision']}"
        )
        return PrediccionOutput(**resultado)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail="Error interno en la predicción")
