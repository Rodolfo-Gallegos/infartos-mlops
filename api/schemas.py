from typing import Literal, Optional
from pydantic import BaseModel, Field


class PacienteInput(BaseModel):
    Genero: Literal["Hombre", "Mujer"]
    Edad: float = Field(ge=0, le=120, description="Edad en años")
    Flag_hipertension: Literal[0, 1]
    Flag_problem_cardiaco: Literal[0, 1]
    Estados_civil: Literal["Si", "No"]
    Tipo_trabajo: str = Field(
        description="Ej: Empresa_privada, Emprendedor, Gobierno, cuidar_ninos"
    )
    Zona_residencia: Literal["Urbano", "Rural"]
    Promedio_nivel_glucosa: float = Field(ge=0, description="Nivel de glucosa en sangre")
    IMC: float = Field(ge=0, le=100, description="Índice de Masa Corporal")
    Flag_fumador: Optional[str] = Field(
        default=None,
        description="Nunca_fuma | antes_fumaba | fuma | Desconocido (None = Desconocido)"
    )

    model_config = {"json_schema_extra": {"example": {
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
    }}}


class PrediccionOutput(BaseModel):
    probabilidad: float = Field(description="Probabilidad de infarto (0-1)")
    decision: str = Field(description="ALTO_RIESGO | REVISAR | BAJO_RIESGO")
    nivel_riesgo: str = Field(description="alto | medio | bajo")


class HealthResponse(BaseModel):
    status: str
    modelo: str
    recall: float
    f1: float
    precision: float
    threshold: float
    version: str
