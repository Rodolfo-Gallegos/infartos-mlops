# Prevención de Infartos - Modelo MLOps

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

Clasificación binaria del riesgo de infarto en asegurados, integrada en un
pipeline MLOps reproducible (lint, tests, train, quality gate, API, monitoreo).

**Entregables**: [`notebooks/analisis_exploratorio.ipynb`](notebooks/analisis_exploratorio.ipynb)
· [`docs/Presentacion_Prevencion_Infartos.pptx`](docs/Presentacion_Prevencion_Infartos.pptx)
· [Caso original](docs/Caso_Prevencion_Infartos.pptx)

**Métrica priorizada:** `Recall` (minimizar falsos negativos clínicos)
con `F1 ≥ 0.10` como piso operativo.

## Resultados del modelo en producción

| Métrica           | Valor    | Gate     |
|-------------------|----------|----------|
| Modelo            | `LR_balanced` | - |
| Threshold         | `0.465`  | aprendido por max-Recall con F1≥piso |
| Recall (test)     | **0.808** | ≥ 0.70 ✓ |
| F1 (test)         | **0.104** | ≥ 0.10 ✓ |
| Precision         | 0.056    | - |
| MCC               | 0.158    | > 0 |
| PR-AUC            | 0.090    | baseline 0.022 (4× sobre random) |
| Balanced Accuracy | 0.752    | > 0.5 ✓ |

Captura 128 de los 156 infartos del test set (7,170 pacientes, 2.18 % positivos).

## Estructura

```
infartos-mlops/
├── src/             # módulos: data_loader, preprocessing, features, train
├── pipeline/        # orquestación: ingest → features → train → evaluate
├── api/             # FastAPI: /health, /predecir
├── tests/           # pytest unitarios + smoke
├── utils/           # cf_matrix.py
├── notebooks/       # analisis_exploratorio.ipynb
├── .github/workflows/  # ml_pipeline.yml + monitoreo_drift.yml
├── config.py        # config centralizada
├── run_pipeline.py  # orquestador
├── Makefile
├── Dockerfile + Dockerfile.trainer
├── docker-compose.yml + docker-compose.preprod.yml
└── deploy.sh        # despliegue manual con rollback
```

## Quickstart

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Copiar dataset (separador ';')
cp "../Prevención_de_infartos/Dataset prevención de infartos.csv" data/dataset.csv

# 3. Pipeline completo (~12 s)
make train       # entrena, calcula threshold óptimo, valida gates
make validate    # quality gate doble (Recall ≥ 0.70 Y F1 ≥ 0.10)
make test        # 32 tests unitarios
make lint        # flake8 sin issues

# 4. Servir API
make docker      # build imagen
docker compose up -d
curl http://localhost:8000/health

# 5. Preprod (3 servicios: MLflow + trainer + API)
make preprod-up
make smoke

# 6. Monitoreo de drift
make monitoreo
```

## Decisiones de diseño clave

### Métrica: Recall, no F1
En salud cardiovascular, un **falso negativo** (paciente con riesgo real
no detectado) cuesta órdenes de magnitud más que un **falso positivo**
(consulta preventiva innecesaria). El piso `F1 ≥ 0.10` evita el modelo
trivial que alerta a todos.

### Sin SMOTE
SMOTE distorsiona los priors. Validado en EDA: XGBoost+SMOTE pasaba de
Recall 0.99 en CV a 0.22 fuera del fold. Se eligió `class_weight='balanced'`
en LR y RF, que no toca la distribución de probabilidades.

### Threshold operativo aprendido
`optimal_recall_with_f1_floor()` busca el threshold que maximiza Recall
sujeto a `F1 ≥ 0.10`. Se persiste en `artifacts/threshold.json` y lo
consume el predictor en producción.

### Features (top-10 por consenso de 4 métodos)
Edad, Flag_problem_cardiaco, Promedio_nivel_glucosa, Estados_civil,
Tipo_trabajo (Empresa_privada, Emprendedor), Flag_hipertension, Genero,
Flag_fumador (Nunca_fuma, antes_fumaba).

Métodos: Mann-Whitney, Point-Biserial, ANOVA, Mutual Information.

### Limpieza
- `Edad < 18` (n=7,539) excluidos - ruido clínico (prevalencia ~0.03 %).
- `Genero='Other'` (n=11) excluido - cero infartos en todo el dataset.
- `IMC` nulos imputados con mediana del train set.
- `Flag_fumador` nulos mapeados a `Desconocido` (categoría válida).
- Winsorización p1–p99 en Edad, IMC, Promedio_nivel_glucosa.

## Estrategia de uso (3 niveles)

| Nivel | Regla | Acción |
|-------|-------|--------|
| `ALTO_RIESGO` | `prob ≥ 0.465` | Chequeo cardiovascular + plan preventivo |
| `REVISAR`     | `0.200 ≤ prob < 0.465` | Campaña de salud por correo/teléfono |
| `BAJO_RIESGO` | `prob < 0.200` | Comunicación estándar, monitoreo pasivo |

## Stack tecnológico

- **ML**: scikit-learn (LogisticRegression, RandomForestClassifier),
  imbalanced-learn (descartado), MLflow (tracking + Model Registry)
- **Servicio**: FastAPI + Uvicorn, Pydantic v2, singleton predictor
- **Contenedores**: Docker, docker-compose (preprod con 3 servicios)
- **CI/CD**: GitHub Actions - lint + tests + train + quality gate + docker build
- **Monitoreo**: Evidently AI, cron semanal (lunes 8am UTC)
- **Calidad**: pytest (32 tests + smoke), flake8, coverage > 80 %

## Endpoints API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET    | `/`         | Info del servicio |
| GET    | `/health`   | Status + métricas + threshold operativo |
| POST   | `/predecir` | Predicción individual con decisión 3-niveles |
| GET    | `/docs`     | Swagger UI |

### Ejemplo de predicción

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "Genero": "Hombre",
    "Edad": 65,
    "Flag_hipertension": 1,
    "Flag_problem_cardiaco": 1,
    "Estados_civil": "Si",
    "Tipo_trabajo": "Empresa_privada",
    "Zona_residencia": "Urbano",
    "Promedio_nivel_glucosa": 220.5,
    "IMC": 32.1,
    "Flag_fumador": "antes_fumaba"
  }'

# → {"probabilidad": 0.9413, "decision": "ALTO_RIESGO", "nivel_riesgo": "alto"}
```

## Comandos Make

| Target | Descripción |
|--------|-------------|
| `make lint`        | flake8 |
| `make test`        | pytest unitarios + coverage |
| `make train`       | pipeline completo (ingest → features → train → evaluate) |
| `make validate`    | quality gate doble (Recall ≥ 0.70 y F1 ≥ 0.10) |
| `make docker`      | construir imagen API |
| `make preprod-up`  | levantar stack de 3 servicios |
| `make preprod-down`| bajar stack y limpiar volúmenes |
| `make smoke`       | smoke tests contra servicios reales |
| `make clean`       | limpiar artifacts y caches |

## Despliegue

### Local
```bash
docker compose up -d
```

### Pre-producción (3 servicios)
```bash
make preprod-up      # MLflow + trainer + API
make smoke           # validación
```

### Local con rollback (`deploy.sh`)
```bash
make deploy VERSION=v1.0.0    # build → up → smoke → tag → verify (rollback si falla)
```

### Cloud Run (`deploy_gcp.sh`)
```bash
gcloud auth login
gcloud config set project mi-proyecto-gcp
make deploy-gcp VERSION=v1.0.0
```
Requiere Artifact Registry repo `infartos-mlops` en `us-central1` y APIs
`artifactregistry.googleapis.com` + `run.googleapis.com` habilitadas.

## Data versioning con DVC

El dataset (`data/dataset.csv`) está versionado con DVC: git tracking solo
el `.dvc` (con md5 hash) y los datos guardados aparte.

```bash
make dvc-pull    # trae la versión actual del dataset
make dvc-push    # publica nuevos cambios
```

Hash actual del dataset: `d6514153cd4ffc9791c36e3f05ebbbb3`

## Cumple los 8+1 niveles del curso MLOps-SDC

| Nivel | Tema | Estado |
|-------|------|--------|
| 1 | config.py + módulos + logging + Makefile | ✅ |
| 2 | MLflow tracking + Model Registry + DVC    | ✅ |
| 3 | FastAPI /health + /predecir + singleton   | ✅ |
| 4 | Docker + compose 3 servicios              | ✅ |
| 5 | Makefile lint/test/train/validate/preprod | ✅ |
| 6 | setup.cfg + pytest + smoke tests          | ✅ |
| 7 | GitHub Actions 3 jobs + quality gates     | ✅ |
| 8 | EvidentlyAI + drift + cron semanal        | ✅ |
| 9 | GCP Cloud Run deployment                  | ✅ |
