---
title: Infartos MLOps
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Prevención de Infartos - Modelo MLOps

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Rodolfo-Gallegos/infartos-mlops)

**Demo en vivo:** https://rodolfo-gallegos-infartos-mlops.hf.space
· [Swagger](https://rodolfo-gallegos-infartos-mlops.hf.space/docs)
· [Health](https://rodolfo-gallegos-infartos-mlops.hf.space/health)

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
├── api/             # FastAPI: /, /health, /predecir (+ auth, CORS, rate-limit)
├── tests/           # pytest unitarios (35) + smoke
├── notebooks/       # analisis_exploratorio.ipynb
├── .github/workflows/  # ml_pipeline.yml (lint→train→docker→deploy-hf) + monitoreo_drift.yml
├── config.py        # config centralizada
├── run_pipeline.py  # orquestador
├── Makefile
├── Dockerfile + Dockerfile.trainer
├── docker-compose.yml + docker-compose.preprod.yml
├── deploy.sh        # despliegue local con rollback
├── deploy_hf.sh     # push a Hugging Face Spaces (orphan snapshot)
└── deploy_gcp.sh    # push a Cloud Run (opcional)
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
make test        # 35 tests unitarios (incluye auth)
make lint        # flake8 sin issues

# 4. Servir API
make docker
docker compose up -d
curl http://localhost:8000/health

# 5. Preprod (MLflow + trainer + API)
make preprod-up
make smoke

# 6. Desplegar a Hugging Face Spaces
export HF_USER=... HF_TOKEN=hf_...
make deploy-hf
```

## Decisiones de diseño clave

### Métrica: Recall, no F1
En salud cardiovascular, un **falso negativo** (paciente con riesgo real
no detectado) cuesta órdenes de magnitud más que un **falso positivo**
(consulta preventiva innecesaria). El piso `F1 ≥ 0.10` evita el modelo
trivial que alerta a todos.

### Threshold operativo aprendido
`optimal_recall_with_f1_floor()` busca el threshold que maximiza Recall
sujeto a `F1 ≥ 0.10`. Se persiste en `artifacts/threshold.json` y lo
consume el predictor en producción.

## Estrategia de uso (3 niveles)

| Nivel | Regla | Acción |
|-------|-------|--------|
| `ALTO_RIESGO` | `prob ≥ 0.465` | Chequeo cardiovascular + plan preventivo |
| `REVISAR`     | `0.200 ≤ prob < 0.465` | Campaña de salud por correo/teléfono |
| `BAJO_RIESGO` | `prob < 0.200` | Comunicación estándar, monitoreo pasivo |

## Endpoints API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET    | `/`         | Info del servicio |
| GET    | `/health`   | Status + métricas + threshold operativo |
| POST   | `/predecir` | Predicción individual con decisión 3-niveles |
| GET    | `/docs`     | Swagger UI |

### Ejemplo de predicción (contra el Space en vivo)

```bash
curl -X POST https://rodolfo-gallegos-infartos-mlops.hf.space/predecir \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
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

# → {"probabilidad": 0.8667, "decision": "ALTO_RIESGO", "nivel_riesgo": "alto"}
```

En local el header `X-API-Key` solo es obligatorio si exportas `API_KEY` antes
de arrancar la API.

### Seguridad y variables de entorno

| Var | Default | Para qué |
|---|---|---|
| `API_KEY`                  | _(vacío)_ | Si está set, `/predecir` exige header `X-API-Key`. Vacío = auth OFF (dev/CI). |
| `ALLOWED_ORIGINS`          | localhost | CORS exact-match, coma-separadas. |
| `ALLOWED_ORIGIN_REGEX`     | `^https://rodolfo-gallegos-infartos-mlops\.hf\.space$` | CORS por regex. Cámbialo si forkeas. |
| `PREDICT_RATE_LIMIT`       | `30/minute` | Límite por IP en `/predecir`. |
| `RATE_LIMIT_STORAGE_URI`   | `memory://` | `redis://...` para que los límites persistan entre reinicios (Upstash free tier). |

En HF Spaces: **Settings → Variables and secrets** → agregar `API_KEY` como secret.

## Despliegue

```bash
# Local
docker compose up -d

# Local con rollback automático
make deploy VERSION=v1.0.0

# Hugging Face Spaces (recomendado, gratis, sin tarjeta)
export HF_USER=tu_usuario_hf
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
make deploy-hf

# Auto-deploy desde GitHub Actions:
#   Settings → Secrets and variables → Actions → New repository secret
#     HF_USER  = tu_usuario_hf
#     HF_TOKEN = hf_xxxxxxxxxxxxxxxx
#   Luego: Actions → ML Pipeline CI/CD → Run workflow
#   (Trigger manual a propósito: el modelo del CI es sintético; producción
#   con dataset real se hace local con `make deploy-hf`)

# Cloud Run (opcional, requiere proyecto GCP con billing)
gcloud auth login && gcloud config set project mi-proyecto-gcp
make deploy-gcp VERSION=v1.0.0
```

**Hugging Face Spaces** (setup único): crear cuenta → `New Space` con
SDK=Docker → token con permiso `write` en
[settings/tokens](https://huggingface.co/settings/tokens). El script
arma un snapshot temporal con los `artifacts/` incluidos (que el
`.gitignore` excluye en el repo) y lo empuja al Space. Build ~3-6 min.

**Cloud Run** requiere Artifact Registry repo `infartos-mlops` en
`us-central1` y APIs `artifactregistry.googleapis.com` + `run.googleapis.com`.

## Data versioning con DVC

El dataset (`data/dataset.csv`) está versionado con DVC: git solo trackea
el `.dvc` (con md5 hash), los datos van aparte.

```bash
make dvc-pull
make dvc-push
```

## Cumple los 8+1 niveles del curso MLOps-SDC

| Nivel | Tema | Estado |
|-------|------|--------|
| 1 | config.py + módulos + logging + Makefile | ✅ |
| 2 | MLflow tracking + Model Registry + DVC    | ✅ |
| 3 | FastAPI /health + /predecir + singleton   | ✅ |
| 4 | Docker + compose 3 servicios              | ✅ |
| 5 | Makefile lint/test/train/validate/preprod | ✅ |
| 6 | setup.cfg + pytest (35) + smoke tests     | ✅ |
| 7 | GitHub Actions 4 jobs + quality gates     | ✅ |
| 8 | EvidentlyAI + drift + cron semanal        | ✅ |
| 9 | Deploy en la nube (Hugging Face Spaces, opc. Cloud Run) | ✅ |

### Endurecimientos extra (más allá del curso)

| Tema | Cómo |
|------|------|
| **Auth** | Header `X-API-Key` en `/predecir`, validado contra env `API_KEY`. Auth OFF si no está set (dev/CI). |
| **CORS** | Regex restrictivo al dominio del Space; localhost para dev. |
| **Rate limiting** | `slowapi` 30/min por IP en `/predecir`. Storage en memoria por default, configurable a Redis vía `RATE_LIMIT_STORAGE_URI`. |
| **CI/CD** | Job `deploy-hf` en GitHub Actions (workflow_dispatch) que descarga artifacts y empuja al Space con secrets. |
| **Errors** | `/predecir` devuelve mensajes genéricos al cliente, stack traces solo en logs. |

### Lo que faltaría para "producción clínica"

- Logging remoto (Sentry/Logflare) en vez de stdout.
- Rate limiter con Redis efectivo (hoy es opt-in vía env).
- Auth más robusta (OAuth/JWT por usuario, no un API key compartido).
- DVC remote en S3/GCS para que `dvc pull` funcione desde CI.
- Trigger de retraining desde el monitor de drift (hoy solo reporta).
