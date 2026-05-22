# Prevención de Infartos - Modelo MLOps

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Rodolfo-Gallegos/infartos-mlops)

**Demo:** https://rodolfo-gallegos-infartos-mlops.hf.space ·
[Swagger](https://rodolfo-gallegos-infartos-mlops.hf.space/docs)

Clasificación binaria de riesgo de infarto en asegurados con pipeline MLOps
reproducible (lint, tests, train, quality gate, API, monitoreo, despliegue).
**Métrica priorizada:** `Recall` (minimizar falsos negativos clínicos) con
`F1 ≥ 0.10` como piso.

## Resultados

| Métrica           | Valor    | Gate     |
|-------------------|----------|----------|
| Modelo            | `LR_balanced` | — |
| Threshold         | `0.465`  | learned (max-Recall con F1 ≥ piso) |
| Recall (test)     | **0.808** | ≥ 0.70 ✓ |
| F1 (test)         | **0.104** | ≥ 0.10 ✓ |
| PR-AUC            | 0.090    | 4× sobre random (baseline 0.022) |
| Balanced Accuracy | 0.752    | > 0.5 ✓ |

Captura **128 de 156 infartos** del test set (7,170 pacientes, 2.18 % positivos).

## Estructura

```
src/             módulos: data_loader, preprocessing, features, train
pipeline/        orquestación: ingest → features → train → evaluate
api/             FastAPI: /, /health, /predecir (auth, CORS, rate-limit)
tests/           pytest unitarios (35) + smoke
.github/workflows/  ml_pipeline.yml + monitoreo_drift.yml (cron semanal)
Dockerfile, docker-compose.yml, Makefile, deploy_hf.sh, deploy_gcp.sh
```

## Quickstart

```bash
pip install -r requirements.txt
make train test lint validate     # pipeline + 35 tests + flake8 + quality gates

# Local
docker compose up -d
curl http://localhost:8000/health

# Deploy a HF Spaces
export HF_USER=tu_usuario HF_TOKEN=hf_xxx
make deploy-hf
```

## Estrategia de uso (3 niveles)

| Nivel | Regla | Acción |
|-------|-------|--------|
| `ALTO_RIESGO` | `prob ≥ 0.465`       | Chequeo cardiovascular + plan preventivo |
| `REVISAR`     | `0.200 ≤ prob < 0.465` | Campaña de salud por correo/teléfono   |
| `BAJO_RIESGO` | `prob < 0.200`       | Comunicación estándar                    |

## API

| Método | Ruta | Auth |
|--------|------|------|
| GET    | `/`         | — |
| GET    | `/health`   | — |
| POST   | `/predecir` | `X-API-Key` |
| GET    | `/docs`     | — (Swagger) |

```bash
curl -X POST https://rodolfo-gallegos-infartos-mlops.hf.space/predecir \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"Genero":"Hombre","Edad":65,"Flag_hipertension":1,
       "Flag_problem_cardiaco":1,"Estados_civil":"Si",
       "Tipo_trabajo":"Empresa_privada","Zona_residencia":"Urbano",
       "Promedio_nivel_glucosa":220.5,"IMC":32.1,
       "Flag_fumador":"antes_fumaba"}'
# → {"probabilidad":0.8667,"decision":"ALTO_RIESGO","nivel_riesgo":"alto"}
```

Desde Swagger `/docs`: botón **Authorize** (candado) → pega el API key.

### Variables de entorno

| Var | Default | Para qué |
|-----|---------|----------|
| `API_KEY`                | _(vacío)_ | Si está set, `/predecir` exige `X-API-Key`. Vacío = auth OFF (dev/CI). |
| `ALLOWED_ORIGIN_REGEX`   | URL exacta del Space | CORS por regex. |
| `PREDICT_RATE_LIMIT`     | `30/minute`          | Límite por IP en `/predecir`. |
| `RATE_LIMIT_STORAGE_URI` | `memory://`          | `redis://...` para persistir entre reinicios. |

En HF: **Settings → Variables and secrets** → agregar `API_KEY` como secret.

## CI/CD

- `ml_pipeline.yml` corre en push a `main`: lint → train (dataset sintético) →
  build Docker → smoke test → opcional `deploy-hf` (manual via workflow_dispatch).
- `monitoreo_drift.yml` corre los lunes 8am UTC con Evidently.
- Auto-deploy desde GH Actions: setear secrets `HF_USER` y `HF_TOKEN` y disparar
  el workflow manualmente. Para deploy con dataset real → `make deploy-hf` local.

## Cumple los 8+1 niveles del curso MLOps-SDC

| Nivel | Tema | Estado |
|-------|------|--------|
| 1 | config + módulos + logging + Makefile | ✅ |
| 2 | MLflow tracking + Model Registry + DVC | ✅ |
| 3 | FastAPI /health + /predecir + singleton | ✅ |
| 4 | Docker + compose 3 servicios | ✅ |
| 5 | Makefile lint/test/train/validate/preprod | ✅ |
| 6 | setup.cfg + pytest (35) + smoke tests | ✅ |
| 7 | GitHub Actions 4 jobs + quality gates | ✅ |
| 8 | Evidently + drift + cron semanal | ✅ |
| 9 | Deploy en la nube (HF Spaces, opc. Cloud Run) | ✅ |

### Endurecimientos extra

Auth con API key, CORS al dominio exacto del Space, rate limiting con
`slowapi` (storage configurable a Redis), errores genéricos al cliente,
auto-deploy opcional desde GH Actions.
