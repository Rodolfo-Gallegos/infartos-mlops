.PHONY: all lint test train validate docker preprod-up preprod-down \
        preprod-logs smoke deploy deploy-gcp dvc-pull dvc-push clean help

PROJECT   = infartos-mlops
API_IMAGE = infartos-api
COMPOSE   = docker compose -f docker-compose.preprod.yml --env-file .env.preprod

all: lint test train validate docker

# ── Calidad de código ───────────────────────────────────────────────────────
lint:
	flake8 src/ pipeline/ api/ tests/ --config=setup.cfg

test:
	pytest tests/ -v --tb=short --cov=src --cov=api --cov-report=term-missing \
	       --ignore=tests/smoke

# ── Pipeline ML ────────────────────────────────────────────────────────────
train:
	cp "../Prevención_de_infartos/Dataset prevención de infartos.csv" data/dataset.csv 2>/dev/null || true
	python run_pipeline.py

validate:
	python -c "
import json; from config import METRICS_PATH, RECALL_MINIMO, F1_MINIMO
m = json.load(open(METRICS_PATH))
r, f = m['recall'], m['f1']
print(f'Recall={r:.4f} (min={RECALL_MINIMO}) | F1={f:.4f} (min={F1_MINIMO})')
fallos = []
if r < RECALL_MINIMO: fallos.append(f'recall {r:.4f} < {RECALL_MINIMO}')
if f < F1_MINIMO:     fallos.append(f'f1 {f:.4f} < {F1_MINIMO}')
assert not fallos, 'Quality gate FALLIDO: ' + ' | '.join(fallos)
print('Quality gate APROBADO')
"

# ── Docker ─────────────────────────────────────────────────────────────────
docker:
	docker build -t $(API_IMAGE):local .

preprod-up:
	cp "../Prevención_de_infartos/Dataset prevención de infartos.csv" \
	   /tmp/infartos-data.csv 2>/dev/null || true
	$(COMPOSE) up --build -d
	@echo "Esperando que el trainer complete..."
	@sleep 60
	@docker compose -f docker-compose.preprod.yml ps

preprod-down:
	$(COMPOSE) down -v

preprod-logs:
	$(COMPOSE) logs -f

preprod-logs-api:
	$(COMPOSE) logs -f api

preprod-logs-trainer:
	$(COMPOSE) logs -f trainer

preprod-logs-mlflow:
	$(COMPOSE) logs -f mlflow

# ── Smoke tests ────────────────────────────────────────────────────────────
smoke:
	pytest tests/smoke/ -v --tb=short

# ── DVC ────────────────────────────────────────────────────────────────────
dvc-pull:
	dvc pull

dvc-push:
	dvc push

# ── Deploy ─────────────────────────────────────────────────────────────────
VERSION ?= latest

deploy:
	bash deploy.sh $(VERSION)

deploy-gcp:
	bash deploy_gcp.sh $(VERSION)

# ── Limpieza ───────────────────────────────────────────────────────────────
clean:
	rm -rf artifacts/ mlruns/ reportes/ __pycache__ .coverage htmlcov/ \
	       src/__pycache__ pipeline/__pycache__ api/__pycache__

help:
	@echo ""
	@echo "Targets disponibles:"
	@echo "  all          - lint + test + train + validate + docker"
	@echo "  lint         - flake8"
	@echo "  test         - pytest con coverage"
	@echo "  train        - pipeline completo (ingest→features→train→evaluate)"
	@echo "  validate     - quality gate (recall >= 0.70 y f1 >= 0.10)"
	@echo "  docker       - construir imagen API"
	@echo "  preprod-up   - levantar stack de 3 servicios"
	@echo "  preprod-down - bajar stack y limpiar volúmenes"
	@echo "  smoke        - smoke tests contra servicios reales"
	@echo "  dvc-pull     - traer datasets versionados (DVC)"
	@echo "  dvc-push     - publicar datasets versionados (DVC)"
	@echo "  deploy       - despliegue local con rollback (deploy.sh)"
	@echo "  deploy-gcp   - despliegue a Cloud Run (deploy_gcp.sh)"
	@echo "  clean        - limpiar artifacts y caches"
	@echo ""
