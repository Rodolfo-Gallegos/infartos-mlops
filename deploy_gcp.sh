#!/bin/bash
# deploy_gcp.sh - Despliegue a Google Cloud Run via Artifact Registry.
#
# Prerrequisitos:
#   - gcloud CLI instalado y autenticado: gcloud auth login
#   - Proyecto GCP creado con billing activo
#   - APIs habilitadas: artifactregistry.googleapis.com, run.googleapis.com
#   - Artifact Registry repo creado: infartos-mlops (region us-central1)
#
# Uso:
#   bash deploy_gcp.sh                         # tag = latest, project del gcloud config
#   bash deploy_gcp.sh v1.0.0 mi-proyecto-gcp  # tag y project explícitos

set -e

VERSION=${1:-latest}
PROJECT_ID=${2:-$(gcloud config get-value project)}
REGION="${REGION:-us-central1}"
REPO="${REPO:-infartos-mlops}"
SERVICE="${SERVICE:-infartos-api}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${VERSION}"

echo "============================================="
echo " DEPLOY CLOUD RUN - $SERVICE:$VERSION"
echo " Project : $PROJECT_ID"
echo " Region  : $REGION"
echo " Image   : $IMAGE"
echo "============================================="

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: no hay project_id. Configura con: gcloud config set project <ID>"
    exit 1
fi

# ── 1. Verificar artifacts locales ───────────────────────────────────────
echo "[1/5] Verificando artifacts del modelo..."
for f in artifacts/modelo.pkl artifacts/scaler.pkl artifacts/features.csv \
         artifacts/metrics.json artifacts/threshold.json; do
    if [ ! -f "$f" ]; then
        echo "ERROR: falta $f - corre 'make train' primero."
        exit 1
    fi
done
echo "  Artifacts OK"

# ── 2. Autenticar Docker contra Artifact Registry ────────────────────────
echo "[2/5] Configurando Docker para Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── 3. Build imagen para linux/amd64 (Cloud Run no soporta arm64) ────────
echo "[3/5] Construyendo imagen Docker..."
docker build --platform=linux/amd64 -t "$IMAGE" .
echo "  Imagen construida: $IMAGE"

# ── 4. Push a Artifact Registry ──────────────────────────────────────────
echo "[4/5] Subiendo imagen a Artifact Registry..."
docker push "$IMAGE"
echo "  Push OK"

# ── 5. Deploy a Cloud Run ────────────────────────────────────────────────
echo "[5/5] Desplegando a Cloud Run..."
gcloud run deploy "$SERVICE" \
    --image="$IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8000 \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=3 \
    --set-env-vars="PIPELINE_VERSION=${VERSION},RECALL_MIN=0.70,F1_MIN=0.10" \
    --quiet

URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')

echo ""
echo "============================================="
echo " DEPLOY EXITOSO - $SERVICE:$VERSION"
echo " URL    : $URL"
echo " Health : $URL/health"
echo " Docs   : $URL/docs"
echo "============================================="

# ── 6. Smoke test contra Cloud Run ───────────────────────────────────────
echo "Smoke test contra Cloud Run..."
curl -sf "$URL/health" | python3 -m json.tool || {
    echo "ADVERTENCIA: el smoke test falló. Revisa los logs:"
    echo "  gcloud run services logs read $SERVICE --region=$REGION"
}
