#!/bin/bash
# deploy_hf.sh - Despliegue a Hugging Face Spaces (Docker SDK).
#
# Prerrequisitos (una sola vez):
#   1. Crear cuenta en https://huggingface.co
#   2. Crear Space vacío: https://huggingface.co/new-space
#      - Space name : infartos-mlops (o el que prefieras)
#      - License    : mit
#      - SDK        : Docker -> Blank
#      - Visibility : Public
#   3. Token de acceso con permiso 'write':
#      https://huggingface.co/settings/tokens
#   4. Exportar:
#        export HF_USER=tu_usuario_hf
#        export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
#
# Uso:
#   bash deploy_hf.sh                 # space_name = infartos-mlops
#   bash deploy_hf.sh otro-nombre     # space_name explícito

set -e

SPACE_NAME="${1:-infartos-mlops}"

if [ -z "$HF_USER" ] || [ -z "$HF_TOKEN" ]; then
    echo "ERROR: exporta HF_USER y HF_TOKEN antes de correr (ver header del script)."
    exit 1
fi

HF_REMOTE="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

echo "============================================="
echo " DEPLOY HUGGING FACE SPACES - ${SPACE_NAME}"
echo " User    : ${HF_USER}"
echo " Space   : https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
echo "============================================="

# ── 1. Verificar artifacts ───────────────────────────────────────────────
echo "[1/4] Verificando artifacts del modelo..."
for f in artifacts/modelo.pkl artifacts/scaler.pkl artifacts/features.csv \
         artifacts/metrics.json artifacts/threshold.json; do
    if [ ! -f "$f" ]; then
        echo "ERROR: falta $f - corre 'make train' primero."
        exit 1
    fi
done
echo "  Artifacts OK"

# ── 2. Preparar staging area limpia (sin historia ni binarios) ──────────
echo "[2/4] Preparando snapshot limpio (sin historia git)..."
TMP=$(mktemp -d -t hf-deploy-XXXXXX)
trap 'rm -rf "$TMP"' EXIT

# Copiar working tree usando rsync para excluir lo que HF no acepta
# (sin LFS): binarios grandes y todo lo que ignora .gitignore se omite.
rsync -a \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='docs/*.pptx' --exclude='docs/*.pdf' \
    --exclude='docs/*.docx' --exclude='docs/*.xlsx' \
    --exclude='*.tar.gz' --exclude='*.zip' \
    --exclude='mlruns' --exclude='reportes' \
    --exclude='.dvc' --exclude='data' \
    --exclude='.env' --exclude='.env.preprod' \
    --exclude='htmlcov' --exclude='.coverage' \
    ./ "$TMP/"

# Asegurar que los artifacts sí están (no los excluye .gitignore en la copia,
# pero pueden no estar si rsync los saltó por alguna razón)
mkdir -p "$TMP/artifacts"
cp artifacts/modelo.pkl artifacts/scaler.pkl artifacts/features.csv \
   artifacts/metrics.json artifacts/threshold.json "$TMP/artifacts/"

# Quitar reglas que ignorarían los artifacts en HF
sed -i '/^artifacts\//d; /^\*\.pkl$/d' "$TMP/.gitignore" 2>/dev/null || true

# Prepender frontmatter YAML que HF Spaces necesita para configurar el Space.
# No se mantiene en el README de GitHub porque GitHub lo renderiza como tabla.
TMP_README=$(mktemp)
cat > "$TMP_README" <<'EOF'
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

EOF
cat "$TMP/README.md" >> "$TMP_README"
mv "$TMP_README" "$TMP/README.md"

# ── 3. Init repo huérfano (un solo commit, sin historia previa) ─────────
echo "[3/4] Creando commit huérfano..."
cd "$TMP"
git init -q -b main
git -c user.email="deploy@hf.local" -c user.name="hf-deploy" \
    add -A && \
git -c user.email="deploy@hf.local" -c user.name="hf-deploy" \
    commit -q -m "deploy: snapshot para HF Spaces ($(date -Iseconds))"

# ── 4. Push a HF (force, sobrescribe main del Space) ─────────────────────
echo "[4/4] Push a Hugging Face Spaces..."
git remote add hf "$HF_REMOTE"
git push hf main --force

URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
echo ""
echo "============================================="
echo " DEPLOY ENVIADO"
echo " Space     : $URL"
echo " API root  : ${URL/spaces/${HF_USER}}"
echo "============================================="
echo "El build tarda 3-6 min. Mira logs en la pestaña 'Logs' del Space."
echo "Cuando esté Running, prueba:"
echo "  curl ${URL/spaces\//}-${HF_USER//\//-}.hf.space/health  # ojo: la URL real la da HF"
echo "  (la URL pública del API aparece arriba a la derecha del Space, click 'Embed this Space')"
