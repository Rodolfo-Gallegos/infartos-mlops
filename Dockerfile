FROM python:3.10-slim

LABEL maintainer="infartos-mlops"
LABEL description="API de predicción de riesgo de infarto"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user:user /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user:user requirements.txt .
USER user
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user api/ ./api/
COPY --chown=user:user src/ ./src/
COPY --chown=user:user utils/ ./utils/
COPY --chown=user:user config.py .
COPY --chown=user:user artifacts/modelo.pkl     ./artifacts/
COPY --chown=user:user artifacts/scaler.pkl     ./artifacts/
COPY --chown=user:user artifacts/features.csv   ./artifacts/
COPY --chown=user:user artifacts/metrics.json   ./artifacts/
COPY --chown=user:user artifacts/threshold.json ./artifacts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
