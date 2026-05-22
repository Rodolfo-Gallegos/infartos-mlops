FROM python:3.10-slim

LABEL maintainer="infartos-mlops"
LABEL description="API de predicción de riesgo de infarto"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY utils/ ./utils/
COPY config.py .
COPY artifacts/modelo.pkl     ./artifacts/
COPY artifacts/scaler.pkl     ./artifacts/
COPY artifacts/features.csv   ./artifacts/
COPY artifacts/metrics.json   ./artifacts/
COPY artifacts/threshold.json ./artifacts/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
