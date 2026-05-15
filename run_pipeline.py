import logging
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import ingest, features, train, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PIPELINE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    inicio = time.time()
    logger.info("==============================")
    logger.info("  PIPELINE — INFARTOS MLOps  ")
    logger.info("==============================")

    df, mediana_imc = ingest.ejecutar()
    X, y, scaler, selector = features.ejecutar(df)
    metricas = train.ejecutar(X, y, scaler=scaler)
    evaluate.ejecutar(X, y, metricas)

    elapsed = time.time() - inicio
    logger.info(f"Pipeline completado en {elapsed:.1f}s")
    logger.info(f"Recall={metricas['recall']} | F1={metricas['f1']} | "
                f"ROC-AUC={metricas['roc_auc']}")


if __name__ == "__main__":
    main()
