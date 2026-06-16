"""Prefect training pipeline flow for the DevOps Log Analyzer.

Orchestrates the full ML training lifecycle: generate data → embed → train
→ evaluate → register. Each step is a retryable Prefect task.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

from prefect import flow, task
from prefect.logging import get_run_logger

from src.embed import embed_and_save
from src.evaluate import evaluate_model
from src.generate_data import main as _generate_data_main
from src.train import train_model


@task(name="generate-data", retries=1, retry_delay_seconds=10)
def generate_data_task() -> None:
    """Generate 600 synthetic labeled CI/CD log samples.

    Returns:
        None — writes data/raw/logs.csv and data/drift/logs_drift.csv.
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: generate-data")

    from src.generate_data import (
        DATA_DRIFT,
        DATA_RAW,
        DRIFT_SAMPLES,
        SAMPLES_PER_CLASS,
        generate_drift_data,
        generate_training_data,
    )
    import pandas as pd

    train_path = DATA_RAW / "logs.csv"
    drift_path = DATA_DRIFT / "logs_drift.csv"

    logger.info(f"Generating training data ({SAMPLES_PER_CLASS * 5} samples)...")
    train_df = generate_training_data()
    train_df.to_csv(train_path, index=False)
    logger.info(f"Training data saved: {train_path} ({len(train_df)} rows)")

    logger.info(f"Generating drift data ({DRIFT_SAMPLES} samples)...")
    drift_df = generate_drift_data()
    drift_df.to_csv(drift_path, index=False)
    logger.info(f"Drift data saved: {drift_path} ({len(drift_df)} rows)")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: generate-data ({elapsed:.1f}s)")


@task(name="embed-training-data", retries=1, retry_delay_seconds=10)
def embed_task() -> None:
    """Compute sentence-transformer embeddings for the training dataset.

    Returns:
        None — writes data/raw/embeddings.npy and data/raw/embeddings_labels.npy.
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: embed-training-data")

    train_csv = str(ROOT / "data" / "raw" / "logs.csv")
    train_emb = str(ROOT / "data" / "raw" / "embeddings.npy")
    embeddings, labels = embed_and_save(train_csv, train_emb)

    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Labels count: {len(labels)}")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: embed-training-data ({elapsed:.1f}s)")


@task(name="train-classifier", retries=1, retry_delay_seconds=10)
def train_task() -> dict:
    """Train LogisticRegression and RandomForest, pick best by weighted F1.

    Returns:
        Metrics dict with best model name, F1 score, and per-class breakdown.
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: train-classifier")

    metrics = train_model()

    logger.info(f"Best model: {metrics['best_model']}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"MLflow run ID: {metrics.get('run_id', 'N/A')}")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: train-classifier ({elapsed:.1f}s)")
    return metrics


@task(name="evaluate-quality-gate", retries=0)
def evaluate_task() -> dict:
    """Enforce F1 >= 0.85 quality gate; raises on failure.

    Returns:
        Metrics dict if the model passes the quality gate.

    Raises:
        ValueError: If the weighted F1 score is below the threshold.
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: evaluate-quality-gate")

    import io
    import sys as _sys

    captured = io.StringIO()
    old_stdout = _sys.stdout
    _sys.stdout = captured

    try:
        from src.evaluate import METRICS_PATH, evaluate_model as _eval
        metrics = _eval(METRICS_PATH)
    except SystemExit as exc:
        _sys.stdout = old_stdout
        output = captured.getvalue()
        logger.error(output)
        raise ValueError(
            f"Quality gate FAILED — weighted F1 below threshold. Details:\n{output}"
        ) from exc
    finally:
        _sys.stdout = old_stdout

    output = captured.getvalue()
    for line in output.splitlines():
        if line.strip():
            logger.info(line)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: evaluate-quality-gate ({elapsed:.1f}s)")
    return metrics


@task(name="register-model", retries=1, retry_delay_seconds=10)
def register_task(metrics: dict) -> None:
    """Log registration confirmation to the Prefect UI.

    Args:
        metrics: Metrics dict returned by train_task / evaluate_task.

    Returns:
        None
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: register-model")

    model_name = metrics.get("best_model", "unknown")
    f1 = metrics.get("weighted_f1", 0.0)
    run_id = metrics.get("run_id", "N/A")
    ts = metrics.get("timestamp", datetime.now().isoformat())

    logger.info("=" * 50)
    logger.info("Model registered successfully")
    logger.info(f"  Registry name : log-failure-classifier")
    logger.info(f"  Algorithm     : {model_name}")
    logger.info(f"  Weighted F1   : {f1:.4f}")
    logger.info(f"  MLflow run ID : {run_id}")
    logger.info(f"  Timestamp     : {ts}")
    logger.info("=" * 50)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: register-model ({elapsed:.1f}s)")


@flow(
    name="training-pipeline",
    description="End-to-end training: generate → embed → train → evaluate → register",
)
def training_flow() -> None:
    """Orchestrate the full DevOps Log Analyzer training pipeline.

    Tasks run sequentially with retries. The pipeline aborts and raises if
    the quality gate task fails.

    Returns:
        None
    """
    logger = get_run_logger()
    logger.info("Starting training-pipeline flow")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    generate_data_task()
    embed_task()
    metrics = train_task()
    evaluated_metrics = evaluate_task()
    register_task(evaluated_metrics)

    logger.info("training-pipeline flow completed successfully.")


if __name__ == "__main__":
    training_flow()
