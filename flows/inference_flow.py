"""Prefect inference pipeline flow for the DevOps Log Analyzer.

Loads logs from a directory, embeds them, classifies with the registered
MLflow model, generates HF explanations (concurrency-limited), saves
structured JSON results, and runs drift detection on the batch.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

import joblib
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONCURRENCY_LIMIT = 3


@task(name="load-logs", retries=1, retry_delay_seconds=10)
def load_logs_task(log_dir: str) -> list[dict]:
    """Read all .csv files from log_dir and return a list of log records.

    Args:
        log_dir: Directory containing one or more CSV files with log_text column.

    Returns:
        List of dicts with keys: id, log_text, true_label (may be None).
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: load-logs from {log_dir}")

    dir_path = Path(log_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Log directory not found: {dir_path}")

    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {dir_path}")

    records = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            records.append({
                "id": str(row.get("id", f"row_{len(records)}")),
                "log_text": str(row.get("log_text", "")),
                "true_label": row.get("label", None),
            })

    logger.info(f"Loaded {len(records)} logs from {len(csv_files)} file(s)")
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: load-logs ({elapsed:.1f}s)")
    return records


@task(name="embed-logs", retries=1, retry_delay_seconds=10)
def embed_task(records: list[dict]) -> np.ndarray:
    """Embed the log_text of each record.

    Args:
        records: List of log record dicts from load_logs_task.

    Returns:
        Numpy array of shape (n, 384).
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: embed-logs ({len(records)} texts)")

    from src.embed import embed_texts
    texts = [r["log_text"] for r in records]
    embeddings = embed_texts(texts)

    logger.info(f"Embeddings shape: {embeddings.shape}")
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: embed-logs ({elapsed:.1f}s)")
    return embeddings


@task(name="classify-logs", retries=1, retry_delay_seconds=10)
def classify_task(embeddings: np.ndarray) -> tuple[list[str], list[float]]:
    """Load the registered model and predict labels + probabilities.

    Tries MLflow model registry first; falls back to local model.pkl.

    Args:
        embeddings: Embedding matrix (n, 384).

    Returns:
        Tuple of (predicted_labels, confidence_scores).
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: classify-logs")

    import mlflow.sklearn

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    clf = None

    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        clf = mlflow.sklearn.load_model("models:/log-failure-classifier/latest")
        logger.info("Model loaded from MLflow registry.")
    except Exception as exc:
        logger.warning(f"MLflow unavailable ({exc}) — falling back to local model.pkl")
        local_model = ROOT / "models" / "model.pkl"
        if not local_model.exists():
            raise FileNotFoundError(
                f"No model found at {local_model}. Run training first."
            ) from exc
        clf = joblib.load(local_model)
        logger.info(f"Model loaded from local file: {local_model}")

    labels = clf.predict(embeddings).tolist()
    probas = clf.predict_proba(embeddings).max(axis=1).tolist()

    logger.info(f"Classified {len(labels)} samples. Label distribution: "
                f"{dict(zip(*np.unique(labels, return_counts=True)))}")
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: classify-logs ({elapsed:.1f}s)")
    return labels, probas


@task(name="explain-logs", retries=1, retry_delay_seconds=10)
def explain_task(records: list[dict], labels: list[str]) -> list[str]:
    """Generate HF explanations for each log, with concurrency limit of 3.

    Args:
        records: Original log records.
        labels: Predicted labels for each record.

    Returns:
        List of explanation strings, one per record.
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: explain-logs ({len(records)} items, concurrency={CONCURRENCY_LIMIT})")

    from src.explain import explain_failure

    explanations = []
    for i in range(0, len(records), CONCURRENCY_LIMIT):
        batch = records[i: i + CONCURRENCY_LIMIT]
        batch_labels = labels[i: i + CONCURRENCY_LIMIT]
        for rec, lbl in zip(batch, batch_labels):
            explanation = explain_failure(rec["log_text"], lbl)
            explanations.append(explanation)
        logger.info(f"Explained {min(i + CONCURRENCY_LIMIT, len(records))}/{len(records)} logs")
        if i + CONCURRENCY_LIMIT < len(records):
            time.sleep(0.5)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: explain-logs ({elapsed:.1f}s)")
    return explanations


@task(name="save-results", retries=1, retry_delay_seconds=10)
def save_results_task(
    records: list[dict],
    labels: list[str],
    confidences: list[float],
    explanations: list[str],
) -> Path:
    """Persist inference results as a timestamped JSON file.

    Args:
        records: Original log records (id, log_text, true_label).
        labels: Predicted labels.
        confidences: Confidence scores.
        explanations: HF-generated explanation strings.

    Returns:
        Path to the saved JSON file.
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: save-results ({len(records)} records)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"inference_{ts}.json"

    results = []
    for rec, lbl, conf, expl in zip(records, labels, confidences, explanations):
        results.append({
            "id": rec["id"],
            "log_text": rec["log_text"][:500],
            "predicted_label": lbl,
            "true_label": rec.get("true_label"),
            "confidence": round(float(conf), 4),
            "explanation": expl,
            "timestamp": datetime.now().isoformat(),
        })

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    logger.info(f"Results saved: {output_path} ({len(results)} records)")
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: save-results ({elapsed:.1f}s)")
    return output_path


@task(name="monitor-drift", retries=1, retry_delay_seconds=10)
def monitor_task(embeddings: np.ndarray, log_dir: str) -> bool:
    """Run drift detection against the reference embeddings.

    Saves the batch embeddings to a temp .npy and compares with the training
    reference. Prints a bold WARNING if drift is detected.

    Args:
        embeddings: Current batch embeddings.
        log_dir: Source directory (used to distinguish drift vs. normal data).

    Returns:
        True if drift was detected.
    """
    logger = get_run_logger()
    start = datetime.now()
    logger.info(f"[{start.strftime('%H:%M:%S')}] Task start: monitor-drift")

    ref_path = ROOT / "data" / "raw" / "embeddings.npy"
    if not ref_path.exists():
        logger.warning("Reference embeddings not found — skipping drift detection.")
        return False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_emb_path = ROOT / "reports" / f"current_embeddings_{ts}.npy"
    np.save(current_emb_path, embeddings)

    try:
        from src.monitor import run_drift_report
        drift = run_drift_report(
            reference_path=str(ref_path),
            current_path=str(current_emb_path),
        )
    except Exception as exc:
        logger.warning(f"Drift detection failed: {exc}")
        return False
    finally:
        if current_emb_path.exists():
            current_emb_path.unlink()

    if drift:
        logger.warning("=" * 60)
        logger.warning("*** WARNING: DATA DRIFT DETECTED ***")
        logger.warning("Current log distribution has shifted from training data.")
        logger.warning("Consider running the training-pipeline to retrain the model.")
        logger.warning("=" * 60)
    else:
        logger.info("No drift detected. Current data is consistent with training distribution.")

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Task done: monitor-drift ({elapsed:.1f}s)")
    return drift


@flow(
    name="inference-pipeline",
    description="Load logs → embed → classify → explain → save results → drift check",
)
def inference_flow(log_dir: str = "data/raw") -> dict:
    """Run the full inference pipeline on a directory of log CSV files.

    Args:
        log_dir: Relative or absolute path to a directory of .csv files.

    Returns:
        Dict summarising the run: n_logs, drift_detected, output_path.
    """
    logger = get_run_logger()
    logger.info(f"Starting inference-pipeline flow | log_dir={log_dir}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    abs_log_dir = str(ROOT / log_dir) if not Path(log_dir).is_absolute() else log_dir

    records = load_logs_task(abs_log_dir)
    embeddings = embed_task(records)
    labels, confidences = classify_task(embeddings)
    explanations = explain_task(records, labels)
    output_path = save_results_task(records, labels, confidences, explanations)
    drift = monitor_task(embeddings, abs_log_dir)

    summary = {
        "n_logs": len(records),
        "drift_detected": drift,
        "output_path": str(output_path),
        "timestamp": datetime.now().isoformat(),
    }
    logger.info(f"inference-pipeline completed: {summary}")
    return summary


if __name__ == "__main__":
    import sys as _sys
    dir_arg = _sys.argv[1] if len(_sys.argv) > 1 else "data/raw"
    result = inference_flow(log_dir=dir_arg)
    print(f"\nInference complete: {result}")
