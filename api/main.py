"""FastAPI application for the DevOps Log Analyzer.

Exposes REST endpoints for log classification, drift reporting, model metrics,
and pipeline status. The classifier model is loaded at startup and cached in
memory; MLflow registry is preferred with a local model.pkl fallback.
"""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    DriftReportResponse,
    HealthResponse,
    MetricsResponse,
    PipelineRunSummary,
    PipelineStatusResponse,
    SimulateDriftResponse,
)
from src.explain import explain_failure

load_dotenv(ROOT / ".env")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
LOCAL_MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "data" / "raw" / "metrics.json"
DRIFT_RESULT_PATH = ROOT / "reports" / "drift_result.json"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = ROOT / "results"

_state: dict = {
    "clf": None,
    "model_version": "not_loaded",
    "mlflow_connected": False,
}


def _try_load_mlflow_model() -> tuple:
    """Attempt to load the registered model from the MLflow registry.

    Returns:
        Tuple of (model, version_string) or raises on failure.
    """
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("log-failure-classifier")
    if not versions:
        raise ValueError("No versions found in registry for 'log-failure-classifier'")
    latest = versions[-1]
    model = mlflow.sklearn.load_model(f"models:/log-failure-classifier/{latest.version}")
    return model, f"mlflow:v{latest.version}:run_{latest.run_id[:8]}"


def load_model() -> None:
    """Load classifier at startup — MLflow first, local fallback second.

    Populates _state['clf'], _state['model_version'], _state['mlflow_connected'].
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading model...")

    try:
        model, version = _try_load_mlflow_model()
        _state["clf"] = model
        _state["model_version"] = version
        _state["mlflow_connected"] = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded from MLflow registry: {version}")
        return
    except Exception as exc:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] MLflow unavailable ({exc}). Trying local model...")
        _state["mlflow_connected"] = False

    if LOCAL_MODEL_PATH.exists():
        try:
            _state["clf"] = joblib.load(LOCAL_MODEL_PATH)
            _state["model_version"] = "local:model.pkl"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded from local file: {LOCAL_MODEL_PATH}")
        except Exception as exc:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Failed to load local model: {exc}")
    else:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: No model found at {LOCAL_MODEL_PATH}. "
            "Run the training pipeline first."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load model on startup."""
    load_model()
    yield


app = FastAPI(
    title="DevOps Log Analyzer API",
    description=(
        "MLOps-grade REST API for CI/CD log failure classification. "
        "Powered by sentence-transformers + scikit-learn + MLflow. "
        "Built by Malek Soula as a portfolio project."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _embed_single(text: str) -> np.ndarray:
    """Embed a single log text string.

    Args:
        text: Raw log text.

    Returns:
        2D numpy array of shape (1, 384).
    """
    from src.embed import embed_texts
    return embed_texts([text])


def _check_mlflow() -> bool:
    """Probe MLflow tracking server reachability.

    Returns:
        True if reachable, False otherwise.
    """
    import requests as _req
    try:
        resp = _req.get(f"{MLFLOW_URI}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Return API health status including model and MLflow connectivity."""
    return HealthResponse(
        status="ok",
        model_loaded=_state["clf"] is not None,
        mlflow_connected=_check_mlflow(),
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
async def classify_log(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a CI/CD log and return the predicted failure category.

    Embeds the log text, predicts the failure label and confidence,
    then requests a 2-sentence explanation from HuggingFace.

    Args:
        request: ClassifyRequest body with log_text field.

    Returns:
        ClassifyResponse with label, confidence, explanation, and timing.

    Raises:
        HTTPException 503: If no model is loaded.
    """
    if _state["clf"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    t_start = time.perf_counter()

    log_text = request.log_text[:4000]

    try:
        embedding = _embed_single(log_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc

    try:
        clf = _state["clf"]
        label = clf.predict(embedding)[0]
        proba = float(clf.predict_proba(embedding).max())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    try:
        explanation = explain_failure(log_text, label)
    except Exception as exc:
        explanation = f"Explanation unavailable: {exc}"

    inference_ms = (time.perf_counter() - t_start) * 1000

    return ClassifyResponse(
        predicted_label=label,
        confidence=round(proba, 4),
        explanation=explanation,
        model_version=_state["model_version"],
        inference_time_ms=round(inference_ms, 2),
    )


@app.get("/drift-report", response_model=DriftReportResponse, tags=["monitoring"])
async def get_drift_report() -> DriftReportResponse:
    """Return the latest drift detection result.

    Raises:
        HTTPException 404: If no drift report has been generated yet.
    """
    if not DRIFT_RESULT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No drift report found. Run POST /simulate-drift or src/monitor.py first.",
        )

    try:
        with open(DRIFT_RESULT_PATH) as fh:
            data = json.load(fh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read drift report: {exc}") from exc

    return DriftReportResponse(
        drift_detected=data.get("drift_detected", False),
        drift_share=data.get("drift_share", 0.0),
        drifted_features=data.get("drifted_features", 0),
        total_features=data.get("total_features", 0),
        timestamp=data.get("timestamp", ""),
        html_report=data.get("html_report", ""),
    )


@app.get("/drift-report/html", response_class=HTMLResponse, tags=["monitoring"])
async def get_drift_report_html() -> HTMLResponse:
    """Serve the latest Evidently drift report as an HTML page.

    Raises:
        HTTPException 404: If no HTML report exists.
    """
    html_files = sorted(REPORTS_DIR.glob("drift_report_*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not html_files:
        raise HTTPException(status_code=404, detail="No HTML drift report found. Run drift detection first.")

    latest = html_files[0]
    try:
        return HTMLResponse(content=latest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read HTML report: {exc}") from exc


@app.get("/pipeline-status", response_model=PipelineStatusResponse, tags=["monitoring"])
async def get_pipeline_status() -> PipelineStatusResponse:
    """Return metadata about the last 10 inference runs.

    Reads results/*.json files sorted newest-first.
    """
    json_files = sorted(
        RESULTS_DIR.glob("inference_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:10]

    drift_result = {}
    if DRIFT_RESULT_PATH.exists():
        try:
            with open(DRIFT_RESULT_PATH) as fh:
                drift_result = json.load(fh)
        except Exception:
            pass

    runs = []
    total_logs = 0
    for jf in json_files:
        try:
            with open(jf) as fh:
                records = json.load(fh)
            if not isinstance(records, list):
                continue
            n = len(records)
            avg_conf = float(np.mean([r.get("confidence", 0.0) for r in records])) if n else 0.0
            first_ts = records[0].get("timestamp", jf.stem.replace("inference_", "")) if records else ""
            total_logs += n
            runs.append(PipelineRunSummary(
                timestamp=first_ts,
                log_count=n,
                drift_detected=drift_result.get("drift_detected", False),
                output_file=jf.name,
                avg_confidence=round(avg_conf, 4),
            ))
        except Exception:
            continue

    return PipelineStatusResponse(runs=runs, total_logs_processed=total_logs)


@app.get("/metrics", response_model=MetricsResponse, tags=["model"])
async def get_metrics() -> MetricsResponse:
    """Return the current model performance metrics from metrics.json.

    Raises:
        HTTPException 404: If metrics.json has not been generated yet.
    """
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run the training pipeline first.",
        )

    try:
        with open(METRICS_PATH) as fh:
            data = json.load(fh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {exc}") from exc

    return MetricsResponse(
        best_model=data.get("best_model", "unknown"),
        weighted_f1=data.get("weighted_f1", 0.0),
        per_class=data.get("per_class", {}),
        n_train=data.get("n_train", 0),
        n_test=data.get("n_test", 0),
        timestamp=data.get("timestamp", ""),
        run_id=data.get("run_id"),
    )


@app.post("/simulate-drift", response_model=SimulateDriftResponse, tags=["monitoring"])
async def simulate_drift() -> SimulateDriftResponse:
    """Run inference on the drift dataset and then detect drift vs. reference.

    Embeds the drift logs, classifies them, compares embeddings to the training
    reference distribution using Evidently, and returns the drift result.

    Raises:
        HTTPException 404: If drift or reference data is missing.
        HTTPException 503: If no model is loaded.
    """
    if _state["clf"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    drift_csv = ROOT / "data" / "drift" / "logs_drift.csv"
    drift_emb_path = ROOT / "data" / "drift" / "embeddings.npy"
    ref_emb_path = ROOT / "data" / "raw" / "embeddings.npy"

    if not drift_csv.exists():
        raise HTTPException(
            status_code=404,
            detail="Drift dataset not found. Run src/generate_data.py first.",
        )

    try:
        import pandas as pd
        from src.embed import embed_and_save
        embeddings, labels = embed_and_save(str(drift_csv), str(drift_emb_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift embedding failed: {exc}") from exc

    try:
        clf = _state["clf"]
        pred_labels = clf.predict(embeddings).tolist()
        confidences = clf.predict_proba(embeddings).max(axis=1).tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift classification failed: {exc}") from exc

    inference_summary = {
        "n_logs": len(pred_labels),
        "avg_confidence": round(float(np.mean(confidences)), 4),
        "label_distribution": {
            k: int(v) for k, v in zip(*np.unique(pred_labels, return_counts=True))
        } if pred_labels else {},
    }

    if not ref_emb_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Reference embeddings missing. Run training pipeline first.",
        )

    try:
        from src.monitor import run_drift_report
        drift_detected = run_drift_report(
            reference_path=str(ref_emb_path),
            current_path=str(drift_emb_path),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {exc}") from exc

    drift_data: dict = {}
    if DRIFT_RESULT_PATH.exists():
        with open(DRIFT_RESULT_PATH) as fh:
            drift_data = json.load(fh)

    status = "DRIFT DETECTED" if drift_detected else "NO DRIFT"
    return SimulateDriftResponse(
        message=f"Simulation complete. Result: {status}",
        drift_detected=drift_detected,
        drift_share=drift_data.get("drift_share", 0.0),
        inference_summary=inference_summary,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
