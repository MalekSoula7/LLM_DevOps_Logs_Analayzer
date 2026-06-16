"""MLflow-instrumented training module for the DevOps Log Analyzer.

Trains LogisticRegression and RandomForest classifiers on pre-computed
sentence-transformer embeddings, compares them by weighted F1, logs
the winner to MLflow, and registers it to the model registry.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

EMBEDDINGS_PATH = ROOT / "data" / "raw" / "embeddings.npy"
LABELS_PATH = ROOT / "data" / "raw" / "embeddings_labels.npy"
METRICS_PATH = ROOT / "data" / "raw" / "metrics.json"
MODEL_PATH = ROOT / "models" / "model.pkl"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "devops-log-analyzer"
REGISTRY_NAME = "log-failure-classifier"
LABEL_NAMES = ["build_failure", "config_error", "infra_failure", "test_failure", "timeout"]

CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], run_dir: Path) -> Path:
    """Save confusion matrix as a PNG artifact.

    Args:
        y_true: Ground-truth label array.
        y_pred: Predicted label array.
        labels: Ordered list of class names.
        run_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = range(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    out = run_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def train_model() -> dict:
    """Train, compare, and register the best classifier.

    Loads embeddings from the canonical data/raw path, trains two classifiers,
    compares by weighted F1, logs everything to MLflow, registers the winner,
    and saves model.pkl and metrics.json.

    Returns:
        Dict with keys: best_model_name, weighted_f1, per_class_metrics, run_id.

    Raises:
        FileNotFoundError: If embedding files are missing.
    """
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_PATH}. Run src/embed.py first."
        )

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading embeddings...")
    X = np.load(EMBEDDINGS_PATH)
    y = np.load(LABELS_PATH, allow_pickle=True)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Split: {len(X_train)} train / {len(X_test)} test")

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_available = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] MLflow connected: {MLFLOW_URI}")
    except Exception as exc:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: MLflow unavailable ({exc}). Training locally only.")
        mlflow_available = False

    results = {}
    for name, clf in CANDIDATES.items():
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training: {name}")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        results[name] = {"clf": clf, "y_pred": y_pred, "f1": f1}
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {name} — weighted F1: {f1:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_clf = results[best_name]["clf"]
    best_pred = results[best_name]["y_pred"]
    best_f1 = results[best_name]["f1"]

    present_labels = sorted(set(y_test) | set(best_pred))
    report = classification_report(y_test, best_pred, labels=present_labels, output_dict=True)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_test, best_pred, labels=present_labels, average=None
    )

    per_class = {}
    for i, lbl in enumerate(present_labels):
        per_class[lbl] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per_class[i]),
        }

    metrics = {
        "best_model": best_name,
        "weighted_f1": float(best_f1),
        "per_class": per_class,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "timestamp": datetime.now().isoformat(),
        "run_id": None,
    }

    cm_path = _save_confusion_matrix(y_test, best_pred, present_labels, REPORTS_DIR)

    run_id = None
    if mlflow_available:
        try:
            with mlflow.start_run(run_name=f"{best_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
                run_id = run.info.run_id
                mlflow.set_tag("model_name", best_name)
                mlflow.log_param("model_type", best_name)
                mlflow.log_param("n_train", len(X_train))
                mlflow.log_param("n_test", len(X_test))
                mlflow.log_param("test_size", 0.2)

                if hasattr(best_clf, "C"):
                    mlflow.log_param("C", best_clf.C)
                if hasattr(best_clf, "n_estimators"):
                    mlflow.log_param("n_estimators", best_clf.n_estimators)

                mlflow.log_metric("weighted_f1", best_f1)
                for lbl, m in per_class.items():
                    mlflow.log_metric(f"{lbl}_precision", m["precision"])
                    mlflow.log_metric(f"{lbl}_recall", m["recall"])
                    mlflow.log_metric(f"{lbl}_f1", m["f1"])

                mlflow.log_artifact(str(cm_path), artifact_path="plots")

                model_info = mlflow.sklearn.log_model(
                    best_clf,
                    artifact_path="model",
                    registered_model_name=REGISTRY_NAME,
                )
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Model logged and registered: {REGISTRY_NAME}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] MLflow run ID: {run_id}")
                metrics["run_id"] = run_id
        except Exception as exc:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: MLflow logging failed: {exc}")

    joblib.dump(best_clf, MODEL_PATH)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model saved locally: {MODEL_PATH}")

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Metrics saved: {METRICS_PATH}")

    print(f"\n{'=' * 60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Model':<25} {'Weighted F1':>12}")
    print(f"  {'-' * 37}")
    for name in CANDIDATES:
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:<25} {results[name]['f1']:>12.4f}{marker}")
    print(f"\n  Per-class breakdown ({best_name}):")
    print(f"  {'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 52}")
    for lbl, m in per_class.items():
        print(f"  {lbl:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print(f"{'=' * 60}\n")

    return metrics


def main() -> None:
    """Entry point for standalone training."""
    result = train_model()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Best model: {result['best_model']} | Weighted F1: {result['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
