"""Evidently-based data drift monitoring for the DevOps Log Analyzer.

Compares embedding distributions between reference (training) data and current
data using the first 50 principal dimensions. Saves an HTML report and a
structured JSON result, and returns a drift boolean for downstream use.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

REFERENCE_EMBEDDINGS = ROOT / "data" / "raw" / "embeddings.npy"
DRIFT_EMBEDDINGS = ROOT / "data" / "drift" / "embeddings.npy"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES = 50


def _build_feature_df(embeddings: np.ndarray, n_features: int = N_FEATURES) -> pd.DataFrame:
    """Convert an embedding matrix to a named-column DataFrame for Evidently.

    Args:
        embeddings: Float32 array of shape (n, D).
        n_features: How many leading dimensions to use.

    Returns:
        DataFrame with columns dim_0 … dim_{n_features-1}.
    """
    cols = [f"dim_{i}" for i in range(n_features)]
    return pd.DataFrame(embeddings[:, :n_features], columns=cols)


def run_drift_report(
    reference_path: str | None = None,
    current_path: str | None = None,
) -> bool:
    """Run an Evidently data-drift report and persist results.

    Loads the reference embeddings (training data) and the current embeddings
    (inference batch or drift simulation), runs DataDriftPreset on the first
    N_FEATURES dimensions, saves an HTML report and a drift_result.json.

    Args:
        reference_path: Path to reference embeddings .npy. Defaults to
            data/raw/embeddings.npy.
        current_path: Path to current embeddings .npy. Defaults to
            data/drift/embeddings.npy.

    Returns:
        True if data drift was detected, False otherwise.

    Raises:
        FileNotFoundError: If either embeddings file is missing.
    """
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report

    ref_path = Path(reference_path) if reference_path else REFERENCE_EMBEDDINGS
    cur_path = Path(current_path) if current_path else DRIFT_EMBEDDINGS

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference embeddings not found: {ref_path}")
    if not cur_path.exists():
        raise FileNotFoundError(f"Current embeddings not found: {cur_path}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading reference embeddings: {ref_path}")
    ref_emb = np.load(ref_path)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading current embeddings:   {cur_path}")
    cur_emb = np.load(cur_path)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Reference shape: {ref_emb.shape} | Current shape: {cur_emb.shape}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using first {N_FEATURES} embedding dimensions for drift detection.")

    ref_df = _build_feature_df(ref_emb, N_FEATURES)
    cur_df = _build_feature_df(cur_emb, N_FEATURES)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Evidently DataDriftPreset...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    report.save_html(str(html_path))
    print(f"[{datetime.now().strftime('%H:%M:%S')}] HTML report saved: {html_path}")

    raw = report.as_dict()
    metrics_list = raw.get("metrics", [])

    drift_detected = False
    drift_share = 0.0
    drifted_features = 0
    total_features = N_FEATURES

    for m in metrics_list:
        result = m.get("result", {})
        if "dataset_drift" in result:
            drift_detected = bool(result["dataset_drift"])
            drift_share = float(result.get("share_of_drifted_columns", result.get("drift_share", 0.0)))
            drifted_features = int(result.get("number_of_drifted_columns", 0))
            total_features = int(result.get("number_of_columns", N_FEATURES))
            break

    drift_result = {
        "drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "drifted_features": drifted_features,
        "total_features": total_features,
        "timestamp": datetime.now().isoformat(),
        "html_report": str(html_path),
        "reference_path": str(ref_path),
        "current_path": str(cur_path),
    }

    result_path = REPORTS_DIR / "drift_result.json"
    with open(result_path, "w") as fh:
        json.dump(drift_result, fh, indent=2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Drift result saved: {result_path}")

    pct = drift_share * 100
    if drift_detected:
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] *** DRIFT DETECTED *** "
            f"{drifted_features}/{total_features} features drifted ({pct:.1f}%)"
        )
    else:
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] NO DRIFT — "
            f"{drifted_features}/{total_features} features drifted ({pct:.1f}%)"
        )

    return drift_detected


def main() -> None:
    """Entry point: run drift detection with optional CLI arguments."""
    ref = sys.argv[1] if len(sys.argv) > 1 else None
    cur = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Drift Monitor ===")
    drift = run_drift_report(reference_path=ref, current_path=cur)
    sys.exit(1 if drift else 0)


if __name__ == "__main__":
    main()
