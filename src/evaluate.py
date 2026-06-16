"""Quality gate for the DevOps Log Analyzer training pipeline.

Reads metrics.json and exits with code 1 if the weighted F1 is below the
minimum threshold, allowing CI/CD pipelines to fail on regressions.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
METRICS_PATH = ROOT / "data" / "raw" / "metrics.json"
F1_THRESHOLD = 0.85


def evaluate_model(metrics_path: Path = METRICS_PATH) -> dict:
    """Read metrics.json and enforce the F1 quality gate.

    Args:
        metrics_path: Path to the metrics JSON file produced by train.py.

    Returns:
        The metrics dict if the model passes.

    Raises:
        SystemExit(1): If weighted F1 is below F1_THRESHOLD.
        FileNotFoundError: If metrics.json does not exist.
    """
    if not metrics_path.exists():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Metrics file not found: {metrics_path}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Run src/train.py first.")
        sys.exit(1)

    with open(metrics_path) as fh:
        metrics = json.load(fh)

    weighted_f1 = metrics.get("weighted_f1", 0.0)
    best_model = metrics.get("best_model", "unknown")
    per_class = metrics.get("per_class", {})
    n_train = metrics.get("n_train", "?")
    n_test = metrics.get("n_test", "?")
    ts = metrics.get("timestamp", "?")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Quality Gate Evaluation ===")
    print(f"  Model:      {best_model}")
    print(f"  Trained at: {ts}")
    print(f"  Train/Test: {n_train} / {n_test}")
    print(f"  Threshold:  weighted F1 >= {F1_THRESHOLD:.2f}")
    print(f"  Actual F1:  {weighted_f1:.4f}")
    print()

    if per_class:
        print(f"  Per-class breakdown:")
        print(f"  {'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}  Status")
        print(f"  {'-' * 62}")
        for lbl, m in sorted(per_class.items()):
            status = "OK" if m["f1"] >= 0.80 else "WARN"
            print(
                f"  {lbl:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}  {status}"
            )
        print()

    if weighted_f1 < F1_THRESHOLD:
        print(f"  FAIL — weighted F1 {weighted_f1:.4f} is below threshold {F1_THRESHOLD:.2f}")
        print(f"  Action required: inspect class-level metrics above and retrain with more data.")
        sys.exit(1)
    else:
        print(f"  PASS — weighted F1 {weighted_f1:.4f} meets the {F1_THRESHOLD:.2f} threshold.")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model is ready for registration and deployment.")
        return metrics


def main() -> None:
    """Entry point for standalone quality gate check."""
    evaluate_model()


if __name__ == "__main__":
    main()
