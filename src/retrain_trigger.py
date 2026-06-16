"""Drift-triggered retraining controller for the DevOps Log Analyzer.

Reads the latest drift_result.json and, when drift is detected, triggers
the Prefect training pipeline flow to retrain and re-register the model.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

DRIFT_RESULT_PATH = ROOT / "reports" / "drift_result.json"
TRAINING_FLOW_PATH = ROOT / "flows" / "training_flow.py"


def read_drift_result(result_path: Path = DRIFT_RESULT_PATH) -> dict:
    """Load and return the latest drift detection result.

    Args:
        result_path: Path to drift_result.json produced by monitor.py.

    Returns:
        Dict with drift detection fields.

    Raises:
        FileNotFoundError: If the result file does not exist.
    """
    if not result_path.exists():
        raise FileNotFoundError(
            f"Drift result not found: {result_path}\n"
            "Run src/monitor.py first to generate a drift report."
        )
    with open(result_path) as fh:
        return json.load(fh)


def trigger_retraining() -> int:
    """Invoke the Prefect training flow as a subprocess.

    Returns:
        Return code of the subprocess (0 = success).
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Triggering Prefect training flow...")
    prefect_api = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
    env = os.environ.copy()
    env["PREFECT_API_URL"] = prefect_api

    try:
        result = subprocess.run(
            [sys.executable, str(TRAINING_FLOW_PATH)],
            env=env,
            cwd=str(ROOT),
            timeout=1800,
        )
        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Training flow completed successfully.")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Training flow exited with code {result.returncode}.")
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Training flow timed out after 30 minutes.")
        return 1
    except Exception as exc:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Failed to start training flow: {exc}")
        return 1


def check_and_trigger(result_path: Path = DRIFT_RESULT_PATH) -> bool:
    """Read drift result and trigger retraining if drift is detected.

    Args:
        result_path: Path to the drift result JSON file.

    Returns:
        True if retraining was triggered, False otherwise.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Retrain Trigger ===")

    try:
        result = read_drift_result(result_path)
    except FileNotFoundError as exc:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {exc}")
        sys.exit(1)

    drift_detected = result.get("drift_detected", False)
    drift_share = result.get("drift_share", 0.0)
    drift_ts = result.get("timestamp", "unknown")
    drifted_features = result.get("drifted_features", 0)
    total_features = result.get("total_features", 0)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Drift report timestamp: {drift_ts}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Drifted features: {drifted_features}/{total_features} ({drift_share * 100:.1f}%)")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Drift detected: {drift_detected}")

    if drift_detected:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] *** DRIFT DETECTED — initiating automatic retraining ***")
        rc = trigger_retraining()
        if rc == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining completed. Model registry updated.")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retraining failed or incomplete. Manual intervention may be required.")
        return True
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No drift detected — retraining not required.")
        return False


def main() -> None:
    """Entry point: check drift result and optionally retrain."""
    triggered = check_and_trigger()
    sys.exit(0)


if __name__ == "__main__":
    main()
