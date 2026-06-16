"""Tests for the synthetic log data generator."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_RAW = ROOT / "data" / "raw"
DATA_DRIFT = ROOT / "data" / "drift"
TRAIN_CSV = DATA_RAW / "logs.csv"
DRIFT_CSV = DATA_DRIFT / "logs_drift.csv"

EXPECTED_LABELS = {"build_failure", "test_failure", "timeout", "config_error", "infra_failure"}
TOTAL_TRAINING = 600
TOTAL_DRIFT = 150


@pytest.fixture(scope="module", autouse=True)
def ensure_data_generated():
    """Generate the data if not already present before running tests."""
    if not TRAIN_CSV.exists() or not DRIFT_CSV.exists():
        from src.generate_data import main
        main()


def _load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def _load_drift() -> pd.DataFrame:
    return pd.read_csv(DRIFT_CSV)


class TestTrainingDataShape:
    """Tests for the training dataset size and structure."""

    def test_exact_sample_count(self):
        """Training dataset must have exactly 600 samples."""
        df = _load_train()
        assert len(df) == TOTAL_TRAINING, (
            f"Expected {TOTAL_TRAINING} training samples, got {len(df)}"
        )

    def test_required_columns_present(self):
        """Training CSV must have id, log_text, label, timestamp columns."""
        df = _load_train()
        for col in ("id", "log_text", "label", "timestamp"):
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_log_text(self):
        """No log_text values may be null or empty."""
        df = _load_train()
        assert df["log_text"].isnull().sum() == 0, "Found null values in log_text"
        assert (df["log_text"].str.strip() == "").sum() == 0, "Found empty strings in log_text"

    def test_no_null_labels(self):
        """No label values may be null."""
        df = _load_train()
        assert df["label"].isnull().sum() == 0, "Found null values in label"

    def test_all_expected_labels_present(self):
        """All 5 expected failure categories must appear in the training data."""
        df = _load_train()
        actual_labels = set(df["label"].unique())
        assert actual_labels == EXPECTED_LABELS, (
            f"Label mismatch. Expected: {EXPECTED_LABELS}, Got: {actual_labels}"
        )


class TestLabelDistribution:
    """Tests for balanced class distribution."""

    def test_each_class_between_100_and_140(self):
        """Each class must have between 100 and 140 samples."""
        df = _load_train()
        counts = df["label"].value_counts()
        for label in EXPECTED_LABELS:
            count = counts.get(label, 0)
            assert 100 <= count <= 140, (
                f"Label '{label}' has {count} samples, expected 100–140"
            )

    def test_no_dominant_class(self):
        """No single class should contain more than 25% of all samples."""
        df = _load_train()
        counts = df["label"].value_counts()
        max_share = counts.max() / len(df)
        assert max_share <= 0.25, (
            f"Class imbalance detected. Max share: {max_share:.1%}"
        )


class TestLogQuality:
    """Tests for the quality and realism of generated log text."""

    def test_log_text_is_multiline(self):
        """At least 90% of log samples should be multi-line (>= 3 lines)."""
        df = _load_train()
        multi_line = df["log_text"].apply(lambda t: len(t.splitlines()) >= 3).sum()
        share = multi_line / len(df)
        assert share >= 0.90, f"Only {share:.1%} of logs are multi-line"

    def test_ids_are_unique(self):
        """Each sample must have a unique id."""
        df = _load_train()
        assert df["id"].nunique() == len(df), "Duplicate IDs found in training data"

    def test_timestamps_are_valid(self):
        """All timestamps must be parseable and within the last 30 days."""
        df = _load_train()
        cutoff = datetime.now() - timedelta(days=32)
        future = datetime.now() + timedelta(days=1)
        for ts_str in df["timestamp"]:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            assert ts >= cutoff, f"Timestamp too old: {ts_str}"
            assert ts <= future, f"Timestamp in the future: {ts_str}"


class TestDriftData:
    """Tests for the drift simulation dataset."""

    def test_drift_sample_count(self):
        """Drift dataset must have exactly 150 samples."""
        df = _load_drift()
        assert len(df) == TOTAL_DRIFT, (
            f"Expected {TOTAL_DRIFT} drift samples, got {len(df)}"
        )

    def test_drift_no_null_log_text(self):
        """No null log_text in drift data."""
        df = _load_drift()
        assert df["log_text"].isnull().sum() == 0

    def test_drift_labels_are_valid(self):
        """All drift sample labels must be from the known label set."""
        df = _load_drift()
        assert set(df["label"].unique()).issubset(EXPECTED_LABELS)

    def test_drift_has_kubernetes_vocabulary(self):
        """Drift logs must contain Kubernetes-specific terminology."""
        df = _load_drift()
        k8s_terms = ["probe", "ImagePull", "Unschedulable", "FailedMount", "Liveness", "Readiness"]
        combined_text = " ".join(df["log_text"].tolist())
        matches = [term for term in k8s_terms if term.lower() in combined_text.lower()]
        assert len(matches) >= 3, (
            f"Drift data lacks Kubernetes vocabulary. Found only: {matches}"
        )

    def test_drift_timestamps_recent(self):
        """Drift timestamps should be within the last 7 days."""
        df = _load_drift()
        cutoff = datetime.now() - timedelta(days=9)
        for ts_str in df["timestamp"]:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            assert ts >= cutoff, f"Drift timestamp too old: {ts_str}"
