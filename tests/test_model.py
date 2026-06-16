"""Tests for the embedding and classification model components."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "model.pkl"
EMBEDDINGS_PATH = ROOT / "data" / "raw" / "embeddings.npy"
LABELS_PATH = ROOT / "data" / "raw" / "embeddings_labels.npy"
TRAIN_CSV = ROOT / "data" / "raw" / "logs.csv"

VALID_LABELS = {"build_failure", "test_failure", "timeout", "config_error", "infra_failure"}
EMBEDDING_DIM = 384

SAMPLE_LOGS = {
    "build_failure": (
        "npm ERR! code ENOENT\n"
        "npm ERR! syscall open\n"
        "npm ERR! path /app/package.json\n"
        "npm ERR! errno -2\n"
        "ERROR: Build failed with exit code 1"
    ),
    "test_failure": (
        "FAILED tests/test_api.py::test_create_user - AssertionError: assert 500 == 200\n"
        "FAILED tests/test_auth.py::test_login - AssertionError: Expected token got None\n"
        "3 failed, 12 passed in 8.23s"
    ),
    "timeout": (
        "##[error]The job running on runner has exceeded the maximum execution time of 30 minutes.\n"
        "##[error]The runner has received a shutdown signal.\n"
        "Job failed: execution took longer than 1800 seconds"
    ),
    "config_error": (
        "KeyError: 'DATABASE_URL'\n"
        "Traceback (most recent call last):\n"
        "  File '/app/config.py', line 45, in load_config\n"
        "    db_url = os.environ['DATABASE_URL']\n"
        "ERROR: Configuration stage failed"
    ),
    "infra_failure": (
        "Warning: OOMKilled: container exceeded memory limit of 512Mi\n"
        "Pod status: CrashLoopBackOff\n"
        "kubectl get pods: 0/1 Running, 3 Restarts\n"
        "ERROR: Deployment failed. Infrastructure issue detected."
    ),
}


@pytest.fixture(scope="module")
def clf():
    """Load the locally saved classifier model."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def embeddings_and_labels():
    """Load the pre-computed training embeddings and labels."""
    if not EMBEDDINGS_PATH.exists() or not LABELS_PATH.exists():
        pytest.skip("Embeddings not found. Run src/embed.py first.")
    X = np.load(EMBEDDINGS_PATH)
    y = np.load(LABELS_PATH, allow_pickle=True)
    return X, y


class TestEmbeddings:
    """Tests for the sentence-transformer embedding module."""

    def test_embeddings_shape(self, embeddings_and_labels):
        """Embedding matrix must have shape (n, 384)."""
        X, _ = embeddings_and_labels
        assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D"
        assert X.shape[1] == EMBEDDING_DIM, (
            f"Expected {EMBEDDING_DIM} embedding dims, got {X.shape[1]}"
        )

    def test_embeddings_count_matches_csv(self, embeddings_and_labels):
        """Number of embeddings must match the number of rows in the CSV."""
        import pandas as pd
        X, _ = embeddings_and_labels
        df = pd.read_csv(TRAIN_CSV)
        assert len(X) == len(df), (
            f"Embeddings count {len(X)} != CSV rows {len(df)}"
        )

    def test_embeddings_are_float32(self, embeddings_and_labels):
        """Embeddings must be stored as float32."""
        X, _ = embeddings_and_labels
        assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"

    def test_embeddings_are_normalised(self, embeddings_and_labels):
        """Embeddings should be approximately unit-normalised (L2 norm ≈ 1)."""
        X, _ = embeddings_and_labels
        norms = np.linalg.norm(X[:100], axis=1)
        assert np.allclose(norms, 1.0, atol=0.05), (
            f"Embeddings not unit-normalised. Mean norm: {norms.mean():.4f}"
        )

    def test_embed_texts_returns_correct_shape(self):
        """embed_texts() must return (n, 384) for any input list."""
        from src.embed import embed_texts
        texts = ["Build failed with exit code 1", "Test assertion error", "OOMKilled"]
        result = embed_texts(texts)
        assert result.shape == (3, EMBEDDING_DIM), (
            f"Expected (3, {EMBEDDING_DIM}), got {result.shape}"
        )

    def test_embed_texts_single(self):
        """embed_texts() must handle a single-element list correctly."""
        from src.embed import embed_texts
        result = embed_texts(["CrashLoopBackOff detected"])
        assert result.shape == (1, EMBEDDING_DIM)


class TestClassifier:
    """Tests for the trained scikit-learn classifier."""

    def test_predict_returns_valid_label(self, clf):
        """Predictions must be one of the 5 known failure categories."""
        from src.embed import embed_texts
        log = SAMPLE_LOGS["build_failure"]
        embedding = embed_texts([log])
        pred = clf.predict(embedding)[0]
        assert pred in VALID_LABELS, f"Invalid prediction: {pred}"

    def test_confidence_between_0_and_1(self, clf):
        """Predict_proba values must be in [0, 1]."""
        from src.embed import embed_texts
        log = SAMPLE_LOGS["timeout"]
        embedding = embed_texts([log])
        probas = clf.predict_proba(embedding)[0]
        assert all(0.0 <= p <= 1.0 for p in probas), f"Probabilities out of range: {probas}"
        assert abs(sum(probas) - 1.0) < 1e-4, f"Probabilities don't sum to 1: {sum(probas)}"

    def test_all_5_classes_predicted_on_test_set(self, clf, embeddings_and_labels):
        """All 5 failure categories must appear in predictions on the held-out set."""
        from sklearn.model_selection import train_test_split
        X, y = embeddings_and_labels
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        preds = set(clf.predict(X_test))
        missing = VALID_LABELS - preds
        assert not missing, f"Model never predicts these classes: {missing}"

    def test_classifier_has_predict_proba(self, clf):
        """Classifier must expose predict_proba for confidence scores."""
        assert hasattr(clf, "predict_proba"), "Classifier must implement predict_proba"

    def test_sample_predictions_are_sensible(self, clf):
        """Predictions on clearly labeled sample logs should be reasonable.

        We don't enforce exact correctness (the model is small), but each
        prediction must be a valid label.
        """
        from src.embed import embed_texts
        for expected_label, log_text in SAMPLE_LOGS.items():
            embedding = embed_texts([log_text])
            pred = clf.predict(embedding)[0]
            conf = float(clf.predict_proba(embedding).max())
            assert pred in VALID_LABELS, f"Invalid prediction '{pred}' for {expected_label} sample"
            assert 0.0 < conf <= 1.0, f"Confidence out of range for {expected_label}: {conf}"

    def test_batch_predict_consistency(self, clf):
        """Batch predictions must match individual predictions."""
        from src.embed import embed_texts
        logs = list(SAMPLE_LOGS.values())
        batch_emb = embed_texts(logs)
        batch_preds = clf.predict(batch_emb).tolist()

        for i, log in enumerate(logs):
            single_emb = embed_texts([log])
            single_pred = clf.predict(single_emb)[0]
            assert batch_preds[i] == single_pred, (
                f"Batch and single prediction mismatch at index {i}: "
                f"batch={batch_preds[i]}, single={single_pred}"
            )
