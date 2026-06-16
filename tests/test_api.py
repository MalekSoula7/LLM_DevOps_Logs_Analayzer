"""Tests for the FastAPI application endpoints."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient with a mock model pre-loaded."""
    import numpy as np
    from fastapi.testclient import TestClient

    mock_clf = MagicMock()
    mock_clf.predict.return_value = np.array(["build_failure"])
    mock_clf.predict_proba.return_value = np.array([[0.05, 0.05, 0.05, 0.80, 0.05]])

    with patch("api.main.load_model") as mock_load:
        from api.main import app, _state
        _state["clf"] = mock_clf
        _state["model_version"] = "test:mock_v1"
        _state["mlflow_connected"] = False

        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module")
def metrics_file(tmp_path_factory):
    """Create a temporary metrics.json for testing /metrics endpoint."""
    tmp = tmp_path_factory.mktemp("metrics")
    metrics_data = {
        "best_model": "LogisticRegression",
        "weighted_f1": 0.92,
        "per_class": {
            "build_failure": {"precision": 0.93, "recall": 0.91, "f1": 0.92},
            "test_failure": {"precision": 0.94, "recall": 0.90, "f1": 0.92},
            "timeout": {"precision": 0.91, "recall": 0.93, "f1": 0.92},
            "config_error": {"precision": 0.90, "recall": 0.92, "f1": 0.91},
            "infra_failure": {"precision": 0.92, "recall": 0.91, "f1": 0.915},
        },
        "n_train": 480,
        "n_test": 120,
        "timestamp": "2025-01-01T12:00:00",
        "run_id": "abc123def456",
    }
    p = tmp / "metrics.json"
    p.write_text(json.dumps(metrics_data))
    return p


REALISTIC_BUILD_FAILURE_LOG = (
    "[2024-01-15 10:23:45] Pipeline: build-and-test #1234\n"
    "[2024-01-15 10:23:46] Running: npm install\n"
    "npm ERR! code ENOENT\n"
    "npm ERR! syscall open\n"
    "npm ERR! path /home/runner/work/myapp/package.json\n"
    "npm ERR! errno -2\n"
    "npm ERR! enoent ENOENT: no such file or directory, open 'package.json'\n"
    "[2024-01-15 10:23:47] ERROR: Command failed with exit code 1\n"
    "[2024-01-15 10:23:47] Pipeline run aborted."
)


class TestHealth:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "mlflow_connected" in data

    def test_health_status_is_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_model_loaded_true(self, client):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True


class TestClassify:
    """Tests for POST /classify."""

    def test_classify_returns_200_with_valid_log(self, client):
        with patch("api.main.explain_failure", return_value="Mocked explanation sentence one. Sentence two."):
            resp = client.post("/classify", json={"log_text": REALISTIC_BUILD_FAILURE_LOG})
        assert resp.status_code == 200

    def test_classify_response_structure(self, client):
        with patch("api.main.explain_failure", return_value="Test explanation. Second sentence."):
            resp = client.post("/classify", json={"log_text": REALISTIC_BUILD_FAILURE_LOG})
        data = resp.json()
        assert "predicted_label" in data
        assert "confidence" in data
        assert "explanation" in data
        assert "model_version" in data
        assert "inference_time_ms" in data

    def test_classify_predicted_label_is_valid(self, client):
        valid_labels = {"build_failure", "test_failure", "timeout", "config_error", "infra_failure"}
        with patch("api.main.explain_failure", return_value="Explanation. Second sentence."):
            resp = client.post("/classify", json={"log_text": REALISTIC_BUILD_FAILURE_LOG})
        label = resp.json()["predicted_label"]
        assert label in valid_labels, f"Invalid predicted label: {label}"

    def test_classify_confidence_between_0_and_1(self, client):
        with patch("api.main.explain_failure", return_value="Explanation. Second sentence."):
            resp = client.post("/classify", json={"log_text": REALISTIC_BUILD_FAILURE_LOG})
        conf = resp.json()["confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_classify_inference_time_is_positive(self, client):
        with patch("api.main.explain_failure", return_value="Explanation. Second."):
            resp = client.post("/classify", json={"log_text": REALISTIC_BUILD_FAILURE_LOG})
        ms = resp.json()["inference_time_ms"]
        assert ms > 0, f"Inference time should be positive, got {ms}"

    def test_classify_rejects_empty_log_text(self, client):
        """POST /classify must return 422 for empty or too-short log_text."""
        resp = client.post("/classify", json={"log_text": ""})
        assert resp.status_code == 422

    def test_classify_rejects_short_log_text(self, client):
        """POST /classify must return 422 for log_text shorter than 10 chars."""
        resp = client.post("/classify", json={"log_text": "err"})
        assert resp.status_code == 422

    def test_classify_rejects_missing_body(self, client):
        """POST /classify must return 422 if body is missing."""
        resp = client.post("/classify", json={})
        assert resp.status_code == 422

    def test_classify_returns_503_when_no_model(self):
        """POST /classify must return 503 when model is not loaded."""
        from fastapi.testclient import TestClient
        from api.main import app, _state

        original_clf = _state["clf"]
        try:
            _state["clf"] = None
            with TestClient(app) as c:
                with patch("api.main.explain_failure", return_value="fallback"):
                    resp = c.post("/classify", json={"log_text": REALISTIC_BUILD_FAILURE_LOG})
            assert resp.status_code == 503
        finally:
            _state["clf"] = original_clf

    def test_classify_explanation_is_string(self, client):
        """Explanation field must be a non-empty string."""
        with patch("api.main.explain_failure", return_value="The build failed due to ENOENT. Check the path."):
            resp = client.post("/classify", json={"log_text": REALISTIC_BUILD_FAILURE_LOG})
        explanation = resp.json()["explanation"]
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestMetrics:
    """Tests for GET /metrics."""

    def test_metrics_returns_404_when_no_file(self, client):
        """GET /metrics should return 404 if metrics.json is missing."""
        import api.main as main_module
        original = main_module.METRICS_PATH
        try:
            main_module.METRICS_PATH = Path("/nonexistent/metrics.json")
            resp = client.get("/metrics")
            assert resp.status_code == 404
        finally:
            main_module.METRICS_PATH = original

    def test_metrics_returns_weighted_f1(self, client, metrics_file):
        """GET /metrics must return a weighted_f1 field when metrics.json exists."""
        import api.main as main_module
        original = main_module.METRICS_PATH
        try:
            main_module.METRICS_PATH = metrics_file
            resp = client.get("/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert "weighted_f1" in data
            assert isinstance(data["weighted_f1"], float)
        finally:
            main_module.METRICS_PATH = original

    def test_metrics_response_structure(self, client, metrics_file):
        """GET /metrics response must have all required fields."""
        import api.main as main_module
        original = main_module.METRICS_PATH
        try:
            main_module.METRICS_PATH = metrics_file
            resp = client.get("/metrics")
            data = resp.json()
            for field in ("best_model", "weighted_f1", "per_class", "n_train", "n_test", "timestamp"):
                assert field in data, f"Missing field in /metrics response: {field}"
        finally:
            main_module.METRICS_PATH = original

    def test_metrics_per_class_has_5_labels(self, client, metrics_file):
        """Per-class metrics must cover all 5 failure categories."""
        import api.main as main_module
        original = main_module.METRICS_PATH
        try:
            main_module.METRICS_PATH = metrics_file
            resp = client.get("/metrics")
            data = resp.json()
            assert len(data["per_class"]) == 5
        finally:
            main_module.METRICS_PATH = original


class TestDriftEndpoints:
    """Tests for GET /drift-report."""

    def test_drift_report_404_when_missing(self, client):
        """GET /drift-report returns 404 if no drift result exists."""
        import api.main as main_module
        original = main_module.DRIFT_RESULT_PATH
        try:
            main_module.DRIFT_RESULT_PATH = Path("/nonexistent/drift_result.json")
            resp = client.get("/drift-report")
            assert resp.status_code == 404
        finally:
            main_module.DRIFT_RESULT_PATH = original


class TestPipelineStatus:
    """Tests for GET /pipeline-status."""

    def test_pipeline_status_returns_200(self, client):
        resp = client.get("/pipeline-status")
        assert resp.status_code == 200

    def test_pipeline_status_has_runs_list(self, client):
        resp = client.get("/pipeline-status")
        data = resp.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_pipeline_status_has_total_count(self, client):
        resp = client.get("/pipeline-status")
        data = resp.json()
        assert "total_logs_processed" in data
        assert isinstance(data["total_logs_processed"], int)
