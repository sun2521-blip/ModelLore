
from fastapi.testclient import TestClient

from api.main import app
from unittest.mock import patch, MagicMock
import mlflow

client = TestClient(app)


def test_train_iris_returns_training_results() -> None:
    response = client.post(
        "/train",
        json={"config_name": "iris"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "completed"
    assert body["run_id"] is not None
    assert body["metrics"] is not None


def test_train_rejects_unknown_config_name() -> None:
    response = client.post(
        "/train",
        json={"config_name": "unknown"},
    )

    assert response.status_code == 422


def test_train_handles_training_value_error() -> None:
    with patch(
        "api.main.train_model",
        side_effect=ValueError("Invalid training configuration"),
    ):
        response = client.post(
            "/train",
            json={"config_name": "iris"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid training configuration"


def test_get_run_returns_run_details() -> None:
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"
    mock_run.info.experiment_id = "experiment-1"
    mock_run.info.status = "FINISHED"
    mock_run.data.params = {"n_estimators": "100"}
    mock_run.data.metrics = {"accuracy": 0.95}

    with patch("api.main.mlflow.get_run", return_value=mock_run):
        response = client.get("/runs/test-run-123")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "test-run-123"
    assert body["status"] == "FINISHED"
    assert body["metrics"]["accuracy"] == 0.95


def test_get_run_returns_404_when_missing() -> None:
    with patch(
        "api.main.mlflow.get_run",
        side_effect=mlflow.MlflowException(
            "Run not found",
            error_code="RESOURCE_DOES_NOT_EXIST",
        ),
    ):
        response = client.get("/runs/missing-run")

    assert response.status_code == 404
    assert response.json()["detail"] == "Training run not found"
    
def test_get_run_returns_500_on_tracking_error() -> None:
    with patch(
        "api.main.mlflow.get_run",
        side_effect=mlflow.MlflowException("Tracking server unavailable"),
    ):
        response = client.get("/runs/test-run-123")

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to retrieve training run"