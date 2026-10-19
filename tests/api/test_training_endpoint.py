
from unittest.mock import MagicMock, patch

import mlflow


def test_train_iris_returns_training_results(api_client):
    with patch(
        "api.main.train_model",
        return_value={
            "status": "completed",
            "run_id": "test-lineage-id",
            "mlflow_run_id": "test-mlflow-id",
            "metrics": {"accuracy": 0.95},
            "parameters": {"n_estimators": 50},
            "lineage_path": "lineage/test-lineage-id.json",
        },
    ):
        response = api_client.post(
            "/train",
            json={"config_name": "iris"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["lineage_run_id"] == "test-lineage-id"
    assert body["mlflow_run_id"] == "test-mlflow-id"
    assert body["metrics"]["accuracy"] == 0.95


def test_train_rejects_unknown_config_name(api_client):
    response = api_client.post(
        "/train",
        json={"config_name": "unknown"},
    )

    assert response.status_code == 422


def test_train_handles_training_value_error(api_client):
    with patch(
        "api.main.train_model",
        side_effect=ValueError("Invalid training configuration"),
    ):
        response = api_client.post(
            "/train",
            json={"config_name": "iris"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid training configuration"


def test_get_run_returns_run_details(api_client):
    mock_run = MagicMock()
    mock_run.info.run_id = "test-mlflow-id"
    mock_run.info.experiment_id = "experiment-1"
    mock_run.info.status = "FINISHED"
    mock_run.data.params = {"n_estimators": "100"}
    mock_run.data.metrics = {"accuracy": 0.95}

    with patch("api.main.mlflow.get_run", return_value=mock_run):
        response = api_client.get("/runs/test-mlflow-id")

    assert response.status_code == 200
    body = response.json()
    assert body["mlflow_run_id"] == "test-mlflow-id"
    assert body["status"] == "FINISHED"
    assert body["metrics"]["accuracy"] == 0.95


def test_get_run_returns_404_when_missing(api_client):
    with patch(
        "api.main.mlflow.get_run",
        side_effect=mlflow.MlflowException(
            "Run not found",
            error_code="RESOURCE_DOES_NOT_EXIST",
        ),
    ):
        response = api_client.get("/runs/missing-run")

    assert response.status_code == 404
    assert response.json()["detail"] == "MLflow run not found"


def test_get_run_returns_500_on_tracking_error(api_client):
    with patch(
        "api.main.mlflow.get_run",
        side_effect=mlflow.MlflowException(
            "Tracking server unavailable",
        ),
    ):
        response = api_client.get("/runs/test-mlflow-id")

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to retrieve MLflow run"
