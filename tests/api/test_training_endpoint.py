from uuid import uuid4
from unittest.mock import MagicMock, patch

import mlflow
import pytest
from sqlalchemy import select
from db.models import ModelRecord, TrainingJob

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


MLFLOW_RUN_ID = "0123456789abcdef0123456789abcdef"


def test_train_iris_returns_training_results(api_client):
    lineage_run_id = str(uuid4())

    with patch(
        "api.main.train_model",
        return_value={
            "status": "completed",
            "run_id": lineage_run_id,
            "mlflow_run_id": MLFLOW_RUN_ID,
            "metrics": {"accuracy": 0.95},
            "parameters": {"n_estimators": 50},
            "model_path": "artifacts/model.joblib",
            "artifact_sha256": "a" * 64,
            "lineage_path": f"lineage/{lineage_run_id}.json",
            "feature_columns": ["sepal length (cm)"],
        },
    ):
        response = api_client.post(
            "/train",
            json={"config_name": "iris"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["lineage_run_id"] == lineage_run_id
    assert body["mlflow_run_id"] == MLFLOW_RUN_ID
    assert body["metrics"]["accuracy"] == 0.95


def test_train_rejects_unknown_config_name(api_client):
    response = api_client.post(
        "/train",
        json={"config_name": "unknown"},
    )

    assert response.status_code == 422



def test_train_handles_training_value_error(
    api_client, db_session_factory
):
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

    with db_session_factory() as session:
        failed_job = session.scalar(
            select(TrainingJob).where(
                TrainingJob.error_message == "Invalid training configuration"
            )
        )

        assert failed_job is not None
        assert failed_job.status == "FAILED"
        assert failed_job.completed_at is not None


def test_get_run_returns_run_details(api_client):
    mock_run = MagicMock()
    mock_run.info.run_id = MLFLOW_RUN_ID
    mock_run.info.experiment_id = "experiment-1"
    mock_run.info.status = "FINISHED"
    mock_run.data.params = {"n_estimators": "100"}
    mock_run.data.metrics = {"accuracy": 0.95}

    with patch("api.main.mlflow.get_run", return_value=mock_run):
        response = api_client.get(f"/runs/{MLFLOW_RUN_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["mlflow_run_id"] == MLFLOW_RUN_ID
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
        response = api_client.get(f"/runs/{MLFLOW_RUN_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "MLflow run not found"


def test_get_run_returns_500_on_tracking_error(api_client):
    with patch(
        "api.main.mlflow.get_run",
        side_effect=mlflow.MlflowException(
            "Tracking server unavailable",
        ),
    ):
        response = api_client.get(f"/runs/{MLFLOW_RUN_ID}")

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to retrieve MLflow run"



def test_train_marks_job_failed_when_model_persistence_fails(
    api_client,
    db_session_factory,
):
    lineage_run_id = str(uuid4())

    training_result = {
        "run_id": lineage_run_id,
        "mlflow_run_id": "test-mlflow-id",
        "metrics": {"accuracy": 0.9},
        "parameters": {},
        "model_path": "artifacts/model.joblib",
        "artifact_sha256": "a" * 64,
        "lineage_path": f"lineage/{lineage_run_id}.json",
        "feature_columns": ["feature"],
    }

    original_commit = Session.commit
    commit_count = 0

    def fail_final_commit(session):
        nonlocal commit_count
        commit_count += 1

        if commit_count == 2:
            raise SQLAlchemyError("Simulated persistence failure")

        return original_commit(session)

    with (
        patch("api.main.train_model", return_value=training_result),
        patch.object(Session, "commit", new=fail_final_commit),
    ):
        response = api_client.post(
            "/train",
            json={"config_name": "iris"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Training completed but its database record could not be saved"
    )

    with db_session_factory() as session:
        failed_job = session.scalar(
            select(TrainingJob).where(
                TrainingJob.error_message
                == "Training completed but its database record could not be saved"
            )
        )

        assert failed_job is not None
        assert failed_job.status == "FAILED"
        assert failed_job.completed_at is not None


UNUSABLE_MODEL_MESSAGE = "Training completed but did not produce a usable model"


def successful_training_result(**overrides):
    result = {
        "run_id": str(uuid4()),
        "mlflow_run_id": "test-mlflow-id",
        "metrics": {"accuracy": 0.9},
        "parameters": {},
        "model_path": "artifacts/model.joblib",
        "artifact_sha256": "a" * 64,
        "lineage_path": "lineage/not-read-by-the-endpoint.json",
        "feature_columns": ["feature"],
    }
    result.update(overrides)
    return result


def test_train_registers_model_from_training_result(
    api_client,
    db_session_factory,
):
    training_result = successful_training_result()

    with patch("api.main.train_model", return_value=training_result):
        response = api_client.post("/train", json={"config_name": "iris"})

    assert response.status_code == 200

    with db_session_factory() as session:
        job = session.scalar(select(TrainingJob))
        model = session.scalar(select(ModelRecord))

        assert job.status == "FINISHED"
        assert model is not None
        assert model.training_job_id == job.id
        assert str(model.lineage_run_id) == training_result["run_id"]
        assert model.artifact_path == training_result["model_path"]
        assert model.feature_columns == ["feature"]


UNUSABLE_RESULTS = {
    "missing-model-path": {"model_path": None},
    "missing-artifact-hash": {"artifact_sha256": None},
    "invalid-lineage-run-id": {"run_id": "not-a-uuid"},
    "missing-feature-columns": {"feature_columns": None},
}


@pytest.mark.parametrize(
    "overrides",
    list(UNUSABLE_RESULTS.values()),
    ids=list(UNUSABLE_RESULTS),
)
def test_train_marks_job_failed_when_no_usable_model(
    api_client,
    db_session_factory,
    overrides,
):
    with patch(
        "api.main.train_model",
        return_value=successful_training_result(**overrides),
    ):
        response = api_client.post("/train", json={"config_name": "iris"})

    assert response.status_code == 500
    assert response.json()["detail"] == UNUSABLE_MODEL_MESSAGE

    with db_session_factory() as session:
        job = session.scalar(select(TrainingJob))

        assert job.status == "FAILED"
        assert job.error_message == UNUSABLE_MODEL_MESSAGE
        assert job.completed_at is not None
        assert session.scalar(select(ModelRecord)) is None
