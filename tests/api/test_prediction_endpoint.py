
import json
from uuid import uuid4
from unittest.mock import patch

import numpy as np
import pytest

from api import main


IRIS_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

IRIS_FEATURES = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
}




@pytest.fixture
def run_setup(tmp_path, monkeypatch, db_session_factory):
    artifact_dir = tmp_path / "artifacts"
    lineage_dir = tmp_path / "lineage"
    artifact_dir.mkdir()
    lineage_dir.mkdir()

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "ARTIFACT_DIR", artifact_dir.resolve())
    monkeypatch.setattr(main, "LINEAGE_DIR", lineage_dir)

    lineage_run_id = uuid4()
    artifact_path = artifact_dir / "model.joblib"
    artifact_path.touch()

    record = {
        "run_id": str(lineage_run_id),
        "status": "FINISHED",
        "artifact_path": "artifacts/model.joblib",
        "feature_columns": IRIS_COLUMNS,
    }

    (lineage_dir / f"{lineage_run_id}.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    with db_session_factory() as db:
        job = main.TrainingJob(config_name="iris", status="FINISHED")
        db.add(job)
        db.flush()

        model_record = main.ModelRecord(
            training_job_id=job.id,
            lineage_run_id=lineage_run_id,
            feature_columns=IRIS_COLUMNS,
            name="iris",
            artifact_path="artifacts/model.joblib",
            metrics={},
        )
        db.add(model_record)
        db.commit()

        job_id = job.id
        model_id = model_record.id

    return (
        lineage_run_id,
        artifact_path,
        record,
        lineage_dir,
        db_session_factory,
        job_id,
        model_id,
    )


def test_predict_returns_predictions(run_setup, api_client):
    lineage_run_id, artifact_path, _, *_ = run_setup

    with (
        patch("api.main.load_model", return_value=object()) as load_model,
        patch("api.main.predict", return_value=np.array([0])) as predict,
    ):
        response = api_client.post(
            "/predict",
            json={
                "lineage_run_id": str(lineage_run_id),
                "features": [IRIS_FEATURES],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "lineage_run_id": str(lineage_run_id),
        "predictions": [0],
        "count": 1,
    }
    load_model.assert_called_once_with(artifact_path.resolve())
    predict.assert_called_once()


def test_predict_supports_multiple_rows(run_setup, api_client):
    lineage_run_id, _, _, *_ = run_setup

    second_row = {
        "sepal length (cm)": 6.0,
        "sepal width (cm)": 2.9,
        "petal length (cm)": 4.5,
        "petal width (cm)": 1.5,
    }

    with (
        patch("api.main.load_model", return_value=object()),
        patch("api.main.predict", return_value=np.array([0, 1])),
    ):
        response = api_client.post(
            "/predict",
            json={
                "lineage_run_id": str(lineage_run_id),
                "features": [IRIS_FEATURES, second_row],
            },
        )

    assert response.status_code == 200
    assert response.json()["predictions"] == [0, 1]
    assert response.json()["count"] == 2


def test_predict_returns_404_for_unknown_run(run_setup, api_client):
    with patch("api.main.load_model") as load_model:
        response = api_client.post(
            "/predict",
            json={
                "lineage_run_id": str(uuid4()),
                "features": [IRIS_FEATURES],
            },
        )

    assert response.status_code == 404
    load_model.assert_not_called()


def test_predict_returns_404_for_missing_artifact(run_setup, api_client):
    lineage_run_id, artifact_path, _, *_ = run_setup
    artifact_path.unlink()

    with patch(
        "api.main.load_model",
        side_effect=FileNotFoundError("Missing model"),
    ):
        response = api_client.post(
            "/predict",
            json={
                "lineage_run_id": str(lineage_run_id),
                "features": [IRIS_FEATURES],
            },
        )

    assert response.status_code == 404


def test_predict_rejects_empty_features(run_setup, api_client):
    lineage_run_id, _, _, *_ = run_setup

    response = api_client.post(
        "/predict",
        json={
            "lineage_run_id": str(lineage_run_id),
            "features": [],
        },
    )

    assert response.status_code == 422


def test_predict_rejects_missing_feature_column(run_setup, api_client):
    lineage_run_id, _, _, *_ = run_setup

    incomplete_features = dict(IRIS_FEATURES)
    incomplete_features.pop("petal width (cm)")

    response = api_client.post(
        "/predict",
        json={
            "lineage_run_id": str(lineage_run_id),
            "features": [incomplete_features],
        },
    )

    assert response.status_code == 422
    assert "expected_columns" in response.json()["detail"]


def test_predict_rejects_extra_feature_column(run_setup, api_client):
    lineage_run_id, _, _, *_ = run_setup

    invalid_features = {
        **IRIS_FEATURES,
        "unexpected_column": 10,
    }

    response = api_client.post(
        "/predict",
        json={
            "lineage_run_id": str(lineage_run_id),
            "features": [invalid_features],
        },
    )

    assert response.status_code == 422


def test_predict_rejects_unsuccessful_training_run(
    run_setup,
    api_client,
):
    lineage_run_id, _, _, _, db_session_factory, job_id, _ = run_setup

    with db_session_factory() as db:
        job = db.get(main.TrainingJob, job_id)
        job.status = "FAILED"
        db.commit()

    with patch("api.main.load_model") as load_model:
        response = api_client.post(
            "/predict",
            json={
                "lineage_run_id": str(lineage_run_id),
                "features": [IRIS_FEATURES],
            },
        )

    assert response.status_code == 409
    load_model.assert_not_called()


def test_predict_rejects_lineage_without_feature_metadata(
    run_setup,
    api_client,
):
    lineage_run_id, _, _, _, db_session_factory, _, model_id = run_setup

    with db_session_factory() as db:
        model_record = db.get(main.ModelRecord, model_id)
        model_record.feature_columns = None
        db.commit()

    response = api_client.post(
        "/predict",
        json={
            "lineage_run_id": str(lineage_run_id),
            "features": [IRIS_FEATURES],
        },
    )

    assert response.status_code == 409


def test_predict_rejects_artifact_outside_artifact_directory(
    run_setup,
    api_client,
):
    lineage_run_id, _, _, _, db_session_factory, _, model_id = run_setup

    with db_session_factory() as db:
        model_record = db.get(main.ModelRecord, model_id)
        model_record.artifact_path = "configs/iris.yaml"
        db.commit()

    with patch("api.main.load_model") as load_model:
        response = api_client.post(
            "/predict",
            json={
                "lineage_run_id": str(lineage_run_id),
                "features": [IRIS_FEATURES],
            },
        )

    assert response.status_code == 500
    load_model.assert_not_called()
