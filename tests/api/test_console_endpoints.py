"""The API fields the console relies on, and the lineage endpoint."""

import json
import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from api import lineage_view, main
from db.models import Dataset, ModelRecord, TrainingJob


DATASET_HASH = "a" * 64
CONFIG_HASH = "b" * 64
ARTIFACT_HASH = "c" * 64
LOCK_HASH = "d" * 64
MLFLOW_RUN_ID = "0123456789abcdef0123456789abcdef"
GIT_COMMIT = "e" * 40


def lineage_record(run_id, **overrides):
    """A lineage file as the pipeline writes it, server paths included."""
    record = {
        "run_id": str(run_id),
        "created_at": "2026-11-10T12:00:00+00:00",
        "dataset_hash": DATASET_HASH,
        "config_hash": CONFIG_HASH,
        "parameters": {"n_estimators": 50, "max_depth": 5, "random_state": 42},
        "metrics": {"accuracy": 0.9, "f1": 0.89},
        "artifact_path": "D:\\secret\\artifacts\\random_forest_csv.joblib",
        "artifact_sha256": ARTIFACT_HASH,
        "git_commit": GIT_COMMIT,
        "git_dirty": False,
        "environment": {
            "python_version": "3.13.18",
            "scikit_learn_version": "1.9.1",
            "lock_file_sha256": LOCK_HASH,
        },
        "status": "FINISHED",
        "error": None,
        "mlflow_artifact_uri": f"file:///D:/secret/mlruns/1/{MLFLOW_RUN_ID}/artifacts",
        "config_path": "D:\\secret\\data\\uploads\\dataset.yaml",
        "config_snapshot": {
            "dataset": {
                "name": "csv",
                "path": "D:\\secret\\data\\uploads\\dataset.csv",
                "target_column": "species",
            },
            "training": {"random_seed": 42, "test_size": 0.2},
            "model": {
                "model_type": "random_forest",
                "task_type": "classification",
                "parameters": {"n_estimators": 50, "max_depth": 5},
            },
        },
        "feature_columns": ["sepal_length", "sepal_width"],
    }
    record.update(overrides)
    return record


@pytest.fixture
def lineage_dir(tmp_path, monkeypatch):
    directory = tmp_path / "lineage"
    directory.mkdir()
    monkeypatch.setattr(main, "LINEAGE_DIR", directory)
    return directory


@pytest.fixture
def trained(db_session_factory, lineage_dir):
    """A dataset, its finished job, the model it produced, and its lineage file."""
    lineage_run_id = uuid4()

    with db_session_factory() as db:
        dataset = Dataset(
            name="species",
            original_filename="species.csv",
            file_path="D:\\secret\\data\\uploads\\dataset.csv",
            sha256="f" * 64,
            row_count=10,
            column_count=3,
            target_column="species",
        )
        db.add(dataset)
        db.flush()

        job = TrainingJob(
            dataset_id=dataset.id,
            config_name="csv-test",
            status="FINISHED",
            mlflow_run_id=MLFLOW_RUN_ID,
        )
        db.add(job)
        db.flush()

        model = ModelRecord(
            training_job_id=job.id,
            lineage_run_id=lineage_run_id,
            feature_columns=["sepal_length", "sepal_width"],
            name="csv-test",
            artifact_path="D:\\secret\\artifacts\\random_forest_csv.joblib",
            artifact_sha256=ARTIFACT_HASH,
            metrics={"accuracy": 0.9},
        )
        db.add(model)
        db.commit()
        ids = {"dataset": dataset.id, "job": job.id, "model": model.id}

    (lineage_dir / f"{lineage_run_id}.json").write_text(
        json.dumps(lineage_record(lineage_run_id)), encoding="utf-8"
    )
    return lineage_run_id, ids


def test_models_carry_their_trace_fields(api_client, trained):
    lineage_run_id, ids = trained

    listed = api_client.get("/models").json()["items"][0]
    detail = api_client.get(f"/models/{ids['model']}").json()

    for body in (listed, detail):
        assert body["lineage_run_id"] == str(lineage_run_id)
        assert body["dataset_id"] == str(ids["dataset"])
        assert body["feature_columns"] == ["sepal_length", "sepal_width"]
        assert body["artifact_sha256"] == ARTIFACT_HASH
        assert "artifact_path" not in body


def test_jobs_point_to_the_model_they_produced(api_client, trained, db_session_factory):
    lineage_run_id, ids = trained
    with db_session_factory() as db:
        db.add(TrainingJob(config_name="iris", status="FAILED"))
        db.commit()

    items = {item["job_id"]: item for item in api_client.get("/jobs").json()["items"]}
    finished = items[str(ids["job"])]
    failed = next(item for item in items.values() if item["status"] == "FAILED")

    assert finished["model_id"] == str(ids["model"])
    assert finished["lineage_run_id"] == str(lineage_run_id)
    assert failed["model_id"] is None
    assert failed["lineage_run_id"] is None
    assert api_client.get(f"/jobs/{ids['job']}").json()["lineage_run_id"] == str(lineage_run_id)


def test_lineage_returns_the_whole_trace(api_client, trained):
    lineage_run_id, ids = trained

    response = api_client.get(f"/lineage/{lineage_run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["lineage_run_id"] == str(lineage_run_id)
    assert body["status"] == "FINISHED"
    assert body["dataset"] == {
        "fingerprint": DATASET_HASH,
        "source": "csv",
        "target_column": "species",
        "dataset_id": str(ids["dataset"]),
        "original_filename": "species.csv",
        "file_sha256": "f" * 64,
        "row_count": 10,
        "column_count": 3,
    }
    assert body["config"]["fingerprint"] == CONFIG_HASH
    assert body["config"]["test_size"] == 0.2
    assert body["config"]["parameters"]["n_estimators"] == 50
    assert body["run"]["mlflow_run_id"] == MLFLOW_RUN_ID
    assert body["run"]["job_id"] == str(ids["job"])
    assert body["run"]["metrics"] == {"accuracy": 0.9, "f1": 0.89}
    assert body["run"]["git_commit"] == GIT_COMMIT
    assert body["model"]["model_id"] == str(ids["model"])
    assert body["model"]["artifact_sha256"] == ARTIFACT_HASH
    assert body["model"]["environment"]["lock_file_sha256"] == LOCK_HASH


def test_lineage_never_shows_server_paths(api_client, trained):
    lineage_run_id, _ = trained

    text = api_client.get(f"/lineage/{lineage_run_id}").text

    assert "secret" not in text
    assert "mlruns" not in text
    for key in ("config_path", "artifact_path", "mlflow_artifact_uri", '"path"', "error"):
        assert key not in text


def test_lineage_for_a_run_without_a_model(api_client, lineage_dir):
    run_id = uuid4()
    (lineage_dir / f"{run_id}.json").write_text(
        json.dumps(lineage_record(run_id, status="FAILED", artifact_sha256=None)),
        encoding="utf-8",
    )

    body = api_client.get(f"/lineage/{run_id}").json()

    assert body["status"] == "FAILED"
    assert body["model"]["model_id"] is None
    assert body["dataset"]["dataset_id"] is None
    # No database row, so the MLflow run ID comes from the artifact URI.
    assert body["run"]["mlflow_run_id"] == MLFLOW_RUN_ID


def test_lineage_from_an_older_record(api_client, lineage_dir):
    run_id = uuid4()
    old = {
        "run_id": str(run_id),
        "dataset_hash": DATASET_HASH,
        "config_hash": CONFIG_HASH,
        "parameters": {"n_estimators": 50},
        "metrics": {"accuracy": 0.95},
        "artifact_path": "artifacts/random_forest_iris.joblib",
        "status": "FINISHED",
    }
    (lineage_dir / f"{run_id}.json").write_text(json.dumps(old), encoding="utf-8")

    body = api_client.get(f"/lineage/{run_id}").json()

    assert body["model"]["environment"] is None
    assert body["model"]["artifact_sha256"] is None
    assert body["config"]["test_size"] is None
    assert body["run"]["feature_columns"] is None
    assert body["run"]["metrics"] == {"accuracy": 0.95}


def test_lineage_unknown_run_is_404(api_client, lineage_dir):
    assert api_client.get(f"/lineage/{uuid4()}").status_code == 404


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "..%2F..%2Fetc%2Fpasswd", "../secret"])
def test_lineage_rejects_anything_but_a_run_id(api_client, lineage_dir, bad_id):
    assert api_client.get(f"/lineage/{bad_id}").status_code in (404, 422)


def test_lineage_requires_the_api_key(api_client, trained):
    lineage_run_id, _ = trained
    api_client.headers.pop("X-API-Key")

    assert api_client.get(f"/lineage/{lineage_run_id}").status_code == 401


@pytest.mark.parametrize(
    "content",
    ["{not json", json.dumps(["a", "list"]), json.dumps({"run_id": "someone-else"})],
    ids=["corrupt", "not-an-object", "mismatched-id"],
)
def test_unreadable_lineage_gives_a_generic_error(api_client, lineage_dir, caplog, content):
    run_id = uuid4()
    (lineage_dir / f"{run_id}.json").write_text(content, encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="modellore.api"):
        response = api_client.get(f"/lineage/{run_id}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Lineage record could not be read"}
    assert f"lineage_run_id={run_id}" in caplog.text
    assert str(lineage_dir) not in caplog.text


def test_oversized_lineage_is_refused(api_client, trained, monkeypatch):
    lineage_run_id, _ = trained
    monkeypatch.setattr(main, "MAX_LINEAGE_BYTES", 100)

    assert api_client.get(f"/lineage/{lineage_run_id}").status_code == 500


def test_runs_return_only_the_trace_tags(api_client):
    run = MagicMock()
    run.info.run_id = MLFLOW_RUN_ID
    run.info.experiment_id = "1"
    run.info.status = "FINISHED"
    run.data.params = {}
    run.data.metrics = {"accuracy": 0.9}
    run.data.tags = {
        "dataset_hash": DATASET_HASH,
        "config_hash": CONFIG_HASH,
        "lineage_run_id": "some-lineage-id",
        "mlflow.source.name": "D:\\secret\\ml\\training\\train.py",
        "mlflow.user": "someone",
    }

    with patch("api.main.mlflow.get_run", return_value=run):
        body = api_client.get(f"/runs/{MLFLOW_RUN_ID}").json()

    assert body["tags"] == {
        "dataset_hash": DATASET_HASH,
        "config_hash": CONFIG_HASH,
        "lineage_run_id": "some-lineage-id",
    }


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (f"file:///D:/x/mlruns/1/{MLFLOW_RUN_ID}/artifacts", MLFLOW_RUN_ID),
        (f"mlflow-artifacts:/1/{MLFLOW_RUN_ID}/artifacts", MLFLOW_RUN_ID),
        (f"D:\\x\\mlruns\\1\\{MLFLOW_RUN_ID}\\artifacts", MLFLOW_RUN_ID),
        ("file:///D:/x/mlruns/1/not-a-run/artifacts", None),
        (None, None),
    ],
)
def test_mlflow_run_id_is_the_only_thing_taken_from_the_uri(uri, expected):
    assert lineage_view.mlflow_run_id_from_uri(uri) == expected


def test_public_view_drops_values_that_are_not_what_they_claim():
    view = lineage_view.public_lineage(
        lineage_record(
            uuid4(),
            dataset_hash="D:\\secret\\not-a-hash",
            git_commit="rm -rf /",
            metrics={"accuracy": 0.9, "note": "D:\\secret"},
            parameters={"n_estimators": 50, "path": "D:\\secret\\" + "x" * 80},
        )
    )

    assert view["dataset"]["fingerprint"] is None
    assert view["run"]["git_commit"] is None
    assert view["run"]["metrics"] == {"accuracy": 0.9}
    assert "path" not in view["config"]["parameters"]
