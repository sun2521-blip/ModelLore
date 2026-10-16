
import json

from ml.lineage import save_lineage


def test_save_lineage_creates_record(tmp_path):
    record = save_lineage(
        dataset_hash="dataset123",
        config_hash="config456",
        parameters={"n_estimators": 50},
        metrics={"accuracy": 0.9},
        artifact_path="artifacts/model.joblib",
        lineage_dir=tmp_path,
    )

    assert record["run_id"]
    assert record["dataset_hash"] == "dataset123"
    assert record["config_hash"] == "config456"
    assert record["metrics"]["accuracy"] == 0.9

    path = tmp_path / f"{record['run_id']}.json"
    assert path.is_file()

    saved_record = json.loads(path.read_text(encoding="utf-8"))
    assert saved_record["run_id"] == record["run_id"]


def test_each_lineage_record_has_unique_run_id(tmp_path):
    arguments = {
        "dataset_hash": "dataset123",
        "config_hash": "config456",
        "parameters": {},
        "metrics": {},
        "artifact_path": "artifacts/model.joblib",
        "lineage_dir": tmp_path,
    }

    first = save_lineage(**arguments)
    second = save_lineage(**arguments)

    assert first["run_id"] != second["run_id"]


def test_lineage_records_git_metadata(tmp_path, monkeypatch):

    monkeypatch.setattr(
        "ml.lineage.get_git_metadata",
        lambda: {
            "git_commit": "abc123",
            "git_dirty": True,
        },
    )

    record = save_lineage(
        dataset_hash="dataset123",
        config_hash="config456",
        parameters={},
        metrics={},
        artifact_path="artifacts/model.joblib",
        lineage_dir=tmp_path,
    )

    assert record["git_commit"] == "abc123"
    assert record["git_dirty"] is True

    saved_record = json.loads(
        (tmp_path / f"{record['run_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_record["git_commit"] == "abc123"
    assert saved_record["git_dirty"] is True


def test_save_lineage_records_failure(tmp_path):
    record = save_lineage(
        dataset_hash="dataset123",
        config_hash="config456",
        parameters={},
        metrics={},
        artifact_path="",
        lineage_dir=tmp_path,
        status="FAILED",
        error="ValueError: unsupported dataset",
    )

    assert record["status"] == "FAILED"
    assert record["error"] == "ValueError: unsupported dataset"

    saved_record = json.loads(
        (tmp_path / f"{record['run_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_record["status"] == "FAILED"
    assert saved_record["error"] == "ValueError: unsupported dataset"


def test_save_lineage_records_mlflow_artifact_uri(tmp_path):
    record = save_lineage(
        dataset_hash="dataset123",
        config_hash="config456",
        parameters={},
        metrics={},
        artifact_path="artifacts/model.joblib",
        lineage_dir=tmp_path,
        mlflow_artifact_uri="file:///tmp/mlruns/123/artifacts",
    )

    assert record["mlflow_artifact_uri"] == (
        "file:///tmp/mlruns/123/artifacts"
    )

    saved_record = json.loads(
        (tmp_path / f"{record['run_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_record["mlflow_artifact_uri"] == (
        "file:///tmp/mlruns/123/artifacts"
    )
