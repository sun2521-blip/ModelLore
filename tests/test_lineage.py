
import hashlib
import json
import platform

import pytest
import sklearn

from ml import lineage as lineage_module
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



@pytest.mark.parametrize(
    "launch_error",
    [FileNotFoundError("git not installed"), PermissionError("[WinError 5] Access is denied")],
)
def test_save_lineage_survives_git_that_cannot_start(tmp_path, monkeypatch, launch_error):
    def refuse_to_start(*args, **kwargs):
        raise launch_error

    monkeypatch.setattr(lineage_module.subprocess, "run", refuse_to_start)

    record = lineage_module.save_lineage(
        dataset_hash="dataset-hash",
        config_hash="config-hash",
        parameters={},
        metrics={},
        artifact_path="",
        lineage_dir=tmp_path,
    )

    assert record["git_commit"] is None
    assert record["git_dirty"] is None
    assert (tmp_path / f"{record['run_id']}.json").is_file()


def test_lineage_records_the_runtime_environment(tmp_path):
    record = save_lineage(
        dataset_hash="dataset-hash",
        config_hash="config-hash",
        parameters={},
        metrics={},
        artifact_path="",
        lineage_dir=tmp_path,
    )

    saved = json.loads((tmp_path / f"{record['run_id']}.json").read_text(encoding="utf-8"))
    environment = saved["environment"]

    assert environment["python_version"] == platform.python_version()
    assert environment["scikit_learn_version"] == sklearn.__version__
    assert environment["lock_file_sha256"] == hashlib.sha256(
        lineage_module.LOCK_FILE.read_bytes()
    ).hexdigest()


def test_lineage_environment_tolerates_a_missing_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lineage_module, "LOCK_FILE", tmp_path / "no-such.lock")

    assert lineage_module.get_runtime_environment()["lock_file_sha256"] is None
