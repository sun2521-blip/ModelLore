
import hashlib
import json
from pathlib import Path

import mlflow
import pytest

import ml.training.train as training_module


@pytest.fixture(autouse=True)
def restore_tracking_uri():
    previous_uri = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous_uri)


def test_training_succeeds_end_to_end(tmp_path, monkeypatch):
    tracking_db = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db.as_posix()}"
    )
    monkeypatch.chdir(tmp_path)

    config_path = tmp_path / "iris.yaml"
    config_path.write_text(
        """
dataset:
  name: iris
training:
  random_seed: 42
  test_size: 0.2
model:
  model_type: random_forest
  task_type: classification
  parameters:
    n_estimators: 5
    max_depth: 3
""",
        encoding="utf-8",
    )

    result = training_module.train_model(
        config_path=str(config_path)
    )

    assert result["metrics"]
    assert result["dataset_hash"]
    assert result["config_hash"]
    assert Path(result["model_path"]).is_file()
    assert result["artifact_sha256"] == hashlib.sha256(
        Path(result["model_path"]).read_bytes()
    ).hexdigest()

    lineage_path = Path(result["lineage_path"])
    record = json.loads(lineage_path.read_text(encoding="utf-8"))

    assert record["status"] == "FINISHED"
    assert record["artifact_sha256"] == result["artifact_sha256"]
    assert record["config_snapshot"]["dataset"]["name"] == "iris"

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(result["mlflow_run_id"])
    assert run.info.status == "FINISHED"


def test_training_failure_records_lineage(tmp_path, monkeypatch):
    tracking_db = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")

    monkeypatch.chdir(tmp_path)

    # Provide a minimal config and dataset so the test reaches the
    # active MLflow run without depending on the real project files.
    config = {
        "training": {"random_seed": 42, "test_size": 0.2},
        "model": {
            "model_type": "random_forest",
            "task_type": "classification",
            "parameters": {"n_estimators": 5},
        },
        "dataset": {"name": "iris"},
    }

    monkeypatch.setattr(training_module, "load_config", lambda _: config)

    class DummyConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(training_module, "TrainingConfig", DummyConfig)
    monkeypatch.setattr(training_module, "ModelConfig", DummyConfig)
    monkeypatch.setattr(training_module, "set_random_seed", lambda _: None)

    import pandas as pd

    X = pd.DataFrame({"feature": range(20)})
    y = pd.Series([0, 1] * 10)

    class DummyIris:
        data = X
        target = y

    monkeypatch.setattr(
        training_module,
        "load_iris",
        lambda as_frame=True: DummyIris(),
    )
    monkeypatch.setattr(
        training_module,
        "hash_dataframe",
        lambda X, y: "dataset-hash",
    )
    monkeypatch.setattr(
        training_module,
        "hash_config",
        lambda config: "config-hash",
    )

    # Keep the test independent of sklearn preprocessing and model creation.
    monkeypatch.setattr(
        training_module,
        "train_test_split",
        lambda X, y, **kwargs: (X, X, y, y),
    )
    monkeypatch.setattr(
        training_module,
        "create_preprocessor",
        lambda X: "dummy-preprocessor",
    )
    monkeypatch.setattr(
        training_module,
        "create_model",
        lambda config: "dummy-model",
    )

    class DummyPipeline:
        def __init__(self, steps):
            self.steps = steps

    monkeypatch.setattr(training_module, "Pipeline", DummyPipeline)

    def fail_training(model, X, y):
        raise ValueError("simulated training failure")

    monkeypatch.setattr(training_module, "train", fail_training)

    with pytest.raises(ValueError, match="simulated training failure"):
        training_module.train_model()

    lineage_files = list((tmp_path / "lineage").glob("*.json"))
    assert len(lineage_files) == 1

    record = json.loads(lineage_files[0].read_text(encoding="utf-8"))
    assert record["status"] == "FAILED"
    assert record["error"] == (
        "ValueError: simulated training failure"
    )

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[
            client.get_experiment_by_name("modellore-training").experiment_id
        ]
    )
    assert runs
    assert runs[0].data.tags["failure_type"] == "ValueError"
