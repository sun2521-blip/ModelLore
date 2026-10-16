
import mlflow
from ml.tracking import start_run, log_training_results


def test_start_run_creates_active_run(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    with start_run("test-experiment", "test-run") as run:
        assert run.info.run_id
        assert mlflow.active_run() is not None


def test_log_training_results_logs_parameters_and_metrics(tmp_path):
    import joblib
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0], [1]], [0, 1])

    artifact = tmp_path / "dummy.joblib"
    joblib.dump(model, artifact)

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("tracking-helper-test")

    with mlflow.start_run() as run:
        log_training_results(
            parameters={"n_estimators": 50},
            metrics={"accuracy": 0.9},
            model=model,
            artifact_path=str(artifact),
        )

        client = mlflow.tracking.MlflowClient()
        logged_run = client.get_run(run.info.run_id)

        assert logged_run.data.params["n_estimators"] == "50"
        assert logged_run.data.metrics["accuracy"] == 0.9
