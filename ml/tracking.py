import mlflow
import mlflow.sklearn


def start_run(experiment_name: str, run_name: str):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_training_results(
    parameters: dict,
    metrics: dict,
    model,
    artifact_path: str,
) -> None:
    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(artifact_path, artifact_path="model_files")
    mlflow.sklearn.log_model(model, name="trained_model",
                            serialization_format="pickle",)
