
import traceback
from uuid import uuid4

import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.artifacts import save_model
from ml.config import ModelConfig, TrainingConfig
from ml.config_loader import load_config
from ml.data.preprocessing import create_preprocessor
from ml.evaluation.evaluator import evaluate_model
from ml.reproducibility import (
    hash_config,
    hash_dataframe,
    set_random_seed,
)
from ml.tracking import log_training_results, start_run
from ml.training.model_factory import create_model
from ml.training.trainer import train
from pathlib import Path
from ml.lineage import save_lineage, update_lineage_status


def train_model(
    config_path="configs/iris.yaml",
    artifact_dir=None,
    lineage_dir=None,
):
    artifact_dir = Path(artifact_dir) if artifact_dir else Path("artifacts")
    lineage_dir = Path(lineage_dir) if lineage_dir else Path("lineage")
    config = load_config(config_path)

    training_config = TrainingConfig(
        random_seed=config["training"]["random_seed"],
        test_size=config["training"]["test_size"],
    )
    set_random_seed(training_config.random_seed)

    model_settings = config["model"]
    model_parameters = dict(model_settings["parameters"])
    model_parameters["random_state"] = training_config.random_seed

    model_config = ModelConfig(
        model_type=model_settings["model_type"],
        task_type=model_settings["task_type"],
        parameters=model_parameters,
    )
    feature_columns = None
    
    # Load dataset
    dataset_settings = config["dataset"]
    dataset_name = dataset_settings["name"]

    if dataset_name == "iris":
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target

    elif dataset_name == "csv":
        from ml.data.loader import load_dataset

        X, y = load_dataset(
            dataset_settings["path"],
            dataset_settings["target_column"],
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    feature_columns = X.columns.tolist()
    
    # Fingerprint the dataset and configuration
    dataset_hash = hash_dataframe(X, y)
    config_hash = hash_config(config)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config.test_size,
        random_state=training_config.random_seed,
        stratify=y,
    )

    preprocessor = create_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", create_model(model_config)),
        ]
    )

    # Create a unique ID for this training run
    run_id = str(uuid4())
    lineage_record = None

    # Track the complete training run in MLflow
    with start_run(
        experiment_name="modellore-training",
        run_name=f"{model_config.model_type}-{dataset_name}",
    ) as mlflow_run:
        mlflow.set_tag("dataset_hash", dataset_hash)
        mlflow.set_tag("config_hash", config_hash)
        mlflow.set_tag("lineage_run_id", run_id)

        try:
            # Log training configuration
            mlflow.log_params({
                "dataset_name": dataset_name,
                "random_seed": training_config.random_seed,
                "test_size": training_config.test_size,
                "model_type": model_config.model_type,
                "task_type": model_config.task_type,
                **model_config.parameters,
            })

            model = train(model, X_train, y_train)

            # Evaluate model
            predictions = model.predict(X_test)
            metrics = evaluate_model(
                task_type=model_config.task_type,
                y_true=y_test,
                y_pred=predictions,
            )

            mlflow.log_metrics(metrics)

            # Save a unique model artifact
            model_path = save_model(
                model,
                f"{model_config.model_type}_{dataset_name}_{run_id}",
                artifact_dir=artifact_dir,
            )

            # Log both the saved file and MLflow model
            log_training_results(
                parameters={},
                metrics={},
                model=model,
                artifact_path=str(model_path),
            )

            # Save successful lineage
            lineage_record = save_lineage(
                dataset_hash=dataset_hash,
                config_hash=config_hash,
                parameters=model_config.parameters,
                metrics=metrics,
                artifact_path=str(model_path),
                run_id=run_id,
                lineage_dir=lineage_dir,
                feature_columns=feature_columns,
                status="FINISHED",
                mlflow_artifact_uri=mlflow.get_artifact_uri(),
                config_path=str(Path(config_path).resolve()),
                config_snapshot=config,
                
            )

            # Attach lineage JSON to this MLflow run
            mlflow.log_artifact(
                lineage_record["lineage_path"],
                artifact_path="lineage",
            )

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            # Record failure details in MLflow
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("failure_type", type(exc).__name__)
            mlflow.set_tag("failure_message", str(exc))
            mlflow.set_tag(
                "failure_traceback",
                traceback.format_exc()[-4000:],
            )

            # Create a failure record only if no lineage record exists
            if lineage_record is None:
                failure_record = save_lineage(
                    dataset_hash=dataset_hash,
                    config_hash=config_hash,
                    parameters=model_config.parameters,
                    metrics={},
                    artifact_path="",
                    run_id=run_id,
                    lineage_dir=lineage_dir,
                    feature_columns=feature_columns,
                    status="FAILED",
                    error=error_message,
                    config_path=str(Path(config_path).resolve()),
                    config_snapshot=config,
                )
                lineage_path = failure_record["lineage_path"]
            else:
                update_lineage_status(
                    lineage_record["lineage_path"],
                    status="FAILED",
                    error=error_message,
                )
                lineage_path = lineage_record["lineage_path"]
            
            
            
            try:
                mlflow.log_artifact(
                    lineage_path,
                    artifact_path="lineage",
                )
            except Exception:
                pass

            raise

    return {
        "model": type(model).__name__,
        "parameters": model_config.parameters,
        "metrics": metrics,
        "model_path": str(model_path),
        "run_id": lineage_record["run_id"],
        "lineage_path": lineage_record["lineage_path"],
        "mlflow_run_id": mlflow_run.info.run_id,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "feature_columns": feature_columns,
    }


if __name__ == "__main__":
    import sys

    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "configs/iris.yaml"
    )

    result = train_model(config_path=config_path)

    print("Training completed.")
    print(f"Model: {result['model']}")

    print("\nMetrics:")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")

    print(f"\nModel saved to: {result['model_path']}")
    print(f"Lineage record: {result['lineage_path']}")
    print(f"MLflow run ID: {result['mlflow_run_id']}")
