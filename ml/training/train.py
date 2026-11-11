
import math
from uuid import uuid4

import mlflow

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.artifacts import save_model
from ml.config import ModelConfig, TrainingConfig
from ml.config_loader import load_config
from ml.data.preprocessing import create_preprocessor
from ml.errors import TrainingInputError
from ml.evaluation.evaluator import evaluate_model
from ml.reproducibility import (
    hash_config,
    hash_dataframe,
    hash_file,
    set_random_seed,
)
from ml.tracking import log_training_results, start_run
from ml.training.model_factory import create_model
from ml.training.trainer import train
from pathlib import Path
from ml.lineage import save_lineage, update_lineage_status


def split_problem(y, target_name, test_size):
    """Say why y cannot be split into stratified training and test rows, or None if it can.

    The message gives counts only, never the target's values, so it is safe to
    show in the console and store on the job.
    """
    counts = y.value_counts()
    distinct = len(counts)
    rows = len(y)

    seen_once = int((counts < 2).sum())
    if seen_once:
        verb = "appears" if seen_once == 1 else "appear"
        return (
            f"Target column '{target_name}' has {distinct:,} different values, and "
            f"{seen_once:,} of them {verb} on only one row. Each value needs at least "
            "two rows, one to learn from and one to test on. Choose a target column "
            "that holds categories, such as yes/no or a customer segment."
        )

    # The same arithmetic scikit-learn uses for a fractional test_size.
    test_rows = math.ceil(test_size * rows)
    train_rows = rows - test_rows
    if test_rows < distinct:
        return (
            f"Target column '{target_name}' has {distinct:,} different values, but only "
            f"{test_rows:,} rows ({test_size:.0%} of {rows:,}) are held back for testing, "
            "so not every value can be tested. Choose a target with fewer categories, "
            "or add rows."
        )
    if train_rows < distinct:
        return (
            f"Target column '{target_name}' has {distinct:,} different values, but only "
            f"{train_rows:,} rows are left for training once {test_size:.0%} are held "
            "back for testing. Lower training.test_size, or add rows."
        )
    return None


def ensure_splittable(y, target_name, test_size):
    """Stop with a readable TrainingInputError when the split cannot work."""
    problem = split_problem(y, target_name, test_size)
    if problem:
        raise TrainingInputError(problem)


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
        raise TrainingInputError(f"Unsupported dataset: {dataset_name}")

    feature_columns = X.columns.tolist()
    
    # Fingerprint the dataset and configuration
    dataset_hash = hash_dataframe(X, y)
    config_hash = hash_config(config)

    # Split dataset
    ensure_splittable(
        y,
        dataset_settings.get("target_column", "target"),
        training_config.test_size,
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=training_config.test_size,
            random_state=training_config.random_seed,
            stratify=y,
        )
    except ValueError as exc:
        raise TrainingInputError(
            "The target column could not be split into training and test rows. "
            "Each value needs at least two rows."
        ) from exc

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
            artifact_sha256 = hash_file(model_path)

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
                artifact_sha256=artifact_sha256,
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
            # Only messages known to be safe are kept; anything else could
            # carry file paths or dataset values, so just its type is recorded.
            if isinstance(exc, TrainingInputError):
                error_message = f"{type(exc).__name__}: {exc}"
            else:
                error_message = type(exc).__name__

            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("failure_type", type(exc).__name__)

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
        "artifact_sha256": artifact_sha256,
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
