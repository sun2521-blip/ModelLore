
from sklearn.pipeline import Pipeline
from ml.data.preprocessing import create_preprocessor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ml.artifacts import save_model
from ml.config import ModelConfig, TrainingConfig
from ml.config_loader import load_config
from ml.evaluation.evaluator import evaluate_model
from ml.reproducibility import set_random_seed
from ml.training.model_factory import create_model
from ml.training.trainer import train
from ml.lineage import save_lineage
from ml.reproducibility import hash_config, hash_dataframe, set_random_seed
from uuid import uuid4


def train_model(config_path="configs/iris.yaml"):
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

    #fingerprint dataset and config
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

    model = train(model, X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    metrics = evaluate_model(
        task_type=model_config.task_type,
        y_true=y_test,
        y_pred=predictions,
    )

    # Save model
    run_id = str(uuid4())
    model_path = save_model(
    model,
    f"{model_config.model_type}_{dataset_name}_{run_id}",
)
    lineage_record = save_lineage(
        dataset_hash=dataset_hash,
        config_hash=config_hash,
        parameters=model_config.parameters,
        metrics=metrics,
        artifact_path=str(model_path),
        run_id=run_id,
    )
    return {
        "model": type(model).__name__,
        "parameters": model_config.parameters,
        "metrics": metrics,
        "model_path": str(model_path),
        "run_id": lineage_record["run_id"],
        "lineage_path": lineage_record["lineage_path"],
    }

if __name__ == "__main__":
    result = train_model()

    print("Training completed.")
    print(f"Model: {result['model']}")

    print("\nMetrics:")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")

    print(f"\nModel saved to: {result['model_path']}")
