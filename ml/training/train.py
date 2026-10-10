from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ml.config import ModelConfig
from ml.training.model_factory import create_model
from ml.training.trainer import train
from ml.evaluation.evaluator import evaluate_model
from ml.artifacts import save_model
from ml.config import TrainingConfig
from ml.reproducibility import set_random_seed

def train_model():
    training_config = TrainingConfig()
    set_random_seed(training_config.random_seed)
    # 1. Load dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config.test_size,
        random_state=training_config.random_seed,
        stratify=y,
    )

    # 3. Create model
    model_config = ModelConfig(
    model_type="random_forest",
    task_type="classification",
    parameters={
        "n_estimators": 50,
        "max_depth": 5,
        "random_state": training_config.random_seed,
    },
)

    model = create_model(model_config)

    # 4. Train
    model = train(model, X_train, y_train)

    # 5. Predict
    predictions = model.predict(X_test)

    metrics = evaluate_model(
    task_type=model_config.task_type,
    y_true=y_test,
    y_pred=predictions,
)
    model_path = save_model(
        model,
        "random_forest_iris",
    )

    # 8. Return structured result
    return {
    "model": "RandomForestClassifier",
    "parameters": model_config.parameters,
    "metrics": metrics,
    "model_path": str(model_path),
    }


if __name__ == "__main__":
    result = train_model()

    print("Training completed.")
    print(f"Model: {result['model']}")

    print("\nMetrics:")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")

    print(f"\nModel saved to: {result['model_path']}")
    