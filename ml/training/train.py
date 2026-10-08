from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ml.config import ModelConfig
from ml.training.model_factory import create_model
from ml.evaluation.metrics import calculate_classification_metrics
from ml.training.trainer import train


def train_model():
    # 1. Load dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3. Create model
    model_config = ModelConfig(
    model_type="random_forest",
    parameters={
        "n_estimators": 50,
        "max_depth": 5,
        "random_state": 42,
    },
)

    model = create_model(model_config)

    # 4. Train
    model = train(model, X_train, y_train)

    # 5. Predict
    predictions = model.predict(X_test)

    # 6. Evaluate
    metrics = calculate_classification_metrics(
        y_test,
        predictions,
    )

    # 7. Save model
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    model_path = artifact_dir / "random_forest_iris.joblib"

    joblib.dump(model, model_path)

    # 8. Return structured result
    return {
        "model": create_model(ModelConfig(model_type="random_forest", parameters={"n_estimators": 100, "random_state": 42})),
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
    