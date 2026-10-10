
from pathlib import Path

import joblib


def load_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return joblib.load(model_path)


def predict(model, data):
    if model is None:
        raise ValueError("Model cannot be None.")

    return model.predict(data)
