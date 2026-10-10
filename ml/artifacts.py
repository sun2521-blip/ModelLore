
from pathlib import Path

import joblib


DEFAULT_ARTIFACT_DIR = Path("artifacts")


def save_model(model, filename, artifact_dir=DEFAULT_ARTIFACT_DIR):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".joblib"):
        filename = f"{filename}.joblib"

    model_path = artifact_dir / filename

    joblib.dump(model, model_path)

    return model_path
