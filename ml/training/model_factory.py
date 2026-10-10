from sklearn.ensemble import RandomForestClassifier

from ml.config import ModelConfig


def create_model(config: ModelConfig):
    if config.model_type == "random_forest":
        return RandomForestClassifier(**config.parameters)

    raise ValueError(f"Unsupported model type: {config.model_type}")