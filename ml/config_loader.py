
from pathlib import Path

import yaml

from ml.errors import TrainingInputError


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TrainingInputError(
            "Configuration file must contain a YAML mapping."
        )

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    required_sections = {"dataset", "training", "model"}

    missing_sections = required_sections - config.keys()
    if missing_sections:
        raise TrainingInputError(
            f"Missing configuration sections: {sorted(missing_sections)}"
        )

    dataset = config["dataset"]
    training = config["training"]
    model = config["model"]

    if not isinstance(dataset, dict) or not isinstance(
        dataset.get("name"), str
    ) or not dataset["name"].strip():
        raise TrainingInputError("dataset.name must be a non-empty string.")

    if not isinstance(training, dict):
        raise TrainingInputError("training must be a mapping.")

    seed = training.get("random_seed")
    if type(seed) is not int or seed < 0:
        raise TrainingInputError("training.random_seed must be a non-negative integer.")

    test_size = training.get("test_size")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not 0 < test_size < 1
    ):
        raise TrainingInputError("training.test_size must be between 0 and 1.")

    if not isinstance(model, dict):
        raise TrainingInputError("model must be a mapping.")

    if model.get("model_type") != "random_forest":
        raise TrainingInputError("Unsupported model_type. Expected 'random_forest'.")

    if model.get("task_type") != "classification":
        raise TrainingInputError(
            "Unsupported task_type. Expected 'classification'."
        )

    parameters = model.get("parameters")
    if not isinstance(parameters, dict):
        raise TrainingInputError("model.parameters must be a mapping.")

    allowed_parameters = {"n_estimators", "max_depth"}
    unknown_parameters = parameters.keys() - allowed_parameters
    if unknown_parameters:
        raise TrainingInputError(
            f"Unsupported model parameters: {sorted(unknown_parameters)}"
        )

    n_estimators = parameters.get("n_estimators")
    if type(n_estimators) is not int or n_estimators < 1:
        raise TrainingInputError(
            "model.parameters.n_estimators must be a positive integer."
        )

    max_depth = parameters.get("max_depth")
    if max_depth is not None and (
        type(max_depth) is not int or max_depth < 1
    ):
        raise TrainingInputError(
            "model.parameters.max_depth must be a positive integer or null."
        )
