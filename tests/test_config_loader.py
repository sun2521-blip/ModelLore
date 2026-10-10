
import pytest

from ml.config_loader import load_config, validate_config


def valid_config():
    return {
        "dataset": {"name": "iris"},
        "training": {
            "random_seed": 42,
            "test_size": 0.2,
        },
        "model": {
            "model_type": "random_forest",
            "task_type": "classification",
            "parameters": {
                "n_estimators": 50,
                "max_depth": 5,
            },
        },
    }


def test_load_valid_yaml():
    config = load_config("configs/iris.yaml")

    assert config["dataset"]["name"] == "iris"
    assert config["training"]["random_seed"] == 42
    assert config["model"]["model_type"] == "random_forest"


def test_reject_invalid_test_size():
    config = valid_config()
    config["training"]["test_size"] = 2

    with pytest.raises(ValueError, match="test_size"):
        validate_config(config)


def test_reject_unsupported_model():
    config = valid_config()
    config["model"]["model_type"] = "unsupported_model"

    with pytest.raises(ValueError, match="model_type"):
        validate_config(config)


def test_reject_missing_section():
    config = valid_config()
    del config["model"]

    with pytest.raises(ValueError, match="Missing configuration sections"):
        validate_config(config)


def test_reject_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
