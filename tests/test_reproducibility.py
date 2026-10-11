
import pandas as pd

from ml.reproducibility import hash_dataframe, hash_config


def test_dataset_hash_is_repeatable():
    X = pd.DataFrame({"age": [20, 30], "city": ["Delhi", "Mumbai"]})
    y = pd.Series(["A", "B"], name="target")

    assert hash_dataframe(X, y) == hash_dataframe(X, y)


def test_dataset_hash_changes_when_data_changes():
    X1 = pd.DataFrame({"feature": [1, 2]})
    X2 = pd.DataFrame({"feature": [1, 3]})
    y = pd.Series([0, 1])

    assert hash_dataframe(X1, y) != hash_dataframe(X2, y)


def test_config_hash_is_repeatable():
    config = {"model": "random_forest", "seed": 42}

    assert hash_config(config) == hash_config(config)
