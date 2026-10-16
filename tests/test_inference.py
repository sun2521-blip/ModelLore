
import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ml.inference.predict import load_model, predict


def test_load_model_and_predict(tmp_path):
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = [0, 0, 1, 1]

    model = Pipeline([
        ("classifier", DummyClassifier(strategy="most_frequent"))
    ])
    model.fit(X, y)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    loaded_model = load_model(model_path)
    new_data = pd.DataFrame({"feature": [5, 6]})
    predictions = predict(loaded_model, new_data)

    assert len(predictions) == 2
    assert all(value in [0, 1] for value in predictions)


def test_load_model_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model(tmp_path / "missing.joblib")


def test_predict_rejects_none_model():
    with pytest.raises(ValueError, match="Model cannot be None"):
        predict(None, pd.DataFrame({"feature": [1]}))
