
import hashlib
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ml.inference.predict import (
    ArtifactIntegrityError,
    load_verified_model,
    predict,
)


@pytest.fixture
def saved_model(tmp_path):
    """A real joblib artifact inside an artifact directory, plus its hash."""
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = [0, 0, 1, 1]

    model = Pipeline([
        ("classifier", DummyClassifier(strategy="most_frequent"))
    ])
    model.fit(X, y)

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    model_path = artifact_dir / "test_model.joblib"
    joblib.dump(model, model_path)

    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return model_path, sha256, artifact_dir


def test_load_verified_model_and_predict(saved_model):
    model_path, sha256, artifact_dir = saved_model

    loaded_model = load_verified_model(model_path, sha256, artifact_dir)
    predictions = predict(loaded_model, pd.DataFrame({"feature": [5, 6]}))

    assert len(predictions) == 2
    assert all(value in [0, 1] for value in predictions)


def test_tampered_artifact_is_never_unpickled(saved_model):
    model_path, sha256, artifact_dir = saved_model

    tampered = bytearray(model_path.read_bytes())
    tampered[-1] ^= 0xFF
    model_path.write_bytes(bytes(tampered))

    with patch("ml.inference.predict.joblib.load") as joblib_load:
        with pytest.raises(ArtifactIntegrityError):
            load_verified_model(model_path, sha256, artifact_dir)

    joblib_load.assert_not_called()


def test_wrong_recorded_hash_is_refused(saved_model):
    model_path, _, artifact_dir = saved_model

    with pytest.raises(ArtifactIntegrityError):
        load_verified_model(model_path, "0" * 64, artifact_dir)


@pytest.mark.parametrize("bad_hash", [None, "", "not-a-hash", "A" * 64, "0" * 63])
def test_missing_or_malformed_hash_is_refused(saved_model, bad_hash):
    model_path, _, artifact_dir = saved_model

    with patch("ml.inference.predict.joblib.load") as joblib_load:
        with pytest.raises(ArtifactIntegrityError):
            load_verified_model(model_path, bad_hash, artifact_dir)

    joblib_load.assert_not_called()


def test_artifact_outside_the_artifact_directory_is_refused(saved_model, tmp_path):
    model_path, sha256, artifact_dir = saved_model
    outside = tmp_path / "elsewhere.joblib"
    outside.write_bytes(model_path.read_bytes())

    with pytest.raises(ArtifactIntegrityError):
        load_verified_model(outside, sha256, artifact_dir)

    with pytest.raises(ArtifactIntegrityError):
        load_verified_model(
            artifact_dir / ".." / "elsewhere.joblib", sha256, artifact_dir
        )


def test_missing_artifact_raises_file_not_found(saved_model):
    _, sha256, artifact_dir = saved_model

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_verified_model(artifact_dir / "missing.joblib", sha256, artifact_dir)


def test_predict_rejects_none_model():
    with pytest.raises(ValueError, match="Model cannot be None"):
        predict(None, pd.DataFrame({"feature": [1]}))
