
import pandas as pd

from ml.data.preprocessing import create_preprocessor


def test_preprocessor_handles_numeric_and_categorical_features():
    X = pd.DataFrame({
        "age": [20, None, 30, 40],
        "city": ["Delhi", "Mumbai", None, "Delhi"],
    })

    preprocessor = create_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 4
    assert transformed.shape[1] >= 2
    assert not pd.isna(transformed).any()


def test_preprocessor_handles_unseen_categories():
    X_train = pd.DataFrame({
        "city": ["Delhi", "Mumbai", "Delhi"],
    })
    X_test = pd.DataFrame({
        "city": ["Kolkata"],
    })

    preprocessor = create_preprocessor(X_train)
    preprocessor.fit(X_train)

    transformed = preprocessor.transform(X_test)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == 2
