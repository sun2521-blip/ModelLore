
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.errors import TrainingInputError


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    if X.empty or X.shape[1] == 0:
        raise TrainingInputError("Features must contain at least one column.")

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    supported_columns = set(numeric_columns + categorical_columns)
    unsupported_columns = [
        column for column in X.columns if column not in supported_columns
    ]

    if unsupported_columns:
        raise TrainingInputError(
            f"Unsupported feature types: {unsupported_columns}"
        )

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
