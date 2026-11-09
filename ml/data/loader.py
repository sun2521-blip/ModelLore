import csv
from pathlib import Path

import pandas as pd

from ml.errors import TrainingInputError


def load_dataset(
    file_path: str | Path,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    if not target_column or not target_column.strip():
        raise TrainingInputError("Target column name cannot be empty.")

    if file_path.suffix.lower() != ".csv":
        raise TrainingInputError("Only CSV datasets are supported.")
    # Check original CSV headers before pandas can rename duplicates.
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file), None)

    if header is None:
        raise TrainingInputError("Dataset is empty.")

    if len(header) != len(set(header)):
        raise TrainingInputError("Dataset contains duplicate column names.")

    df = pd.read_csv(file_path)

    if df.empty:
        raise TrainingInputError("Dataset is empty.")

    if df.columns.duplicated().any():
        raise TrainingInputError("Dataset contains duplicate column names.")

    if target_column not in df.columns:
        raise TrainingInputError(
            f"Target column '{target_column}' not found in dataset."
        )

    if df[target_column].isna().any():
        raise TrainingInputError(
            f"Target column '{target_column}' contains missing values."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if X.empty or X.shape[1] == 0:
        raise TrainingInputError("Dataset must contain at least one feature column.")

    if X.isna().any().any():
        raise TrainingInputError("Feature columns contain missing values.")

    return X, y
