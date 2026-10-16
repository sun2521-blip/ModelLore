
import pandas as pd
import pytest

from ml.data.loader import load_dataset


def test_load_valid_dataset(tmp_path):
    file_path = tmp_path / "sample.csv"
    pd.DataFrame({
        "feature": [1, 2, 3],
        "target": ["A", "B", "A"],
    }).to_csv(file_path, index=False)

    X, y = load_dataset(file_path, "target")

    assert list(X.columns) == ["feature"]
    assert list(y) == ["A", "B", "A"]
    assert X.shape == (3, 1)


def test_reject_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv", "target")


def test_reject_missing_target_column(tmp_path):
    file_path = tmp_path / "sample.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Target column"):
        load_dataset(file_path, "target")


def test_reject_missing_values(tmp_path):
    file_path = tmp_path / "sample.csv"
    pd.DataFrame({
        "feature": [1, None],
        "target": ["A", "B"],
    }).to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        load_dataset(file_path, "target")


def test_reject_empty_dataset(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("feature,target\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_dataset(file_path, "target")


def test_reject_duplicate_columns(tmp_path):
    file_path = tmp_path / "duplicate.csv"
    file_path.write_text(
        "feature,feature,target\n1,2,A\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(file_path, "target")


def test_reject_non_csv_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("feature,target\n1,A\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Only CSV"):
        load_dataset(file_path, "target")


def test_reject_missing_target_values(tmp_path):
    file_path = tmp_path / "missing_target.csv"
    pd.DataFrame({
        "feature": [1, 2],
        "target": ["A", None],
    }).to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Target column.*missing values"):
        load_dataset(file_path, "target")


def test_reject_dataset_without_features(tmp_path):
    file_path = tmp_path / "no_features.csv"
    pd.DataFrame({"target": ["A", "B"]}).to_csv(
        file_path, index=False
    )

    with pytest.raises(ValueError, match="at least one feature"):
        load_dataset(file_path, "target")
