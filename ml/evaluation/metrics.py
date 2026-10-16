from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def calculate_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": mse ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }
