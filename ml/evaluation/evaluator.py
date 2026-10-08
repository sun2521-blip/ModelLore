from ml.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)


def evaluate_model(task_type, y_true, y_pred):
    if task_type == "classification":
        return calculate_classification_metrics(
            y_true,
            y_pred,
        )

    if task_type == "regression":
        return calculate_regression_metrics(
            y_true,
            y_pred,
        )

    raise ValueError(
        f"Unsupported task type: {task_type}"
    )