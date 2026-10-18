
from fastapi.testclient import TestClient

from api.main import app
from unittest.mock import patch

client = TestClient(app)


def test_train_iris_returns_training_results() -> None:
    response = client.post(
        "/train",
        json={"config_name": "iris"},
    )

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "completed"
    assert body["run_id"] is not None
    assert body["metrics"] is not None


def test_train_rejects_unknown_config_name() -> None:
    response = client.post(
        "/train",
        json={"config_name": "unknown"},
    )

    assert response.status_code == 422


def test_train_handles_training_value_error() -> None:
    with patch(
        "api.main.train_model",
        side_effect=ValueError("Invalid training configuration"),
    ):
        response = client.post(
            "/train",
            json={"config_name": "iris"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid training configuration"
