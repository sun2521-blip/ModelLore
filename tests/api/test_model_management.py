
from uuid import uuid4

from db.models import ModelRecord, TrainingJob


def create_model(db, name="iris-model", metrics=None):
    job = TrainingJob(
        config_name="iris",
        status="FINISHED",
        mlflow_run_id="test-run-id",
    )
    db.add(job)
    db.flush()

    model = ModelRecord(
        training_job_id=job.id,
        name=name,
        version="1",
        artifact_path="/internal/artifacts/model.joblib",
        metrics=metrics or {"accuracy": 0.95},
    )
    db.add(model)
    db.flush()

    return model


def test_list_models_returns_metadata(api_client, db_session_factory):
    with db_session_factory() as db:
        create_model(db)
        db.commit()

    response = api_client.get("/models")

    assert response.status_code == 200
    body = response.json()

    assert body["total"] == 1
    assert len(body["items"]) == 1

    model = body["items"][0]
    assert model["name"] == "iris-model"
    assert model["version"] == "1"
    assert model["metrics"]["accuracy"] == 0.95
    assert "artifact_path" not in model


def test_list_models_supports_pagination(
    api_client,
    db_session_factory,
):
    with db_session_factory() as db:
        for index in range(3):
            create_model(db, name=f"model-{index}")
        db.commit()

    response = api_client.get("/models?limit=2&offset=1")

    assert response.status_code == 200
    body = response.json()

    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_list_models_rejects_invalid_pagination(api_client):
    response = api_client.get("/models?limit=0")

    assert response.status_code == 422


def test_get_model_returns_details(api_client, db_session_factory):
    with db_session_factory() as db:
        model = create_model(db, metrics={"accuracy": 0.98})
        db.commit()
        model_id = model.id

    response = api_client.get(f"/models/{model_id}")

    assert response.status_code == 200
    body = response.json()

    assert body["model_id"] == str(model_id)
    assert body["name"] == "iris-model"
    assert body["metrics"]["accuracy"] == 0.98
    assert body["training_job"]["status"] == "FINISHED"
    assert body["training_job"]["config_name"] == "iris"
    assert "artifact_path" not in body


def test_get_model_returns_404_for_unknown_id(api_client):
    response = api_client.get(f"/models/{uuid4()}")

    assert response.status_code == 404


def test_model_management_requires_api_key():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.get("/models")

    assert response.status_code == 401
