
from uuid import uuid4

from db.models import TrainingJob


def make_job(config_name="iris", status="FINISHED"):
    return TrainingJob(
        config_name=config_name,
        status=status,
        mlflow_run_id="test-mlflow-run",
        error_message=(
            "Training failed" if status == "FAILED" else None
        ),
    )


def test_list_jobs_returns_persisted_jobs(api_client, db_session_factory):
    with db_session_factory() as db:
        db.add(make_job())
        db.add(make_job(config_name="csv-test", status="FAILED"))
        db.commit()

    response = api_client.get("/jobs")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert {item["status"] for item in body["items"]} == {
        "FINISHED",
        "FAILED",
    }


def test_list_jobs_supports_pagination(api_client, db_session_factory):
    with db_session_factory() as db:
        for i in range(3):
            db.add(make_job(config_name=f"config-{i}"))
        db.commit()

    response = api_client.get("/jobs?limit=2&offset=1")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_list_jobs_rejects_invalid_pagination(api_client):
    response = api_client.get("/jobs?limit=0")

    assert response.status_code == 422


def test_get_job_returns_details(api_client, db_session_factory):
    with db_session_factory() as db:
        job = make_job(status="FAILED")
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    response = api_client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == str(job_id)
    assert body["status"] == "FAILED"
    assert body["error_message"] == "Training failed"


def test_get_job_returns_404_for_unknown_id(api_client):
    response = api_client.get(f"/jobs/{uuid4()}")

    assert response.status_code == 404


def test_job_management_requires_api_key():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.get("/jobs")

    assert response.status_code == 401
