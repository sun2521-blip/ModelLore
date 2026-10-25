
from uuid import uuid4

from db.models import Dataset


def make_dataset(
    *,
    name: str,
    original_filename: str,
) -> Dataset:
    return Dataset(
        id=uuid4(),
        name=name,
        original_filename=original_filename,
        file_path=f"/internal/uploads/{name}.csv",
        sha256="a" * 64,
        row_count=10,
        column_count=3,
        target_column="target",
    )


def test_list_datasets_returns_metadata(
    api_client,
    db_session_factory,
):
    with db_session_factory() as db:
        db.add(
            make_dataset(
                name="first",
                original_filename="first.csv",
            )
        )
        db.commit()

    response = api_client.get("/datasets")

    assert response.status_code == 200
    body = response.json()

    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["name"] == "first"
    assert body["items"][0]["column_count"] == 3
    assert "file_path" not in body["items"][0]


def test_list_datasets_supports_pagination(
    api_client,
    db_session_factory,
):
    with db_session_factory() as db:
        db.add_all(
            [
                make_dataset(
                    name=f"dataset-{i}",
                    original_filename=f"dataset-{i}.csv",
                )
                for i in range(3)
            ]
        )
        db.commit()

    response = api_client.get("/datasets?limit=2&offset=1")

    assert response.status_code == 200
    body = response.json()

    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 1


def test_list_datasets_rejects_invalid_pagination(api_client):
    response = api_client.get("/datasets?limit=0")

    assert response.status_code == 422


def test_get_dataset_returns_metadata(
    api_client,
    db_session_factory,
):
    dataset = make_dataset(
        name="iris",
        original_filename="iris.csv",
    )

    with db_session_factory() as db:
        db.add(dataset)
        db.commit()
        dataset_id = dataset.id

    response = api_client.get(f"/datasets/{dataset_id}")

    assert response.status_code == 200
    body = response.json()

    assert body["dataset_id"] == str(dataset_id)
    assert body["original_filename"] == "iris.csv"
    assert body["sha256"] == "a" * 64
    assert body["target_column"] == "target"
    assert "file_path" not in body


def test_get_dataset_returns_404_for_unknown_id(api_client):
    response = api_client.get(f"/datasets/{uuid4()}")

    assert response.status_code == 404



def test_dataset_management_requires_api_key(monkeypatch):
    import os

    from fastapi.testclient import TestClient
    from api.main import app

    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        response = client.get("/datasets")

    assert response.status_code == 401

