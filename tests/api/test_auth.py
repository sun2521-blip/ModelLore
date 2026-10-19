
from fastapi.testclient import TestClient

from api.main import app


def test_protected_endpoint_rejects_missing_api_key(monkeypatch):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        response = client.get("/runs/test-run-123")

    assert response.status_code == 401


def test_protected_endpoint_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        response = client.get(
            "/runs/test-run-123",
            headers={"X-API-Key": "wrong-api-key"},
        )

    assert response.status_code == 401


def test_health_endpoint_does_not_require_api_key(monkeypatch):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
