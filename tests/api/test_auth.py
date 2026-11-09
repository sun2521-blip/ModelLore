
import pytest
from fastapi.testclient import TestClient

from api.main import app


RUN_ID = "0123456789abcdef0123456789abcdef"


def test_protected_endpoint_rejects_missing_api_key(monkeypatch):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        response = client.get(f"/runs/{RUN_ID}")

    assert response.status_code == 401


def test_protected_endpoint_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        response = client.get(
            f"/runs/{RUN_ID}",
            headers={"X-API-Key": "wrong-api-key"},
        )

    assert response.status_code == 401


def test_protected_endpoint_rejects_key_that_only_shares_a_prefix(monkeypatch):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        response = client.get(
            f"/runs/{RUN_ID}",
            headers={"X-API-Key": "test-api-key-but-longer"},
        )

    assert response.status_code == 401


def test_protected_endpoint_fails_closed_when_key_is_not_set(monkeypatch):
    monkeypatch.delenv("MLFLOWHUB_API_KEY", raising=False)

    with TestClient(app) as client:
        response = client.get(
            f"/runs/{RUN_ID}",
            headers={"X-API-Key": "anything"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "API authentication is not configured"}


@pytest.mark.parametrize("blank_key", ["", "   ", "\t\n"])
def test_protected_endpoint_fails_closed_when_key_is_blank(monkeypatch, blank_key):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", blank_key)

    with TestClient(app) as client:
        response = client.get(
            f"/runs/{RUN_ID}",
            headers={"X-API-Key": "anything"},
        )

    assert response.status_code == 503


def test_protected_endpoint_accepts_the_configured_key(api_client):
    response = api_client.get("/datasets")

    assert response.status_code == 200


def test_health_endpoint_does_not_require_api_key(monkeypatch):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
