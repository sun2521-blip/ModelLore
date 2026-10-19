
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    with TestClient(app) as client:
        client.headers.update({"X-API-Key": "test-api-key"})
        yield client
