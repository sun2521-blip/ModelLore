
"""Shared fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from api.rate_limit import limiter
from db import models  # noqa: F401
from db.base import Base
from db.session import get_db


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Every test client shares one address, so start each test with fresh counts."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def db_session_factory():
    """Provide sessions for an isolated in-memory test database."""
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    Base.metadata.create_all(bind=test_engine)

    try:
        yield testing_session_local
    finally:
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()


@pytest.fixture
def api_client(monkeypatch, db_session_factory):
    """Create an API client using the isolated test database."""
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    def override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            client.headers.update({"X-API-Key": "test-api-key"})
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)
