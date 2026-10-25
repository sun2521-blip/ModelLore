
"""Shared fixtures for API and database tests."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from db.base import Base
from db import models  # Register all ORM models before creating tables.
from db.session import get_db


@pytest.fixture
def api_client(monkeypatch):
    """Provide an API client backed by an isolated in-memory database."""
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestSessionLocal = sessionmaker(
        bind=test_engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )

    # Create all registered tables in the test database.
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSessionLocal()
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
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()
