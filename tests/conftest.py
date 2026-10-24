import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from db import models  # noqa: F401  (registers the tables on Base)
from db.base import Base
from db.session import get_db


@pytest.fixture
def db_session_factory():
    """A throwaway in-memory database with all tables created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )

    yield factory

    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def api_client(monkeypatch, db_session_factory):
    monkeypatch.setenv("MLFLOWHUB_API_KEY", "test-api-key")

    def override_get_db():
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            client.headers.update({"X-API-Key": "test-api-key"})
            yield client
    finally:
        app.dependency_overrides.clear()