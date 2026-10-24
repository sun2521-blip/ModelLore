
"""Database engine and session management."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str:
    """Return the configured database URL."""
    return os.getenv("DATABASE_URL", "sqlite:///./modellore.db")


def build_engine(database_url: str | None = None) -> Engine:
    """Build an engine; SQLite gets its required connection options."""
    url = database_url or get_database_url()
    kwargs: dict[str, object] = {"pool_pre_ping": True}

    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}

    return create_engine(url, **kwargs)


engine = build_engine()
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Yield a session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
