"""Alembic migrations apply, roll back, and re-apply cleanly."""

from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def alembic_config():
    # Built in code rather than from alembic.ini so the ini's logging setup
    # does not reconfigure the loggers other tests rely on.
    config = Config()
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    return config


def model_columns(database_url):
    engine = sa.create_engine(database_url)
    try:
        return {column["name"] for column in sa.inspect(engine).get_columns("models")}
    finally:
        engine.dispose()


def test_artifact_hash_migration_round_trip(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'migrations.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = alembic_config()

    command.upgrade(config, "head")
    assert "artifact_sha256" in model_columns(database_url)

    command.downgrade(config, "-1")
    columns = model_columns(database_url)
    assert "artifact_sha256" not in columns
    assert {"lineage_run_id", "feature_columns"} <= columns

    command.upgrade(config, "head")
    assert "artifact_sha256" in model_columns(database_url)
