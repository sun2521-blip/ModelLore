
"""Persistent database models for ModelLore."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class Dataset(Base):
    """A dataset uploaded or registered with ModelLore."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_column: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    training_jobs: Mapped[list["TrainingJob"]] = relationship(
        back_populates="dataset"
    )


class TrainingJob(Base):
    """A training run and its lifecycle state."""

    __tablename__ = "training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    config_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING", index=True
    )
    mlflow_run_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    dataset: Mapped[Dataset | None] = relationship(
        back_populates="training_jobs"
    )
    model: Mapped["ModelRecord | None"] = relationship(
        back_populates="training_job", uselist=False
    )


class ModelRecord(Base):
    """Metadata for a trained model artifact."""

    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("training_job_id", name="uq_models_training_job"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    training_job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("training_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    training_job: Mapped[TrainingJob] = relationship(
        back_populates="model"
    )
    lineage_run_id: Mapped[uuid.UUID | None] = mapped_column(
    Uuid, nullable=True, unique=True, index=True
    )
    feature_columns: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True
    )
    artifact_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
