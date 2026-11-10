"""The public view of a lineage record.

A lineage JSON file is written by the training pipeline and also holds things
that must stay on the server, such as the configuration path, the artifact
path, and MLflow's artifact location. This module builds the view the API may
return: fingerprints, settings, metrics, and links, and nothing path-shaped.
"""

import re
from typing import Any

from db.models import Dataset, ModelRecord, TrainingJob


# Lineage records are small; anything larger is not one of ours.
MAX_LINEAGE_BYTES = 1_000_000

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_MLFLOW_RUN_IN_URI = re.compile(r"(?:^|/)([0-9a-f]{32})(?:/|$)")


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value: Any, max_length: int = 255) -> str | None:
    return value if isinstance(value, str) and len(value) <= max_length else None


def _fingerprint(value: Any) -> str | None:
    return value if isinstance(value, str) and _HEX_64.fullmatch(value) else None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _numbers(value: Any) -> dict[str, int | float] | None:
    if not isinstance(value, dict):
        return None
    numbers = {
        key: number
        for key, item in value.items()
        if isinstance(key, str) and (number := _number(item)) is not None
    }
    return numbers


def _settings(value: Any) -> dict[str, int | float | str | None] | None:
    """Model parameters: names mapped to numbers, short strings, or null."""
    if not isinstance(value, dict):
        return None
    settings: dict[str, int | float | str | None] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if item is None or _number(item) is not None:
            settings[key] = item
        elif (text := _text(item, 64)) is not None:
            settings[key] = text
    return settings


def _columns(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [column for column in value if isinstance(column, str)]


def _environment(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None
    return {
        "python_version": _text(value.get("python_version"), 32),
        "scikit_learn_version": _text(value.get("scikit_learn_version"), 32),
        "lock_file_sha256": _fingerprint(value.get("lock_file_sha256")),
    }


def mlflow_run_id_from_uri(uri: Any) -> str | None:
    """Pull the 32-character MLflow run ID out of an artifact URI, and nothing else."""
    if not isinstance(uri, str):
        return None
    match = _MLFLOW_RUN_IN_URI.search(uri.replace("\\", "/"))
    return match.group(1) if match else None


def public_lineage(
    record: dict,
    *,
    model: ModelRecord | None = None,
    job: TrainingJob | None = None,
    dataset: Dataset | None = None,
) -> dict:
    """Build the API's view of one lineage record, linked to its database rows."""
    snapshot = _mapping(record.get("config_snapshot"))
    dataset_settings = _mapping(snapshot.get("dataset"))
    training = _mapping(snapshot.get("training"))
    model_settings = _mapping(snapshot.get("model"))

    git_commit = record.get("git_commit")
    mlflow_run_id = (job.mlflow_run_id if job else None) or mlflow_run_id_from_uri(
        record.get("mlflow_artifact_uri")
    )

    return {
        "lineage_run_id": _text(record.get("run_id"), 36),
        "status": _text(record.get("status"), 32),
        "created_at": _text(record.get("created_at"), 64),
        "dataset": {
            "fingerprint": _fingerprint(record.get("dataset_hash")),
            "source": _text(dataset_settings.get("name"), 32),
            "target_column": _text(dataset_settings.get("target_column")),
            "dataset_id": str(dataset.id) if dataset else None,
            "original_filename": dataset.original_filename if dataset else None,
            "file_sha256": dataset.sha256 if dataset else None,
            "row_count": dataset.row_count if dataset else None,
            "column_count": dataset.column_count if dataset else None,
        },
        "config": {
            "fingerprint": _fingerprint(record.get("config_hash")),
            "random_seed": _number(training.get("random_seed")),
            "test_size": _number(training.get("test_size")),
            "model_type": _text(model_settings.get("model_type"), 64),
            "task_type": _text(model_settings.get("task_type"), 64),
            "parameters": _settings(record.get("parameters"))
            or _settings(model_settings.get("parameters")),
        },
        "run": {
            "mlflow_run_id": mlflow_run_id,
            "job_id": str(job.id) if job else None,
            "metrics": _numbers(record.get("metrics")),
            "feature_columns": _columns(record.get("feature_columns")),
            "git_commit": git_commit
            if isinstance(git_commit, str) and _HEX_40.fullmatch(git_commit)
            else None,
            "git_dirty": record.get("git_dirty")
            if isinstance(record.get("git_dirty"), bool)
            else None,
        },
        "model": {
            "model_id": str(model.id) if model else None,
            "artifact_sha256": _fingerprint(record.get("artifact_sha256"))
            or (model.artifact_sha256 if model else None),
            "environment": _environment(record.get("environment")),
        },
    }
