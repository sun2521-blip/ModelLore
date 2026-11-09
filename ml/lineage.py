
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import subprocess


DEFAULT_LINEAGE_DIR = Path("lineage")


def get_git_metadata() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        return {
            "git_commit": commit,
            "git_dirty": bool(status),
        }

    # Git metadata is best-effort: a missing git (Docker image) or a refused
    # process launch (OSError covers FileNotFoundError and PermissionError)
    # must never stop lineage from being saved.
    except (subprocess.CalledProcessError, OSError):
        return {
            "git_commit": None,
            "git_dirty": None,
        }


def save_lineage(
    dataset_hash: str,
    config_hash: str,
    parameters: dict,
    metrics: dict,
    artifact_path: str,
    run_id: str | None = None,
    lineage_dir: str | Path = DEFAULT_LINEAGE_DIR,
    status: str = "FINISHED",
    error: str | None = None,
    mlflow_artifact_uri: str | None = None,
    config_path: str | None = None,
    config_snapshot: dict | None = None,
    feature_columns: list[str] | None = None,
    artifact_sha256: str | None = None,
) -> dict:
    lineage_dir = Path(lineage_dir)
    lineage_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = str(uuid4())

    git_metadata = get_git_metadata()

    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "parameters": parameters,
        "metrics": metrics,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "git_commit": git_metadata["git_commit"],
        "git_dirty": git_metadata["git_dirty"],
        "status": status,
        "error": error,
        "mlflow_artifact_uri": mlflow_artifact_uri,
        "config_path": config_path,
        "config_snapshot": config_snapshot,
        "feature_columns": feature_columns,
    }

    record_path = lineage_dir / f"{run_id}.json"

    with record_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)

    record["lineage_path"] = str(record_path)
    return record


def update_lineage_status(
    record_path: str | Path,
    status: str,
    error: str | None = None,
) -> dict:
    record_path = Path(record_path)

    with record_path.open("r", encoding="utf-8") as file:
        record = json.load(file)

    record["status"] = status
    record["error"] = error

    with record_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)

    return record
