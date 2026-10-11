
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_LINEAGE_DIR = Path("lineage")


def save_lineage(
    dataset_hash: str,
    config_hash: str,
    parameters: dict,
    metrics: dict,
    artifact_path: str,
    run_id: str | None = None,
    lineage_dir: str | Path = DEFAULT_LINEAGE_DIR,
) -> dict:
    lineage_dir = Path(lineage_dir)
    lineage_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = str(uuid4())

    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "parameters": parameters,
        "metrics": metrics,
        "artifact_path": artifact_path,
    }

    record_path = lineage_dir / f"{run_id}.json"

    with record_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)

    record["lineage_path"] = str(record_path)
    return record
