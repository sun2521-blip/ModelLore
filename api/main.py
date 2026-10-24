
import csv
import json
import os
import secrets
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import mlflow
import pandas as pd
import yaml
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from ml.inference.predict import load_model, predict
from ml.training.train import train_model

from sqlalchemy import text
from db.session import engine, get_db
import hashlib

from db.models import Dataset
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from db.models import Dataset, TrainingJob, ModelRecord


app = FastAPI(title="ModelLore API", version="0.2.0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACT_DIR = (PROJECT_ROOT / "artifacts").resolve()
LINEAGE_DIR = PROJECT_ROOT / "lineage"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected_key = os.getenv("MLFLOWHUB_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key.encode("utf-8"),
        expected_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


class TrainingRequest(BaseModel):
    config_name: Literal["iris", "csv-test"] = "iris"
    dataset_id: UUID | None = None
    target_column: str | None = Field(default=None, min_length=1)

class PredictionRequest(BaseModel):
    lineage_run_id: UUID
    features: list[dict[str, float | int | str | bool | None]] = Field(
        min_length=1
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}



@app.get("/health/database")
def database_health():
    """Check whether the configured application database is reachable."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"database": "connected"}
    except Exception:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="Application database unavailable",
        )



@app.post(
    "/datasets/upload",
    dependencies=[Depends(require_api_key)],
)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=415,
            detail="Only CSV files are supported",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="CSV upload exceeds the 10 MiB limit",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV is empty",
        )

    try:
        decoded = content.decode("utf-8-sig")
        header = next(csv.reader(decoded.splitlines()), None)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV must be valid UTF-8",
        ) from exc

    if (
        not header
        or len(header) != len(set(header))
        or any(not column.strip() for column in header)
    ):
        raise HTTPException(
            status_code=400,
            detail="CSV must have unique, non-empty column headers",
        )

    dataset_id = uuid4()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{dataset_id}.csv"

    try:
        destination.write_bytes(content)

        dataset = Dataset(
            id=dataset_id,
            name=Path(file.filename).stem,
            original_filename=Path(file.filename).name,
            file_path=str(destination),
            sha256=hashlib.sha256(content).hexdigest(),
            column_count=len(header),
        )
        db.add(dataset)
        db.commit()

    except Exception as exc:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to save dataset metadata",
        ) from exc

    return {
        "dataset_id": str(dataset_id),
        "filename": destination.name,
        "size_bytes": len(content),
        "sha256": dataset.sha256,
        "column_count": dataset.column_count,
    }



@app.post("/train", dependencies=[Depends(require_api_key)])
def start_training(
    request: TrainingRequest,
    db: Session = Depends(get_db),
) -> dict:
    dataset_path = None
    dataset_record = None

    if request.dataset_id:
        dataset_path = (
            UPLOAD_DIR / f"{request.dataset_id}.csv"
        ).resolve()

        if not dataset_path.is_relative_to(UPLOAD_DIR.resolve()):
            raise HTTPException(
                status_code=400,
                detail="Invalid dataset ID",
            )

        if not dataset_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Uploaded dataset not found",
            )

        dataset_record = db.get(Dataset, request.dataset_id)
        

        template_path = CONFIG_DIR / "csv-test.yaml"

        if not template_path.is_file():
            raise HTTPException(
                status_code=500,
                detail="CSV training configuration is missing",
            )

        try:
            config = yaml.safe_load(
                template_path.read_text(encoding="utf-8")
            )

            if not isinstance(config, dict):
                raise ValueError("Invalid CSV training configuration")

            dataset_config = config.get("dataset")
            if not isinstance(dataset_config, dict):
                raise ValueError(
                    "CSV training configuration needs a dataset section"
                )

            dataset_config["path"] = str(dataset_path)
            if request.target_column is not None:
                dataset_config["target_column"] = request.target_column

            generated_config = UPLOAD_DIR / f"{request.dataset_id}.yaml"
            generated_config.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            config_path = generated_config

        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise HTTPException(
                status_code=500,
                detail="Unable to prepare dataset training configuration",
            ) from exc
    else:
        config_path = CONFIG_DIR / f"{request.config_name}.yaml"

    if not config_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Training configuration not found",
        )

    job = TrainingJob(
        dataset_id=dataset_record.id if dataset_record else None,
        config_name=request.config_name,
        status="RUNNING",
        started_at=datetime.now(timezone.utc),
    )

    try:
        db.add(job)
        db.commit()
        db.refresh(job)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to create training job",
        ) from exc

    try:
        result = train_model(
            config_path=str(config_path),
            artifact_dir=ARTIFACT_DIR,
            lineage_dir=LINEAGE_DIR,
        )
    except (ValueError, FileNotFoundError) as exc:
        job.status = "FAILED"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        job.status = "FAILED"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        raise HTTPException(
            status_code=500,
            detail="Training failed unexpectedly",
        ) from exc

    job.status = "FINISHED"
    job.mlflow_run_id = result.get("mlflow_run_id")
    job.completed_at = datetime.now(timezone.utc)

    artifact_path = None
    lineage_path = result.get("lineage_path")
    lineage_run_id = result.get("run_id")

    if lineage_path:
        try:
            with Path(lineage_path).open("r", encoding="utf-8") as file:
                lineage = json.load(file)
            artifact_path = lineage.get("artifact_path")
        except (OSError, json.JSONDecodeError):
            artifact_path = None

    if artifact_path:
        db.add(
            ModelRecord(
                training_job_id=job.id,
                name=request.config_name,
                artifact_path=str(artifact_path),
                metrics=result.get("metrics"),
            )
        )

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Training completed but its database record could not be saved",
        ) from exc

    return {
        "status": "completed",
        "lineage_run_id": lineage_run_id,
        "mlflow_run_id": result.get("mlflow_run_id"),
        "metrics": result.get("metrics"),
        "parameters": result.get("parameters"),
        "lineage_path": lineage_path,
    }



@app.get(
    "/runs/{mlflow_run_id}",
    dependencies=[Depends(require_api_key)],
)
def get_run(mlflow_run_id: str) -> dict:
    try:
        run = mlflow.get_run(mlflow_run_id)
    except mlflow.MlflowException as exc:
        error_code = getattr(exc, "error_code", None)
        message = str(exc).lower()

        if (
            error_code == "RESOURCE_DOES_NOT_EXIST"
            or "run not found" in message
            or "does not exist" in message
        ):
            raise HTTPException(
                status_code=404,
                detail="MLflow run not found",
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve MLflow run",
        ) from exc

    return {
        "mlflow_run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "parameters": run.data.params,
        "metrics": run.data.metrics,
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
def make_prediction(request: PredictionRequest) -> dict:
    lineage_path = LINEAGE_DIR / f"{request.lineage_run_id}.json"

    try:
        with lineage_path.open("r", encoding="utf-8") as file:
            lineage = json.load(file)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Lineage run not found",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to read training lineage",
        ) from exc

    if lineage.get("status") != "FINISHED":
        raise HTTPException(
            status_code=409,
            detail="Cannot predict with an unsuccessful training run",
        )

    feature_columns = lineage.get("feature_columns")
    if not feature_columns:
        raise HTTPException(
            status_code=409,
            detail="This run has no saved feature-column metadata",
        )

    expected_columns = set(feature_columns)
    for row in request.features:
        if set(row.keys()) != expected_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Each feature row must contain exactly the "
                        "feature columns used during training."
                    ),
                    "expected_columns": feature_columns,
                },
            )

    artifact_value = lineage.get("artifact_path")
    if not artifact_value:
        raise HTTPException(
            status_code=404,
            detail="Model artifact not found for this training run",
        )

    artifact_path = Path(artifact_value)
    if not artifact_path.is_absolute():
        artifact_path = PROJECT_ROOT / artifact_path
    artifact_path = artifact_path.resolve()

    try:
        artifact_path.relative_to(ARTIFACT_DIR)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Training lineage references an invalid model artifact",
        ) from exc

    feature_data = pd.DataFrame(
        request.features,
        columns=feature_columns,
    )

    try:
        model = load_model(artifact_path)
        predictions = predict(model, feature_data)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Model artifact not found",
        ) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prediction input: {exc}",
        ) from exc
    except (OSError, EOFError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to load the model artifact",
        ) from exc

    return {
        "lineage_run_id": str(request.lineage_run_id),
        "predictions": predictions.tolist(),
        "count": len(predictions),
    }
