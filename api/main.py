import csv
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
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
    Query,
    UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from db.models import Dataset, ModelRecord, TrainingJob
from db.session import engine, get_db
from ml.inference.predict import load_model, predict
from ml.training.train import train_model
import io

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
    config_name: Literal["iris", "csv-test"] | None = None
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
        reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
        header = next(reader, None)
        row_count = sum(1 for row in reader if row)
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
            row_count=row_count,
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
        "row_count": dataset.row_count,
    }



@app.get(
    "/datasets",
    dependencies=[Depends(require_api_key)],
)
def list_datasets(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    """List uploaded datasets, newest first."""
    total = db.query(Dataset).count()
    datasets = (
        db.query(Dataset)
        .order_by(Dataset.created_at.desc(), Dataset.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "items": [
            {
                "dataset_id": str(dataset.id),
                "name": dataset.name,
                "original_filename": dataset.original_filename,
                "sha256": dataset.sha256,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "target_column": dataset.target_column,
                "created_at": (
                    dataset.created_at.isoformat()
                    if dataset.created_at
                    else None
                ),
            }
            for dataset in datasets
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get(
    "/datasets/{dataset_id}",
    dependencies=[Depends(require_api_key)],
)
def get_dataset(
    dataset_id: UUID,
    db: Session = Depends(get_db),
) -> dict:
    """Retrieve metadata for one dataset."""
    dataset = db.get(Dataset, dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found",
        )

    return {
        "dataset_id": str(dataset.id),
        "name": dataset.name,
        "original_filename": dataset.original_filename,
        "sha256": dataset.sha256,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "target_column": dataset.target_column,
        "created_at": (
            dataset.created_at.isoformat()
            if dataset.created_at
            else None
        ),
    }
    
    

@app.get(
    "/jobs",
    dependencies=[Depends(require_api_key)],
)
def list_training_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    """List training jobs, newest first."""
    total = db.query(TrainingJob).count()
    jobs = (
        db.query(TrainingJob)
        .order_by(
            TrainingJob.created_at.desc(),
            TrainingJob.id.desc(),
        )
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "items": [
            {
                "job_id": str(job.id),
                "dataset_id": (
                    str(job.dataset_id) if job.dataset_id else None
                ),
                "config_name": job.config_name,
                "status": job.status,
                "mlflow_run_id": job.mlflow_run_id,
                "metrics": job.model.metrics if job.model else None,
                "error_message": job.error_message,
                "created_at": (
                    job.created_at.isoformat()
                    if job.created_at else None
                ),
                "started_at": (
                    job.started_at.isoformat()
                    if job.started_at else None
                ),
                "completed_at": (
                    job.completed_at.isoformat()
                    if job.completed_at else None
                ),
            }
            for job in jobs
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get(
    "/jobs/{job_id}",
    dependencies=[Depends(require_api_key)],
)
def get_training_job(
    job_id: UUID,
    db: Session = Depends(get_db),
) -> dict:
    """Retrieve details for one training job."""
    job = db.get(TrainingJob, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Training job not found",
        )

    return {
        "job_id": str(job.id),
        "dataset_id": str(job.dataset_id) if job.dataset_id else None,
        "config_name": job.config_name,
        "status": job.status,
        "mlflow_run_id": job.mlflow_run_id,
        "metrics": job.model.metrics if job.model else None,
        "error_message": job.error_message,
        "created_at": (
            job.created_at.isoformat() if job.created_at else None
        ),
        "started_at": (
            job.started_at.isoformat() if job.started_at else None
        ),
        "completed_at": (
            job.completed_at.isoformat() if job.completed_at else None
        ),
    }




@app.post("/train", dependencies=[Depends(require_api_key)])
def start_training(
    request: TrainingRequest,
    db: Session = Depends(get_db),
) -> dict:
    dataset_path = None
    dataset_record = None
    config_name = (
        "csv-test" if request.dataset_id else (request.config_name or "iris")
    )

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

            target_column = request.target_column or dataset_config.get("target_column")
            if not isinstance(target_column, str) or not target_column.strip():
                raise HTTPException(
                    status_code=400,
                    detail="A target column must be specified",
                )

            dataset_config["target_column"] = target_column
            if dataset_record is not None:
                dataset_record.target_column = target_column

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

        config_path = CONFIG_DIR / f"{config_name}.yaml"

    if not config_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Training configuration not found",
        )

    job = TrainingJob(
        dataset_id=dataset_record.id if dataset_record else None,
        config_name=config_name,
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
                lineage_run_id=UUID(str(lineage_run_id)),
                feature_columns=result.get("feature_columns"),
                name=job.config_name,
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
    "/models",
    dependencies=[Depends(require_api_key)],
)
def list_models(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> dict:
    """List registered models, newest first."""
    total = db.query(ModelRecord).count()

    models = (
        db.query(ModelRecord)
        .order_by(
            ModelRecord.created_at.desc(),
            ModelRecord.id.desc(),
        )
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "items": [
            {
                "model_id": str(model.id),
                "training_job_id": str(model.training_job_id),
                "name": model.name,
                "version": model.version,
                "metrics": model.metrics,
                "created_at": (
                    model.created_at.isoformat()
                    if model.created_at
                    else None
                ),
            }
            for model in models
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get(
    "/models/{model_id}",
    dependencies=[Depends(require_api_key)],
)
def get_model(
    model_id: UUID,
    db: Session = Depends(get_db),
) -> dict:
    """Retrieve metadata for one registered model."""
    model = db.get(ModelRecord, model_id)

    if model is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found",
        )

    job = model.training_job

    return {
        "model_id": str(model.id),
        "training_job_id": str(model.training_job_id),
        "name": model.name,
        "version": model.version,
        "metrics": model.metrics,
        "created_at": (
            model.created_at.isoformat()
            if model.created_at
            else None
        ),
        "training_job": {
            "status": job.status,
            "config_name": job.config_name,
            "mlflow_run_id": job.mlflow_run_id,
        },
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
def make_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db),
) -> dict:
    model_record = (
        db.query(ModelRecord)
        .filter(ModelRecord.lineage_run_id == request.lineage_run_id)
        .first()
    )

    if model_record is None:
        raise HTTPException(
            status_code=404,
            detail="Registered model not found for this lineage run",
        )

    job = model_record.training_job
    if job.status != "FINISHED":
        raise HTTPException(
            status_code=409,
            detail="Cannot predict with an unsuccessful training run",
        )

    feature_columns = model_record.feature_columns
    if not feature_columns:
        raise HTTPException(
            status_code=409,
            detail="This model has no saved feature-column metadata; retrain it",
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

    artifact_path = Path(model_record.artifact_path)
    if not artifact_path.is_absolute():
        artifact_path = PROJECT_ROOT / artifact_path
    artifact_path = artifact_path.resolve()

    try:
        artifact_path.relative_to(ARTIFACT_DIR)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Registered model references an invalid artifact path",
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
