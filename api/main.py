import csv
import hashlib
import hmac
import io
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
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
    Path as PathParam,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.limits import (
    MAX_FEATURE_VALUE_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PREDICT_ROWS,
    MAX_UPLOAD_COLUMNS,
    MAX_UPLOAD_ROWS,
)
from db.models import Dataset, ModelRecord, TrainingJob
from db.session import engine, get_db
from ml.errors import TrainingInputError
from ml.inference.predict import (
    ArtifactIntegrityError,
    load_verified_model,
    predict,
)
from ml.training.train import train_model


app = FastAPI(title="ModelLore API", version="0.5.0")

# Logs name the event, the IDs involved, and the exception type. They never
# include exception messages, tracebacks, file paths, dataset values, or keys.
logger = logging.getLogger("modellore.api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(levelname)s:     %(name)s %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACT_DIR = (PROJECT_ROOT / "artifacts").resolve()
LINEAGE_DIR = PROJECT_ROOT / "lineage"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Room for multipart boundaries and part headers around the CSV itself.
UPLOAD_ENVELOPE_BYTES = 64 * 1024


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    """Refuse an upload by its declared size, before the body is read."""
    if request.method == "POST" and request.url.path == "/datasets/upload":
        declared_size = request.headers.get("content-length")

        if declared_size is not None:
            try:
                too_large = (
                    int(declared_size)
                    > MAX_UPLOAD_BYTES + UPLOAD_ENVELOPE_BYTES
                )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header"},
                )

            if too_large:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "CSV upload exceeds the 10 MiB limit"},
                )

    return await call_next(request)


def clean_filename(raw_filename: str) -> str:
    """Keep the last path segment of a client filename, minus control characters."""
    name = re.split(r"[\\/]", raw_filename)[-1]
    name = "".join(
        character
        for character in name
        if not unicodedata.category(character).startswith("C")
    )
    return name.strip()


def mark_job_failed(db: Session, job: TrainingJob, message: str) -> None:
    """Record a failed training job with a message that is safe to show."""
    job.status = "FAILED"
    job.error_message = message
    job.completed_at = datetime.now(timezone.utc)
    db.commit()


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected_key = os.getenv("MLFLOWHUB_API_KEY")

    # A blank or whitespace-only key counts as "not configured": fail closed.
    if not expected_key or not expected_key.strip():
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    if not x_api_key or not hmac.compare_digest(
        x_api_key.encode("utf-8"),
        expected_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


ColumnName = Annotated[
    str, StringConstraints(min_length=1, max_length=MAX_NAME_LENGTH)
]
FeatureValue = (
    float
    | int
    | Annotated[str, StringConstraints(max_length=MAX_FEATURE_VALUE_LENGTH)]
    | bool
    | None
)
FeatureRow = Annotated[
    dict[ColumnName, FeatureValue],
    Field(min_length=1, max_length=MAX_UPLOAD_COLUMNS),
]
MLFLOW_RUN_ID_PATTERN = r"^[0-9a-f]{32}$"


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_name: Literal["iris", "csv-test"] | None = None
    dataset_id: UUID | None = None
    target_column: str | None = Field(
        default=None, min_length=1, max_length=MAX_NAME_LENGTH
    )


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lineage_run_id: UUID
    features: list[FeatureRow] = Field(
        min_length=1, max_length=MAX_PREDICT_ROWS
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
    filename = clean_filename(file.filename or "")
    file_stem = Path(filename).stem

    if not file_stem or Path(filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=415,
            detail="Only CSV files are supported",
        )

    # Keep the display name within the database column, extension intact.
    if len(filename) > MAX_NAME_LENGTH:
        file_stem = file_stem[: MAX_NAME_LENGTH - len(".csv")]
        filename = f"{file_stem}.csv"

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

    if b"\x00" in content:
        raise HTTPException(
            status_code=400,
            detail="CSV must not contain NUL bytes",
        )

    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV must be valid UTF-8",
        ) from exc

    reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)

    try:
        header = next(reader, None)

        if (
            not header
            or len(header) != len(set(header))
            or any(not column.strip() for column in header)
        ):
            raise HTTPException(
                status_code=400,
                detail="CSV must have unique, non-empty column headers",
            )

        if len(header) > MAX_UPLOAD_COLUMNS:
            raise HTTPException(
                status_code=413,
                detail=f"CSV has more than {MAX_UPLOAD_COLUMNS} columns",
            )

        if len(header) < 2:
            raise HTTPException(
                status_code=400,
                detail="CSV needs at least two columns: features and a target",
            )

        if any(len(column) > MAX_NAME_LENGTH for column in header):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Column headers must be at most {MAX_NAME_LENGTH} characters"
                ),
            )

        row_count = 0
        for row in reader:
            if not row:
                continue

            if len(row) != len(header):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Every row must have the same number of columns "
                        "as the header"
                    ),
                )

            row_count += 1
            if row_count > MAX_UPLOAD_ROWS:
                raise HTTPException(
                    status_code=413,
                    detail=f"CSV has more than {MAX_UPLOAD_ROWS} data rows",
                )
    except csv.Error as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV could not be parsed",
        ) from exc

    if row_count == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain at least one data row",
        )

    dataset_id = uuid4()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{dataset_id}.csv"

    try:
        destination.write_bytes(content)

        dataset = Dataset(
            id=dataset_id,
            name=file_stem,
            original_filename=filename,
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
        logger.error(
            "dataset upload not saved dataset_id=%s error_type=%s",
            dataset_id,
            type(exc).__name__,
        )
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
        # Only datasets the API itself registered can be trained on.
        dataset_record = db.get(Dataset, request.dataset_id)
        if dataset_record is None:
            raise HTTPException(
                status_code=404,
                detail="Uploaded dataset not found",
            )

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
            logger.error(
                "training config not prepared dataset_id=%s error_type=%s",
                request.dataset_id,
                type(exc).__name__,
            )
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
        job_id = job.id
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
    except TrainingInputError as exc:
        # Written for the user: names their columns or config fields, nothing else.
        message = str(exc)
        mark_job_failed(db, job, message)
        logger.warning(
            "training rejected job_id=%s error_type=%s",
            job_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=400, detail=message) from exc

    except (ValueError, FileNotFoundError) as exc:
        # Library messages can include file paths or cell values, so they stay out.
        message = "The dataset or configuration could not be used for training"
        mark_job_failed(db, job, message)
        logger.warning(
            "training rejected job_id=%s error_type=%s",
            job_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=400, detail=message) from exc

    except Exception as exc:
        message = "Training failed unexpectedly"
        mark_job_failed(db, job, message)
        logger.error(
            "training failed job_id=%s error_type=%s",
            job_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail=message) from exc

    job.mlflow_run_id = result.get("mlflow_run_id")
    lineage_run_id = result.get("run_id")
    artifact_path = result.get("model_path")
    artifact_sha256 = result.get("artifact_sha256")
    feature_columns = result.get("feature_columns")

    try:
        model_lineage_run_id = UUID(str(lineage_run_id))
    except ValueError:
        model_lineage_run_id = None

    # A job only counts as FINISHED when it leaves a model that /predict can use.
    if (
        not artifact_path
        or not artifact_sha256
        or model_lineage_run_id is None
        or not feature_columns
    ):
        unusable_message = (
            "Training completed but did not produce a usable model"
        )
        mark_job_failed(db, job, unusable_message)
        logger.error("training left no usable model job_id=%s", job_id)

        raise HTTPException(
            status_code=500,
            detail=unusable_message,
        )

    job.status = "FINISHED"
    job.completed_at = datetime.now(timezone.utc)

    db.add(
        ModelRecord(
            training_job_id=job.id,
            lineage_run_id=model_lineage_run_id,
            feature_columns=feature_columns,
            name=job.config_name,
            artifact_path=str(artifact_path),
            artifact_sha256=artifact_sha256,
            metrics=result.get("metrics"),
        )
    )


    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        failure_message = (
            "Training completed but its database record could not be saved"
        )
        logger.error(
            "training result not saved job_id=%s error_type=%s",
            job_id,
            type(exc).__name__,
        )

        try:
            failed_job = db.get(TrainingJob, job_id)
            if failed_job is not None:
                failed_job.status = "FAILED"
                failed_job.error_message = failure_message
                failed_job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            db.rollback()

        raise HTTPException(
            status_code=500,
            detail=failure_message,
        ) from exc


    return {
        "status": "completed",
        "lineage_run_id": lineage_run_id,
        "mlflow_run_id": result.get("mlflow_run_id"),
        "metrics": result.get("metrics"),
        "parameters": result.get("parameters"),
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
def get_run(
    mlflow_run_id: str = PathParam(pattern=MLFLOW_RUN_ID_PATTERN),
) -> dict:
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

        logger.error(
            "mlflow run lookup failed error_type=%s", type(exc).__name__
        )
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

    if not model_record.artifact_sha256:
        raise HTTPException(
            status_code=409,
            detail="This model has no recorded artifact hash; retrain it",
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
        model = load_verified_model(
            artifact_path,
            model_record.artifact_sha256,
            ARTIFACT_DIR,
        )
        predictions = predict(model, feature_data)
    except ArtifactIntegrityError as exc:
        logger.error(
            "artifact integrity check failed model_id=%s", model_record.id
        )
        raise HTTPException(
            status_code=500,
            detail="Model artifact failed its integrity check",
        ) from exc
    except FileNotFoundError as exc:
        logger.error("model artifact missing model_id=%s", model_record.id)
        raise HTTPException(
            status_code=404,
            detail="Model artifact not found",
        ) from exc
    except (ValueError, TypeError) as exc:
        # The library message can echo the offending input value, so it stays out.
        logger.warning(
            "prediction input rejected model_id=%s error_type=%s",
            model_record.id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid prediction input",
        ) from exc
    except (OSError, EOFError) as exc:
        logger.error(
            "model artifact unreadable model_id=%s error_type=%s",
            model_record.id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to load the model artifact",
        ) from exc

    return {
        "lineage_run_id": str(request.lineage_run_id),
        "predictions": predictions.tolist(),
        "count": len(predictions),
    }
