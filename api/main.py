
import json
from pathlib import Path
from typing import Literal
from uuid import UUID

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.inference.predict import load_model, predict
from ml.training.train import train_model


app = FastAPI(
    title="MLFlowHub API",
    version="0.2.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACT_DIR = (PROJECT_ROOT / "artifacts").resolve()
LINEAGE_DIR = PROJECT_ROOT / "lineage"


class TrainingRequest(BaseModel):
    config_name: Literal["iris", "csv-test"] = "iris"


class PredictionRequest(BaseModel):
    run_id: UUID
    features: list[dict[str, float | int | str | bool | None]] = Field(
        min_length=1
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/train")
def start_training(request: TrainingRequest) -> dict:
    config_path = CONFIG_DIR / f"{request.config_name}.yaml"

    if not config_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Training configuration not found",
        )

    try:
        result = train_model(config_path=str(config_path))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "completed",
        "run_id": result.get("run_id"),
        "mlflow_run_id": result.get("mlflow_run_id"),
        "metrics": result.get("metrics"),
        "parameters": result.get("parameters"),
        "lineage_path": result.get("lineage_path"),
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        run = mlflow.get_run(run_id)
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
                detail="Training run not found",
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve training run",
        ) from exc

    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "parameters": run.data.params,
        "metrics": run.data.metrics,
    }


@app.post("/predict")
def make_prediction(request: PredictionRequest) -> dict:
    lineage_path = LINEAGE_DIR / f"{request.run_id}.json"

    try:
        with lineage_path.open("r", encoding="utf-8") as file:
            lineage = json.load(file)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Training run not found",
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
            detail=(
                "This run has no saved feature-column metadata. "
                "Train a new model before using prediction."
            ),
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
        "run_id": str(request.run_id),
        "predictions": predictions.tolist(),
        "count": len(predictions),
    }
