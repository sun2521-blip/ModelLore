
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.training.train import train_model

app = FastAPI(
    title="MLFlowHub API",
    version="0.2.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


class TrainingRequest(BaseModel):
    config_name: Literal["iris", "csv-test"] = "iris"


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
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "status": "completed",
        "run_id": result.get("run_id"),
        "mlflow_run_id": result.get("mlflow_run_id"),
        "metrics": result.get("metrics"),
        "parameters": result.get("parameters"),
        "lineage_path": result.get("lineage_path"),
    }
