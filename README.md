# ModelLore

### Know how your models came to be.

Imagine a world where training a machine learning model is the easy part. The real mystery begins afterwards.

Which dataset did it learn from? What configuration shaped its decisions? Can we reproduce its results? And when a model suddenly starts behaving like it has a mind of its own, can we trace its history?

That is where **ModelLore** comes in.

ModelLore is my attempt at building a mini machine learning platform that takes a model from raw data to a trained, evaluated, saved, and traceable artifact. Instead of treating model training as a script that runs once and disappears into the void, I want every experiment to leave behind a trail that we can actually investigate.

Think of it as a small mission control for machine learning experiments.

And this is just the beginning.

---

## The mission

ModelLore is being built around one idea: **a model should never be separated from the story of how it was created.**

The goal is to build a platform that can eventually handle the journey from training a model to tracking its versions, serving predictions through an API, and monitoring its behavior.

For now, the focus is on getting the training engine and API foundations right.

### What it can do right now

- **Configuration-driven training:** Define dataset settings, model parameters, random seeds, and train-test split settings in YAML.

- **Dataset loading and validation:** Load the built-in Iris dataset or your own CSV dataset, with checks for common data problems.

- **Preprocessing pipelines:** Handle numeric and categorical features using scikit-learn transformers, including imputation and one-hot encoding.

- **Model training and evaluation:** Train a scikit-learn pipeline and evaluate its predictions using classification metrics.

- **Reproducibility:** Use controlled random seeds and SHA-256 fingerprints for datasets and configurations.

- **Experiment tracking:** Use MLflow to record training parameters, metrics, and model artifacts.

- **Model lineage:** Store JSON records connecting a training run to its dataset fingerprint, configuration fingerprint, metrics, and saved model artifact.

- **Inference:** Load a saved model and use it to make predictions on new input.

- **FastAPI backend:** Expose endpoints for health checks, CSV uploads, training, MLflow run retrieval, and predictions.

- **API-key authentication:** Protect API endpoints using an environment-configured API key.

- **Automated testing:** Test important parts of the project instead of relying on the classic "it worked on my machine" philosophy.

The current version focuses on tabular classification. Regression support, additional model families, and expanded deployment infrastructure are future work rather than features this release claims to provide.

---

## Inside the machine

The current architecture is intentionally small. Each component has a specific responsibility, so the project can grow without turning into one enormous Python file.

```text

                 CONFIGURATION

                  YAML files

                      |

                      v

                 DATA LOADER

             Iris dataset or CSV

                      |

                      v

             VALIDATION AND DATA

                PREPROCESSING

                      |

                      v

              TRAIN / TEST SPLIT

                      |

                      v

               MODEL TRAINING

             scikit-learn pipeline

                      |

                      v

               MODEL EVALUATION

            Accuracy, precision, recall, F1

                      |

             +--------+--------+

             |                 |

             v                 v

        MODEL ARTIFACT    EXPERIMENT TRACKING

           .joblib             MLflow

             |                 |

             v                 v

          INFERENCE         PARAMETERS

       AND PREDICTIONS   METRICS AND ARTIFACTS

             |

             v

         JSON LINEAGE

    Dataset -> Config -> Run -> Model

             FASTAPI SERVICE

      Health / Upload / Train / Runs / Predict

```

The JSON lineage records and MLflow tracking serve different purposes. Lineage gives each training run a traceable record, while MLflow makes experiments easier to inspect and compare.

The bigger plan is to connect these pieces to a proper backend and eventually give them a frontend of their own.

---

## The technology aboard this ship

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| pandas and NumPy | Data handling |
| scikit-learn | Preprocessing, training, evaluation and inference |
| PyYAML | Configuration loading |
| joblib | Saving and loading trained models |
| MLflow | Experiment tracking and model artifact logging |
| FastAPI | API backend |
| Pydantic | Request validation |
| pytest | Automated testing |
| uv | Python environment and dependency management |
| Git and GitHub Actions | Version control and CI |

---

## Getting started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker Desktop (for the containerized setup)

### 1. Clone the repository

```bash
git clone https://github.com/sun2521-blip/ModelLore.git
cd ModelLore
```

### 2. Install dependencies

Install the project dependencies using the committed lockfile:

```bash
uv sync
```

This creates the project's virtual environment and installs the locked dependencies.

### 3. Set your API key

The API reads its key from the `MLFLOWHUB_API_KEY` environment variable. Until it is set, protected endpoints refuse every request with `503` instead of running unauthenticated.

**PowerShell:**

```powershell
$env:MLFLOWHUB_API_KEY = "your-api-key"
```

**Git Bash:**

```bash
export MLFLOWHUB_API_KEY="your-api-key"
```

Use a long, random value and keep it private. Never commit it or your `.env` file.

Prefer a file? Copy `.env.example` to `.env` and fill in the values. Docker Compose reads `.env` on its own; for a local run, hand it to uvicorn with `--env-file .env` (step 5).

**Why "MLFLOWHUB"?** ModelLore started out as MLFlowHub; the v0.1.0 release still carries that name. The key kept its original variable name so existing setups keep working, and it is staying that way: `MLFLOWHUB_API_KEY` is the only name ModelLore reads.

### 4. Create the database tables

ModelLore keeps datasets, training jobs, and models in a database. Without a `DATABASE_URL`, that is a local SQLite file called `modellore.db`. Create its tables once, and again whenever you pull new migrations:

```bash
uv run alembic upgrade head
```

### 5. Run the API

```bash
uv run uvicorn api.main:app --reload
```

Or, with your key in `.env`:

```bash
uv run uvicorn api.main:app --reload --env-file .env
```

Open the interactive API documentation at http://127.0.0.1:8000/docs.

### 6. Run tests

```bash
uv run pytest
```

### 7. Train your first model from the command line

The default configuration uses the Iris dataset.

```bash
uv run python -m ml.training.train
```

The training pipeline loads the configuration, prepares the data, trains and evaluates the model, saves a unique model artifact, and creates a JSON lineage record. It also logs the training run to MLflow.

The terminal prints the evaluation metrics, saved model path, lineage record path, and MLflow run ID.

### Dependency management

- `pyproject.toml` lists the direct dependencies: what the app needs to run under `[project]`, and the test and lint tools (pytest, httpx2, flake8) in the `dev` group.
- `uv.lock` pins every package, direct and indirect. It is the single source of truth.
- `requirements.txt` is generated from `uv.lock` for pip and Docker. It holds the runtime packages only, each with a hash. Don't edit it by hand.

`uv sync` installs everything, including the dev group. `uv sync --no-dev` installs only what the app needs to run.

After changing dependencies in `pyproject.toml`, update the lock and regenerate `requirements.txt`:

```bash
uv lock
uv export --frozen --no-dev --no-emit-project --format requirements-txt -o requirements.txt
```

CI fails if `uv.lock` is out of date or if `requirements.txt` no longer matches it.

To install the runtime packages with pip instead of uv:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python -m pip install -r requirements.txt`. This skips the test tools, so use `uv sync` if you want to run the tests.

## Verify a training run

ModelLore can verify a completed training run by loading its lineage record, rerunning training with the saved configuration, and comparing the results.

### Run verification

First, train a model:

```bash
uv run python -m ml.training.train
```

The command prints the path to the generated lineage record. Pass that path to the verification command:

```bash
uv run python -m ml.verification lineage/YOUR_RUN_ID.json
```

Replace `YOUR_RUN_ID.json` with the actual filename generated during training.

### Example output

```json
{
  "original_run_id": "original-run-id",
  "verification_run_id": "verification-run-id",
  "dataset_matches": true,
  "config_matches": true,
  "metrics_match": true,
  "metric_comparison": {
    "accuracy": true,
    "precision": true,
    "recall": true,
    "f1": true
  },
  "verified": true
}
```

The IDs and values above are illustrative.

### What is checked?

- **Dataset fingerprint:** Checks whether the dataset used for the rerun matches the original dataset.

- **Configuration fingerprint:** Checks whether the saved configuration matches the original configuration.

- **Metrics:** Compares the rerun's recorded metrics with the original metrics within the configured numerical tolerance.

- **Overall result:** `verified` is `true` only when the dataset, configuration, and compared metrics all match.

### Limitations

Verification requires a saved configuration snapshot, an accessible dataset, and a compatible execution environment. Older lineage records without a configuration snapshot cannot be verified by rerunning.

Matching metrics do not guarantee byte-for-byte identical model artifacts. Differences in dependency versions, hardware, or execution behavior may affect reproducibility.

---

## Enter the experiment room

Start the MLflow UI in a second terminal from the project directory:

```bash
uv run mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Find the `modellore-training` experiment to inspect your runs, parameters, metrics, and artifacts.

Keep the terminal running while using the UI. The default local tracking database and generated artifacts are development data and should not be committed to Git.

---

## Bring your own dataset

You don't have to stay in the Iris universe forever.

ModelLore also supports CSV datasets through YAML configuration. For example, a dataset configuration can look like this:

```yaml
dataset:
  name: csv
  path: data/sample.csv
  target_column: target
training:
  random_seed: 42
  test_size: 0.2
model:
  model_type: random_forest
  task_type: classification
  parameters:
    n_estimators: 50
    max_depth: 5
```

Save the configuration as `configs/my_dataset.yaml`, then run:

```bash
uv run python -m ml.training.train configs/my_dataset.yaml
```

Replace the dataset path and target column with the values appropriate for your data. The current training pipeline is designed for classification datasets, and the target must contain enough examples of each class for a stratified train-test split. Every row needs a value in every column: training rejects datasets with missing feature or target values.

---

## API usage

ModelLore includes a FastAPI service for health checks, CSV dataset uploads, training, MLflow run retrieval, and predictions.

Interactive API documentation is available at `/docs` while the service is running.

### 1. Configure an API key

Set `MLFLOWHUB_API_KEY` as described in [Getting started](#3-set-your-api-key), including why it still carries the project's old name. If it is not set, protected endpoints fail closed rather than accepting unauthenticated requests.

### 2. Start the API

From the project root, create the database tables if you haven't yet, then start the server:

```bash
uv run alembic upgrade head
uv run uvicorn api.main:app --reload
```

The local service runs at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 3. Check service health

The health endpoint is public and does not require an API key:

```bash
curl http://127.0.0.1:8000/health
```

### 4. Upload a CSV dataset

Protected endpoints require the `X-API-Key` header:

```bash
curl -X POST "http://127.0.0.1:8000/datasets/upload" \
  -H "X-API-Key: $MLFLOWHUB_API_KEY" \
  -F "file=@data/sample.csv"
```

The response includes a dataset ID. Save it for the training request.

Current upload restrictions:

- CSV files only.

- Maximum file size of 10 MiB.

- Invalid UTF-8, empty files, and invalid or duplicate headers are rejected.

### 5. Train on the uploaded dataset

Replace `DATASET_UUID` with the ID returned by the upload endpoint. Set `target_column` to the actual target header in your CSV.

```bash
curl -X POST "http://127.0.0.1:8000/train" \
  -H "X-API-Key: $MLFLOWHUB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "csv-test",
    "dataset_id": "DATASET_UUID",
    "target_column": "target"
  }'
```

For the built-in Iris dataset, training can be requested without uploading a CSV:

```bash
curl -X POST "http://127.0.0.1:8000/train" \
  -H "X-API-Key: $MLFLOWHUB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"config_name": "iris"}'
```

The training response includes identifiers and information about the generated run and artifacts.

### 6. Retrieve an MLflow run

Replace `MLFLOW_RUN_ID` with the MLflow run ID returned by the training response:

```bash
curl "http://127.0.0.1:8000/runs/MLFLOW_RUN_ID" \
  -H "X-API-Key: $MLFLOWHUB_API_KEY"
```

### 7. Make predictions

Send the `lineage_run_id` from the training response and a list of feature rows. Feature names must match those the model was trained on.

The following example uses the Iris feature names:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "X-API-Key: $MLFLOWHUB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "lineage_run_id": "LINEAGE_RUN_ID",
    "features": [
      {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
      }
    ]
  }'
```

Replace the lineage ID and feature names/values with those for your trained model.

For the exact request schemas and response shapes, visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Models trained before v0.4.1

`/predict` finds a model by its `lineage_run_id` and checks each row against the model's saved `feature_columns`. Models registered before v0.4.1 were saved without these fields, so the newer prediction flow can't use them:

- No `lineage_run_id`: `/predict` can't find the model and returns `404`.
- No `feature_columns`: `/predict` returns `409` and asks you to retrain.

Retrain affected models with `POST /train`; the new model is saved with both fields. Separately, lineage records written before configuration snapshots were added can't be [verified by rerunning](#verify-a-training-run).

### API behavior and current limits

- `GET /health` is public; protected endpoints require `X-API-Key`.

- If `MLFLOWHUB_API_KEY` is unset, protected requests fail closed.

- CSV uploads accept `.csv` files up to 10 MiB and validate encoding and headers.

- `target_column` is optional. When omitted for uploaded CSV training, the configured target column is used.

- Uploaded datasets are referenced by IDs rather than arbitrary client-supplied filesystem paths.

- This is an early-stage local development API, not a hardened public production service. Do not expose it directly to the public internet without additional security, rate limiting, and operational controls.

---

## Dataset, Job, and Model Management API

ModelLore provides read-only management endpoints for inspecting uploaded datasets, training jobs, and registered models.

All management endpoints require API-key authentication using the `X-API-Key` header.

### Authentication

Set `MLFLOWHUB_API_KEY` as described in [Getting started](#3-set-your-api-key), then include the key with each request:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/datasets"
```

### Pagination

List endpoints support pagination through query parameters:

| Parameter | Default | Allowed values |
|---|---:|---|
| `limit` | 20 | 1–100 |
| `offset` | 0 | 0 or greater |

For example, retrieve up to 10 datasets, starting from the 20th result:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/datasets?limit=10&offset=20"
```

Paginated responses include `items`, `total`, `limit`, and `offset`.

### Dataset Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/datasets` | List uploaded datasets |
| GET | `/datasets/{dataset_id}` | Retrieve details for a specific dataset |

List datasets:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/datasets?limit=20&offset=0"
```

Each dataset entry includes its identifier, name, original filename, SHA-256 hash, row count, column count, target column, and creation timestamp.

Retrieve a specific dataset:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/datasets/YOUR_DATASET_ID"
```

### Training Job Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/jobs` | List training jobs |
| GET | `/jobs/{job_id}` | Retrieve details for a specific training job |

List training jobs:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/jobs?limit=20&offset=0"
```

Job records expose available metadata such as job status, metrics, errors, and timestamps.

Retrieve a specific job:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/jobs/YOUR_JOB_ID"
```

### Model Registry Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/models` | List registered models |
| GET | `/models/{model_id}` | Retrieve details for a specific model |

List registered models:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/models?limit=20&offset=0"
```

Retrieve a specific model:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/models/YOUR_MODEL_ID"
```

Model records include available model metadata and related training-job information. Artifact filesystem paths are not exposed by these management responses.

### Error Responses

- `401 Unauthorized`: API-key authentication failed.

- `404 Not Found`: The requested dataset, job, or model does not exist.

- `422 Unprocessable Entity`: Request parameters failed validation.

- `503 Service Unavailable`: The API key is not configured on the server.

These endpoints provide visibility into existing resources; they do not replace the dataset upload, training, or prediction endpoints.

## The lineage experiment

A model's metrics tell us how it performed. They don't tell us the whole story behind that performance.

ModelLore creates SHA-256 fingerprints for the loaded dataset and configuration. It records those fingerprints alongside the run ID, model parameters, evaluation metrics, and artifact information.

That gives us a starting point for answering questions like:

- Did two experiments use the same dataset and configuration?

- Which parameters were used to train a particular model?

- Which metrics belonged to a specific run?

- Where is the model artifact associated with that run?

These fingerprints are identifiers, not proof that a model is correct or secure. Their purpose is to make experiment history easier to trace.

Eventually, I want this lineage to extend across model versions, deployments, and predictions. A model should have a history, not just a filename.

---

## Project roadmap

The plan is a focused, reliable, traceable training workflow, not a copy of a large MLOps suite.

| Version | Mission |
|---|---|
| `v0.1.0` | Training engine, evaluation, reproducibility, lineage, and MLflow tracking |
| `v0.2.0` | FastAPI backend, API-key protection, CSV uploads, training, run lookup, and prediction endpoints |
| `v0.3.0` | PostgreSQL persistence, database migrations, and Docker Compose deployment |
| `v0.4.0` | Dataset, training job, and model management APIs with pagination and resource metadata |
| `v0.4.1` | Predictions use the model metadata saved in the database; failed training is recorded reliably |
| `v0.5.0` | Release hygiene and reliability: one dependency source of truth, a verified clean install, and docs that match the code |
| `v0.6.0` | End-to-end experience and demo: a clearer path through upload, training, results, lineage, and predictions, plus a reproducible demo |
| `v1.0.0` | Completion: a new user can clone, configure, upload, train, inspect, trace, and predict using only the docs and demo |

Out of scope for v1.0: Kubernetes or distributed training, enterprise-scale orchestration, every possible model family, and background workers unless the workflow actually needs them.

These are planned milestones, not promises that every feature already exists.

---

## What I'm learning along the way

ModelLore is also a learning project. I'm exploring how the pieces of a real ML system fit together, from data validation and model persistence to experiment tracking, reproducibility, and eventually deployment.

The interesting part isn't just getting a model to predict correctly. It's building the infrastructure that makes those predictions traceable, repeatable, and useful.

There will probably be bugs, questionable design decisions, and a few moments where the machine refuses to cooperate. That's part of the experiment.

## The destination

A trained model is not the finish line.

It's one checkpoint in a much longer journey.

**The mission is simple: train it, track it, trace it, and know how it came to be.**

The rest of the universe can wait until the foundations work.

## Database and Docker Setup

ModelLore uses PostgreSQL for persistent metadata and Docker Compose for local deployment.

### Run with Docker Compose

Prerequisites: Docker Desktop with the Docker Compose plugin enabled.

Set the API key in your shell before starting the services:

PowerShell

```powershell
$env:MLFLOWHUB_API_KEY = "local-dev-key-change-me"
docker compose up --build
```

Git Bash

```bash
export MLFLOWHUB_API_KEY="your-local-api-key"
docker compose up -d --build
```

The API container runs Alembic migrations before starting Uvicorn. PostgreSQL must become healthy before the API starts.

Check service status and database connectivity:

```bash
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/health/database
```

Protected API endpoints require the `X-API-Key` request header.

### Local database configuration

Set `DATABASE_URL` to your PostgreSQL connection URL when running the application outside Docker. Apply migrations before starting the API:

```bash
uv run alembic upgrade head
```

The default application configuration uses SQLite when `DATABASE_URL` is not set.

### Persistent storage

Docker Compose persists PostgreSQL data in a named volume and mounts dataset uploads, model artifacts, lineage metadata, and MLflow runs from the project directory. Avoid `docker compose down -v` when you want to preserve the database volume.
