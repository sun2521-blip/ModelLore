
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

### Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/sun2521-blip/ModelLore.git
cd ModelLore
```

### 2. Set up the environment

This project has been developed with Python 3.13.

If you have `uv` installed, create an environment and install the project dependencies:

```bash
uv venv .venv --python 3.13
source .venv/Scripts/activate
uv pip install -r requirements.txt
```

On Windows, the activation command above is intended for Git Bash. Other terminals use different activation commands.

If you already have an existing project environment, you do not need to recreate it just to follow these instructions.

### 3. Run the tests

```bash
python -m pytest tests -v
```

The tests cover configuration validation, dataset loading, preprocessing, reproducibility, lineage, experiment tracking, inference, and API behavior.

### 4. Train your first model

The default configuration uses the Iris dataset.

```bash
python -m ml.training.train
```

The training pipeline loads the configuration, prepares the data, trains and evaluates the model, saves a unique model artifact, and creates a JSON lineage record. It also logs the training run to MLflow.

The terminal prints the evaluation metrics, saved model path, lineage record path, and MLflow run ID.

---

## Verify a training run

ModelLore can verify a completed training run by loading its lineage record, rerunning training with the saved configuration, and comparing the results.

### Run verification

First, train a model:

```bash
python -m ml.training.train
```

The command prints the path to the generated lineage record. Pass that path to the verification command:

```bash
python -m ml.verification lineage/YOUR_RUN_ID.json
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
mlflow ui
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Find the `mlflowhub-training` experiment to inspect your runs, parameters, metrics, and artifacts.

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
python -m ml.training.train configs/my_dataset.yaml
```

Replace the dataset path and target column with the values appropriate for your data. The current training pipeline is designed for classification datasets, and the target must contain enough examples of each class for a stratified train-test split.

---

## API usage

ModelLore includes a FastAPI service for health checks, CSV dataset uploads, training, MLflow run retrieval, and predictions.

Interactive API documentation is available at `/docs` while the service is running.

### 1. Configure an API key

The API reads its secret from the `MLFLOWHUB_API_KEY` environment variable.

For Windows using Git Bash:

```bash
export MLFLOWHUB_API_KEY="replace-with-a-long-random-secret"
```

Use a strong, randomly generated secret in real environments. Never commit real API keys to Git.

If the environment variable is not configured, protected endpoints fail closed rather than accepting unauthenticated requests.

### 2. Start the API

From the project root, with the project environment activated:

```bash
uvicorn api.main:app --reload
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

Send the lineage run ID and a list of feature rows. Feature names must match those expected by the trained model.

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

### API behavior and current limits

- `GET /health` is public; protected endpoints require `X-API-Key`.
- If `MLFLOWHUB_API_KEY` is unset, protected requests fail closed.
- CSV uploads accept `.csv` files up to 10 MiB and validate encoding and headers.
- `target_column` is optional. When omitted for uploaded CSV training, the configured target column is used.
- Uploaded datasets are referenced by IDs rather than arbitrary client-supplied filesystem paths.
- This is an early-stage local development API, not a hardened public production service. Do not expose it directly to the public internet without additional security, rate limiting, and operational controls.

---

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

The long-term plan is to evolve ModelLore from a training engine into a small end-to-end ML platform.

| Version | Mission |
|---|---|
| `v0.1.0` | Training engine, evaluation, reproducibility, lineage, and MLflow tracking |
| `v0.2.0` | FastAPI backend, API-key protection, CSV uploads, training, run lookup, and prediction endpoints |
| `v0.3.0` | PostgreSQL and Docker Compose |
| `v0.4.0` | Asynchronous training with Redis and Celery |
| `v0.5.0` | Model registry and expanded model serving |
| `v0.6.0` | CI/CD and AWS deployment |
| `v0.7.0` | Model monitoring |
| `v0.8.0` | UI polish, documentation, and portfolio-ready demos |

These are planned milestones, not promises that every feature already exists. The first releases focus on making the training engine reliable before building the infrastructure around it.

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
python -m alembic upgrade head
```

The default application configuration uses SQLite when `DATABASE_URL` is not set.

### Persistent storage

Docker Compose persists PostgreSQL data in a named volume and mounts dataset uploads, model artifacts, lineage metadata, and MLflow runs from the project directory. Avoid `docker compose down -v` when you want to preserve the database volume.