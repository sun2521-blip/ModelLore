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

- **The console:** A browser front end, served by the API itself, for inserting datasets, recording training runs, playing back their metrics, taking a model apart into its lineage, and making predictions.

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
| React, Vite, and TypeScript | The browser console |
| Vitest | Console tests |
| uv | Python environment and dependency management |
| Git and GitHub Actions | Version control and CI |

---

## Getting started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Node.js](https://nodejs.org/) 24+ with npm (to build or work on the console)
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

The console has its own tests:

```bash
cd frontend
npm ci
npm test
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

The console keeps its own lock: `frontend/package.json` lists its packages with exact versions, and `frontend/package-lock.json` pins everything underneath. Install with `npm ci` so the lock is followed exactly. CI builds and tests the console and runs `npm audit` on it.

---

## Open the console

Curl commands are honest, but they don't make it easy to see a model's story at a glance. So ModelLore also has a console: a small browser front end that works like the control panel of a lineage recorder. You insert a dataset, press record, play back the runs, and take a finished model apart to see what it was made of.

The API serves the console itself, from the same address, so the browser never has to talk to another origin.

### Build it once

```bash
cd frontend
npm ci
npm run build
cd ..
```

Then start the API as in [Getting started](#5-run-the-api) and open http://127.0.0.1:8000/. Rebuild after pulling changes to the console. Until the console is built, `/` answers with a note saying so; the API itself works either way.

### What's on the panel

- **00 Access:** paste your `MLFLOWHUB_API_KEY`. The console checks it against the API before keeping it. It stays in the page's memory, and in the tab's session storage only if you tick "Keep for this tab"; it is never saved anywhere longer-lived.

- **01 Insert dataset:** drop a CSV, or choose one. The API validates and fingerprints it, and the console reads the header row so you can pick the target column.

- **02 Record:** train on Iris or on an inserted dataset. The counter runs while the model trains.

- **03 Playback:** every training run becomes a bar, as tall as its accuracy. Move the orange playhead with a click or the arrow keys to read out that run's metrics, parameters, and MLflow run.

- **04 Trace:** an exploded drawing of the selected model's lineage, from the dataset at the base, through its configuration and training run, up to the model artifact, with each layer's fingerprint.

- **05 Play:** fill in a row of features and press play to get a prediction from the selected model.

### Working on the console

Run the API on port 8000 as usual, and start Vite's development server in a second terminal:

```bash
cd frontend
npm run dev
```

Open http://127.0.0.1:5173. Vite forwards API calls to `http://127.0.0.1:8000`, so the browser still sees a single origin and CORS can stay closed. If port 8000 is busy (a Docker stack, say), run the API on another port and point Vite at it with `MODELLORE_API_URL`:

Git Bash

```bash
MODELLORE_API_URL=http://127.0.0.1:8001 npm run dev
```

PowerShell

```powershell
$env:MODELLORE_API_URL = "http://127.0.0.1:8001"
npm run dev
```

## Deploy it

The hosted demo runs in two halves. The API lives on [Render](https://render.com) as a free web service built from the `Dockerfile`, and the console lives on [Vercel](https://vercel.com) as a static site. Vercel forwards every API request to Render, so the browser only ever talks to one address and CORS can stay closed.

### What the free demo trades away

- **It forgets.** Render's free web services have no persistent disk, so the demo keeps its SQLite database and its files on the instance itself. Everything resets together whenever Render restarts, redeploys, or wakes the service after 15 quiet minutes. That keeps the demo consistent, just empty after a nap.
- **It naps.** The first request after a nap can take about a minute while Render wakes the service. The console says so and reconnects on its own.
- **It is shared.** The demo key is shown on the page, so anyone can upload, train, and predict. The limits in `render.yaml` are much smaller than the defaults (1 MiB uploads, 5,000 rows), and the rate limits are shared by every visitor, because they all arrive through the same proxies. Don't upload anything private to it.

For data that survives restarts, move the service to a paid plan with a [persistent disk](https://render.com/docs/disks) and point `DATABASE_URL` at a PostgreSQL database instead.

### 1. The API on Render

In the Render dashboard, choose **New** > **Blueprint** and pick this repository. Render reads `render.yaml` and creates a free web service called `modellore-api`.

When Render asks for `MLFLOWHUB_API_KEY`, enter the key the demo will use. It will be public, so make it a value you use nowhere else. Render keeps it as a secret; it is never written into the repository.

Render builds the image, runs the migrations, and deploys. After that, it redeploys on its own whenever a commit on `main` passes CI. Check that it is up:

```bash
curl https://modellore-api.onrender.com/health
```

If Render gives the service a different address, replace `modellore-api.onrender.com` in `frontend/vercel.json` with it.

### 2. The console on Vercel

In Vercel, import this repository and set **Root Directory** to `frontend`. Vercel reads `frontend/vercel.json`, which builds the console with Vite and forwards the API paths to Render.

Add one environment variable before the first deploy:

| Variable | Value |
|---|---|
| `VITE_DEMO_KEY` | the same key you gave Render |

With it set, the console shows the key with a "Use demo key" button and a note about what a shared demo means. Leave it unset for a console that only you can use with your own key.

Deploy, then open the address Vercel gives you.

---

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

- CSV files only, saved under a server-generated name; the name you upload with is kept for display only.

- Maximum file size of 10 MiB, 100,000 data rows, and 200 columns.

- At least one data row and at least two columns (features and a target), with every row the same width as the header.

- Invalid UTF-8, NUL bytes, empty files, and invalid or duplicate headers are rejected.

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

### 8. Trace a model

Use the same `lineage_run_id` to take the model apart:

```bash
curl "http://127.0.0.1:8000/lineage/LINEAGE_RUN_ID" \
  -H "X-API-Key: $MLFLOWHUB_API_KEY"
```

The response follows the model's lineage from the bottom up: the dataset (its fingerprint, and the uploaded file it came from), the configuration (its fingerprint and settings), the training run (metrics, MLflow run, and git commit), and the model artifact (its SHA-256 and the Python, scikit-learn, and `uv.lock` it was trained with). Server file paths stay on the server.

`GET /runs/MLFLOW_RUN_ID` also returns the run's `dataset_hash`, `config_hash`, and `lineage_run_id` tags, so an MLflow run can be traced back the same way.

### Older models

`/predict` finds a model by its `lineage_run_id`, checks each row against the model's saved `feature_columns`, and verifies the artifact's recorded SHA-256 before loading it. Models registered by earlier versions were saved without some of these fields, so the current prediction flow can't use them:

- No `lineage_run_id` (registered before v0.4.1): `/predict` can't find the model and returns `404`.
- No `feature_columns` (registered before v0.4.1): `/predict` returns `409` and asks you to retrain.
- No recorded artifact hash (registered before v0.5.0): `/predict` returns `409` and asks you to retrain.

Retrain affected models with `POST /train`; the new model is saved with all three. Run `uv run alembic upgrade head` after upgrading so the database has the new column. Separately, lineage records written before configuration snapshots were added can't be [verified by rerunning](#verify-a-training-run).

### API behavior and current limits

- `GET /health` and `GET /limits` are public; protected endpoints require `X-API-Key`. `/limits` reports the limits below as this server has them set.

- If `MLFLOWHUB_API_KEY` is unset or blank, protected requests fail closed.

- CSV uploads accept `.csv` files up to 10 MiB, 100,000 rows, and 200 columns, and validate encoding, headers, and row shape.

- `target_column` is optional. When omitted for uploaded CSV training, the configured target column is used.

- Uploaded datasets are referenced by IDs rather than arbitrary client-supplied filesystem paths, and only datasets the API registered can be trained on.

- A prediction request can carry up to 1,000 rows.

- Uploads, training, and predictions are rate limited per client: 10, 5, and 60 requests a minute. Going over returns `429` with a `Retry-After` header.

- Requests with unknown fields are rejected with `422`.

These limits can be changed with environment variables:

| Variable | Default |
|---|---|
| `MODELLORE_MAX_UPLOAD_MB` | 10 |
| `MODELLORE_MAX_UPLOAD_ROWS` | 100000 |
| `MODELLORE_MAX_UPLOAD_COLUMNS` | 200 |
| `MODELLORE_MAX_PREDICT_ROWS` | 1000 |
| `MODELLORE_RATE_LIMIT_UPLOAD` | 10 per minute |
| `MODELLORE_RATE_LIMIT_TRAIN` | 5 per minute |
| `MODELLORE_RATE_LIMIT_PREDICT` | 60 per minute |
| `MODELLORE_CORS_ORIGINS` | empty: no browser on another origin may call the API |

`MODELLORE_CORS_ORIGINS` takes a comma-separated list of exact origins, for example `http://localhost:5173`. Wildcards are refused at startup.

This is an early-stage local development API, not a hardened public production service. See [Security baseline](#security-baseline) for what it does and doesn't protect against.

---

## Security baseline

ModelLore aims for a sensible baseline for a project that runs on your own machine. It is not hardened for public, multi-user deployment.

### What is covered

- **Authentication:** every endpoint except the health checks and the interactive docs needs `X-API-Key`. The key is compared in constant time, there is no default key, and an unset or blank key makes the API refuse all protected requests instead of opening up.

- **Model artifacts:** joblib files can run code when they are loaded, so ModelLore treats them carefully. It only loads artifacts it wrote itself, from the `artifacts/` directory, and only after checking that the file's SHA-256 still matches the hash recorded at training time. The file is read once and loaded from those same bytes. A model is never loaded from a path or file supplied by a client.

- **Uploads:** size, row, and column limits; extension and content checks; server-generated file names; client file names reduced to a plain display name, so path tricks like `../../` go nowhere. CSV cells are parsed as data and never evaluated.

- **Input validation:** request bodies and parameters are checked with Pydantic, unknown fields are rejected, and YAML configuration is loaded with `yaml.safe_load` only. Database access goes through SQLAlchemy with bound parameters.

- **Errors and logs:** clients get messages written for them or a generic message, never stack traces, file paths, or dataset values. ModelLore's own log lines record the event, the IDs involved, and the type of error; they never record API keys, dataset contents, request bodies, or exception messages. MLflow keeps only the failure type for a failed run, not its traceback.

- **Network:** CORS is off unless you list exact origins. Uploads, training, and predictions are rate limited. In Docker Compose, PostgreSQL and the API listen on `127.0.0.1` only, there is no default database password, and the API container runs as an unprivileged user.

- **The console:** it is served from the API's own origin, so CORS can stay off. Its pages carry a strict content security policy: scripts, styles, fonts, and API calls may only come from that same origin, and the page can't be framed by another site. The API key is sent only in the `X-API-Key` header, kept in the page's memory, and never written to long-lived browser storage.

- **Dependencies:** every package is pinned with a hash in `uv.lock` and `requirements.txt`, and the set was checked with `pip-audit` for v0.5.0 with no known vulnerabilities found. The console's packages are pinned in `frontend/package-lock.json` and checked with `npm audit` in CI.

### What is not covered

- **One shared key:** there are no user accounts, roles, or per-user permissions. Anyone with the key can do everything.

- **No TLS:** traffic is plain HTTP. Put a reverse proxy with HTTPS in front if the API ever leaves your machine.

- **A simple rate limiter:** counts live in memory, reset when the API restarts, aren't shared between processes, and key on the client's IP address, which a proxy hides. Under Docker Compose every caller arrives from the Docker network's gateway address, so all clients share one budget.

- **Trust in the artifact directory:** the hash check catches changed files, but anyone who can write to both `artifacts/` and the database could replace a model and its hash together.

- **Uploads are checked after they arrive:** an upload that declares its size is refused straight away if it's too big, but one sent without a declared size is received in full before it is refused.

- **Unexpected errors:** an error the API doesn't handle at all still returns a plain `500` to the client, but Uvicorn writes its traceback to the server log.

- **The console trusts its browser:** while the page is open, the key sits in its memory, and in the tab's session storage if you ask it to. Anything that can run code in that page, such as a malicious browser extension, could read it.

- **No audit trail** of who called which endpoint, and no automated secret rotation.

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

Job records expose available metadata such as job status, metrics, errors, and timestamps, plus the `model_id` and `lineage_run_id` of the model a finished job produced.

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

Model records include available model metadata and related training-job information, along with what you need to use and trace the model: its `lineage_run_id`, the `feature_columns` it expects, the `dataset_id` it was trained on, and its `artifact_sha256`. Artifact filesystem paths are not exposed by these management responses.

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

Compose needs two secrets: the API key and a password for the PostgreSQL database. There are no built-in defaults, so the services won't start until both are set. The easiest place is `.env` (copy `.env.example`), which Compose reads on its own. Or set them in your shell:

PowerShell

```powershell
$env:MLFLOWHUB_API_KEY = "local-dev-key-change-me"
$env:POSTGRES_PASSWORD = "local-db-password-change-me"
docker compose up --build
```

Git Bash

```bash
export MLFLOWHUB_API_KEY="your-local-api-key"
export POSTGRES_PASSWORD="your-local-db-password"
docker compose up -d --build
```

Use letters and digits for `POSTGRES_PASSWORD`; it is placed inside the database URL, where characters such as `@`, `:` and `/` would break it.

The image builds the console as part of `docker compose up --build`, so there is nothing to install for it. Once the services are up, open http://localhost:8000/ for the console, or http://localhost:8000/docs for the API reference.

The API container runs Alembic migrations before starting Uvicorn. PostgreSQL must become healthy before the API starts. Both services listen on `127.0.0.1` only, so they are reachable from your own machine but not from the rest of the network.

**Upgrading an existing setup?** PostgreSQL only reads `POSTGRES_PASSWORD` when it first creates its data volume. Keep using the password your volume was created with. If you never set one, that was the old built-in `modellore_dev_password`: set `POSTGRES_PASSWORD` to that value, then change it with `ALTER USER` if you like.

**On Linux:** the API container runs as an unprivileged user (UID 10001), not as root. Give that user the folders Compose mounts before the first start:

```bash
mkdir -p data/uploads artifacts lineage mlruns
sudo chown -R 10001:10001 data/uploads artifacts lineage mlruns
```

Docker Desktop on Windows and macOS handles this for you.

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

Docker Compose persists PostgreSQL data in a named volume and mounts dataset uploads, model artifacts, lineage metadata, and MLflow runs from the project directory. MLflow's run database lives at `mlruns/mlflow.db` on that mount, so run history survives rebuilds and `docker compose down`. Avoid `docker compose down -v` when you want to preserve the database volume.
