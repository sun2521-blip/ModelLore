# ML-Flow-Hub# MLFlowHub

### Every model has a story. Let's find out what happened before it learned to predict the future.

Imagine a world where training a machine learning model is the easy part. The real mystery begins afterwards.

Which dataset did it learn from? What configuration shaped its decisions? Can we reproduce its results? And when a model suddenly starts behaving like it has a mind of its own, can we trace its history?

That is where **MLFlowHub** comes in.

MLFlowHub is my attempt at building a mini machine learning platform that takes a model from raw data to a trained, evaluated, saved, and traceable artifact. Instead of treating model training as a script that runs once and disappears into the void, I want every experiment to leave behind a trail that we can actually investigate.

Think of it as a small mission control for machine learning experiments.

And this is just the beginning.

---

## The mission

MLFlowHub is being built around one idea: **a model should never be separated from the story of how it was created.**

The goal is to build a platform that can eventually handle the journey from training a model to tracking its versions, serving predictions through an API, and monitoring its behavior.

For now, the focus is on getting the training engine right.

### What it can do right now

- **Configuration-driven training:** Define dataset settings, model parameters, random seeds, and train-test split settings in YAML.
- **Dataset loading and validation:** Load the built-in Iris dataset or your own CSV dataset, with checks for common data problems.
- **Preprocessing pipelines:** Handle numeric and categorical features using scikit-learn transformers, including imputation and one-hot encoding.
- **Model training and evaluation:** Train a scikit-learn pipeline and evaluate its predictions using classification metrics.
- **Reproducibility:** Use controlled random seeds and SHA-256 fingerprints for datasets and configurations.
- **Experiment tracking:** Use MLflow to record training parameters, metrics, and model artifacts.
- **Model lineage:** Store JSON records connecting a training run to its dataset fingerprint, configuration fingerprint, metrics, and saved model artifact.
- **Inference:** Load a saved model and use it to make predictions on new input.
- **Automated testing:** Test the important pieces instead of relying on the classic "it worked on my machine" philosophy.

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
       MODEL ARTIFACT      EXPERIMENT TRACKING
          .joblib               MLflow
             |                 |
             v                 v
          INFERENCE        PARAMETERS
          AND PREDICTIONS  METRICS AND ARTIFACTS
             |
             v
          JSON LINEAGE
     Dataset -> Config -> Run -> Model
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
| pytest | Automated testing |
| Git and GitHub Actions | Version control and CI |

The current version focuses on tabular classification. Regression support, additional model families, and deployment infrastructure are future work rather than features this release claims to provide.

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/sun2521-blip/ML-Flow-Hub.git
cd ML-Flow-Hub
```

### 2. Set up the environment

This project has been developed and tested with Python 3.13.

If you have `uv` installed, you can create an environment and install the project dependencies like this:

```bash
uv venv .venv --python 3.13
source .venv/Scripts/activate
uv pip install -r requirements.txt
```

On Windows, the activation command above is intended for Git Bash. Other terminals use different activation commands.

### 3. Run the tests

```bash
python -m pytest tests -v
```

The tests cover configuration validation, dataset loading, preprocessing, reproducibility, lineage, experiment tracking, and inference.

### 4. Train your first model

The default configuration uses the Iris dataset.

```bash
python -m ml.training.train
```

The training pipeline loads the configuration, prepares the data, trains and evaluates the model, saves a unique model artifact, and creates a JSON lineage record. It also logs the training run to MLflow.

The terminal prints the evaluation metrics, saved model path, lineage record path, and MLflow run ID.

### 5. Enter the experiment room

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

MLFlowHub also supports CSV datasets through YAML configuration. For example, a dataset configuration can look like this:

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

## The lineage experiment

This is one of the parts I find most interesting.

A model's metrics tell us how it performed. They don't tell us the whole story behind that performance.

MLFlowHub creates a SHA-256 fingerprint for the loaded dataset and another for the configuration. It then records those fingerprints alongside the run ID, model parameters, evaluation metrics, and artifact path.

Each training run gets a unique ID and its own saved model artifact.

That gives us a starting point for answering questions like:

- Did two experiments use the same dataset and configuration?
- Which parameters were used to train a particular model?
- Which metrics belonged to a specific run?
- Where is the model artifact associated with that run?

These fingerprints are identifiers, not proof that a model is correct or secure. Their purpose is to make experiment history easier to trace.

Eventually, I want this lineage to extend across model versions, deployments, and predictions. A model should have a history, not just a filename.

---

## Project roadmap

The long-term plan is to evolve MLFlowHub from a training engine into a small end-to-end ML platform.

| Version | Mission |
|---|---|
| `v0.1.0` | Training engine, evaluation, reproducibility, lineage, and MLflow tracking |
| `v0.2.0` | FastAPI backend |
| `v0.3.0` | PostgreSQL and Docker Compose |
| `v0.4.0` | Asynchronous training with Redis and Celery |
| `v0.5.0` | Model registry and model serving |
| `v0.6.0` | CI/CD and AWS deployment |
| `v0.7.0` | Model monitoring |
| `v0.8.0` | Polish, documentation, and portfolio-ready demos |

These are planned milestones, not promises that every feature already exists. The first release is about making the training engine reliable before building the infrastructure around it.

---

## What I'm learning along the way

MLFlowHub is also a learning project. I'm exploring how the pieces of a real ML system fit together, from data validation and model persistence to experiment tracking, reproducibility, and eventually deployment.

The interesting part isn't just getting a model to predict correctly. It's building the infrastructure that makes those predictions traceable, repeatable, and useful.

There will probably be bugs, questionable design decisions, and a few moments where the machine refuses to cooperate. That's part of the experiment.

## The destination

A trained model is not the finish line.

It's one checkpoint in a much longer journey.

MLFlowHub is being built to make that journey visible, from the data that started it to the model that came out the other side.

**The mission is simple: train it, track it, trace it, and eventually deploy it.**

The rest of the universe can wait until the foundations work.