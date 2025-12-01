## Churn Prediction – Serverless, Zero-Cost-Friendly Setup

This project is a production-grade churn prediction system built to run on free-tier resources:

- **Training compute**: GitHub Actions
- **Inference**: AWS Lambda (container image)
- **Experiment tracking**: MLflow hosted on DagsHub
- **Feature store**: Feast in **offline-only** mode over local Parquet files

### Layout

- `src/` – core Python package
  - `config.py` – loads MLflow/DagsHub config from env vars
  - `train.py` – training entrypoint (CSV or Feast-based features) + MLflow logging
  - `predict.py` – loads Production model from MLflow and runs inference
  - `app.py` – FastAPI app, wrapped with Mangum for AWS Lambda
- `features/` – Feast config (`feature_store.yaml`, `definitions.py`)
- `.github/workflows/` – CI for tests + training, CD for pushing Lambda image to ECR
- `Dockerfile` – Lambda-optimized container using `public.ecr.aws/lambda/python:3.10`

### Key Environment Variables

**MLflow / DagsHub**

- `MLFLOW_TRACKING_URI` – MLflow tracking URI (DagsHub project URI)
- `MLFLOW_EXPERIMENT_NAME` – experiment name (default: `churn_prediction`)
- `DAGSHUB_TOKEN` – personal access token for DagsHub (used as `MLFLOW_TRACKING_USERNAME`/`PASSWORD`)
- `MLFLOW_MODEL_NAME` – registered model name for inference (default: `ChurnModel`)
- `MLFLOW_MODEL_STAGE` – model stage for inference (default: `Production`)

**Training data**

- `DATA_PATH` – path to the CSV file (default: `data/churn.csv`)
- `TARGET_COLUMN` – name of the target column in the CSV (default: `churn`)

**Feast / Feature store (optional)**

- `USE_FEAST` – if set to `"1"` or `"true"`, training uses Feast feature retrieval
- `FEAST_FEATURE_VIEW` – name of the `FeatureView` to pull from (default: `churn_features`)
- `FEAST_ENTITY_COLUMN` – entity column name in the label CSV (default: `customer_id`)
- `FEAST_REGISTRY_PATH` – optional override for the Feast registry path; otherwise uses `features/feature_store.yaml`

### Training Workflows

**1. Local CSV-based training**

```bash
export MLFLOW_TRACKING_URI=...
export DAGSHUB_TOKEN=...
export DATA_PATH=data/churn.csv
export TARGET_COLUMN=churn

python -m src.train
```

This will:

- Load `DATA_PATH` CSV
- Train an XGBoost classifier (inside a scikit-learn pipeline)
- Log **accuracy**, **F1**, **ROC-AUC** and the model artifact to MLflow
- Register the model as `ChurnModel` in the MLflow Model Registry

**2. Feast-based training (offline features)**

1. Create a Parquet file with features and timestamps (matching `features/definitions.py` schema).
2. Initialize Feast metadata locally (e.g., `feast apply` from the repo root).
3. Run:

```bash
export MLFLOW_TRACKING_URI=...
export DAGSHUB_TOKEN=...
export DATA_PATH=data/labels.csv      # labels + entity id + timestamp
export TARGET_COLUMN=churn
export USE_FEAST=true

python -m src.train
```

`src/train.py` will:

- Load labels from `DATA_PATH`
- Use Feast to join offline features from local Parquet
- Train and log the model to MLflow as above

### Inference (Lambda + FastAPI)

The `Dockerfile` builds a Lambda-compatible image:

```bash
docker build -t churn-lambda .
```

The container:

- Starts the Lambda runtime with `CMD ["src.app.handler"]`
- `src/app.py` creates the FastAPI app and loads the latest **Production** model from MLflow on cold start
- Exposes `POST /predict` that accepts:
  - A single JSON record: `{ "feature1": ..., "feature2": ... }`, or
  - Multiple records: `{ "records": [ { ... }, { ... } ] }`

### CI / CD

- **`.github/workflows/train.yaml`**
  - Installs dependencies
  - Runs tests with `pytest`
  - Executes `python -m src.train` with MLflow/DagsHub secrets
- **`.github/workflows/deploy.yaml`**
  - On GitHub release, builds the Docker image and pushes it to ECR using OIDC (no long-lived AWS keys)


