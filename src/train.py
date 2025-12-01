import os
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from feast import FeatureStore
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import get_mlflow_config
from src.processing.data_loader import load_data, train_test_split_data


def _get_feature_types(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    return numeric_features, categorical_features


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features, categorical_features = _get_feature_types(X)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False,
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return clf


def configure_mlflow() -> MlflowClient:
    cfg = get_mlflow_config()

    # Set tracking URI from env (DagsHub / MLflow)
    mlflow.set_tracking_uri(cfg.tracking_uri)

    # Optional: if running against DagsHub, you can use a token via env vars
    # DAGSHUB_TOKEN or MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_experiment(cfg.experiment_name)
    return MlflowClient()


def load_training_data_with_optional_feast(
    data_path: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load training data either directly from CSV or by joining with Feast offline features,
    depending on USE_FEAST env var.
    """
    use_feast = os.getenv("USE_FEAST", "").lower() in {"1", "true", "yes"}

    # Basic CSV path for labels and (optionally) raw features
    X, y = load_data(data_path, target_column)

    if not use_feast:
        return X, y

    # When using Feast:
    # - DATA_PATH is expected to contain at minimum: [entity_column, event_timestamp, target_column]
    # - offline features are defined in features/definitions.py and materialized locally
    entity_column = os.getenv("FEAST_ENTITY_COLUMN", "customer_id")
    feature_view_name = os.getenv("FEAST_FEATURE_VIEW", "churn_features")

    if entity_column not in X.columns:
        raise ValueError(
            f"Expected entity column '{entity_column}' in training CSV when USE_FEAST is enabled."
        )

    # Feast store expects feature_store.yaml in the repo (configured for local/file offline store)
    store = FeatureStore(repo_path="features")

    # Build a reference dataframe for Feast: entity + event_timestamp
    if "event_timestamp" not in X.columns:
        raise ValueError(
            "Column 'event_timestamp' must be present in the CSV when USE_FEAST is enabled."
        )

    entity_df = X[[entity_column, "event_timestamp"]].copy()

    training_df = store.get_historical_features(
        entity_df=entity_df,
        feature_refs=[f"{feature_view_name}:*"],
    ).to_df()

    # Ensure target column is aligned with features by merging on entity + timestamp
    label_df = X[[entity_column, "event_timestamp"]].copy()
    label_df[target_column] = y.values

    joined = pd.merge(
        training_df,
        label_df,
        on=[entity_column, "event_timestamp"],
        how="inner",
    )

    if joined.empty:
        raise ValueError("No rows after joining Feast features with labels; check entity/timestamps.")

    y_final = joined[target_column]
    X_final = joined.drop(columns=[target_column])
    return X_final, y_final


def log_and_register_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: Optional[str] = "xgboost_churn_model",
    register_as: Optional[str] = "ChurnModel",
):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log model artifact
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=register_as,
        )

        run_id = run.info.run_id
        print(f"Run {run_id} logged to MLflow. accuracy={acc:.4f}, f1={f1:.4f}, roc_auc={roc:.4f}")


def main():
    """
    Main training entrypoint:
    - Load CSV data from DATA_PATH (env) and TARGET_COLUMN (env)
    - Train XGBoost classifier inside sklearn pipeline
    - Log metrics + model to MLflow (DagsHub)
    """
    data_path = os.getenv("DATA_PATH", "data/churn.csv")
    target_column = os.getenv("TARGET_COLUMN", "churn")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    X, y = load_training_data_with_optional_feast(data_path, target_column)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    clf = build_pipeline(X_train)

    client = configure_mlflow()

    clf.fit(X_train, y_train)

    log_and_register_model(clf, X_test, y_test)


if __name__ == "__main__":
    main()


