import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str = "churn_prediction"
    dagshub_repo_owner: Optional[str] = None
    dagshub_repo_name: Optional[str] = None


def get_mlflow_config() -> MLflowConfig:
    """
    Load MLflow/DagsHub-related settings from environment variables.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable must be set.")

    return MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction"),
        dagshub_repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
        dagshub_repo_name=os.getenv("DAGSHUB_REPO_NAME"),
    )


def get_env_var(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} must be set.")
    return value


