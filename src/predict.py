import os
from typing import Any, Dict, List

import mlflow
import pandas as pd

from src.config import get_mlflow_config


def load_production_model() -> mlflow.pyfunc.PyFuncModel:
    cfg = get_mlflow_config()
    mlflow.set_tracking_uri(cfg.tracking_uri)

    model_name = os.getenv("MLFLOW_MODEL_NAME", "ChurnModel")
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")

    uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(uri)


def predict_from_json(model: mlflow.pyfunc.PyFuncModel, payload: Dict[str, Any]) -> List[float]:
    """
    Accepts either:
    - {"records": [ {feature: value, ...}, ... ]}
    - or a single record dict {feature: value, ...}
    """
    if "records" in payload:
        df = pd.DataFrame(payload["records"])
    else:
        df = pd.DataFrame([payload])

    preds = model.predict(df)
    # Ensure list of floats (probabilities or labels depending on model)
    if hasattr(preds, "tolist"):
        return preds.tolist()
    return list(preds)


