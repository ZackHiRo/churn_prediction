import os
from typing import Any, Dict, List

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from src.config import get_mlflow_config


def load_production_model() -> mlflow.pyfunc.PyFuncModel:
    """
    Load production model from MLflow.
    
    Attempts to load from model registry first (models:/{name}/{stage}).
    Falls back to loading from the latest run if registry is not supported
    (e.g., DagsHub limitations).
    """
    cfg = get_mlflow_config()
    mlflow.set_tracking_uri(cfg.tracking_uri)

    # Optional: if running against DagsHub, you can use a token via env vars
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    model_name = os.getenv("MLFLOW_MODEL_NAME", "ChurnModel")
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")

    # Try to load from model registry first
    try:
        uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(uri)
        print(f"Loaded model from registry: {uri}")
        return model
    except RestException as e:
        # Check if this is an unsupported endpoint error (DagsHub limitation)
        error_str = str(e).lower()
        if "unsupported endpoint" in error_str or "dagshub" in error_str or "not found" in error_str:
            print(f"Warning: Model registry not supported or model not found. Falling back to latest run.")
            # Fallback: Load from latest run in the experiment
            try:
                mlflow.set_experiment(cfg.experiment_name)
                client = MlflowClient()
                
                # Search for the latest run with the model artifact
                experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
                if experiment is None:
                    raise ValueError(f"Experiment '{cfg.experiment_name}' not found.")
                
                # Get all runs, sorted by start_time descending
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=10
                )
                
                # Try to load model from each run until we find one
                for run in runs:
                    try:
                        run_uri = f"runs:/{run.info.run_id}/model"
                        model = mlflow.pyfunc.load_model(run_uri)
                        print(f"Loaded model from run: {run.info.run_id} (fallback from registry)")
                        return model
                    except Exception:
                        # Try next run if this one doesn't have a model
                        continue
                
                raise ValueError(
                    f"Could not find a model artifact in any recent runs. "
                    f"Please ensure at least one training run has logged a model."
                )
            except Exception as e2:
                raise ValueError(
                    f"Failed to load model from registry and fallback to latest run also failed: {e2}"
                )
        else:
            # Re-raise if it's a different RestException
            raise
    except Exception as e:
        # For any other exception, try the fallback
        print(f"Warning: Model loading from registry failed ({type(e).__name__}). Trying fallback.")
        try:
            mlflow.set_experiment(cfg.experiment_name)
            client = MlflowClient()
            
            experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{cfg.experiment_name}' not found.")
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=10
            )
            
            for run in runs:
                try:
                    run_uri = f"runs:/{run.info.run_id}/model"
                    model = mlflow.pyfunc.load_model(run_uri)
                    print(f"Loaded model from run: {run.info.run_id} (fallback)")
                    return model
                except Exception:
                    continue
            
            raise ValueError(
                f"Could not find a model artifact in any recent runs. "
                f"Please ensure at least one training run has logged a model."
            )
        except Exception as e2:
            raise ValueError(
                f"Failed to load model from registry and fallback also failed: {e2}"
            )


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

    # Handle ID columns: The model expects customerID if it was in training data
    # The ColumnTransformer was fitted with customerID, so we need to provide it
    id_column_variants = ['customerID', 'customer_id', 'CustomerID', 'Customer_ID', 
                          'customerId', 'CustomerId']
    
    # Check if any ID column variant exists
    existing_id_col = None
    for col in id_column_variants:
        if col in df.columns:
            existing_id_col = col
            break
    
    # If customerID is missing but model expects it, add a dummy value
    # The OneHotEncoder will handle it, and since it's a unique dummy value,
    # it won't match any training categories and will be ignored (handle_unknown='ignore')
    if 'customerID' not in df.columns:
        if existing_id_col:
            # Rename existing ID column to customerID
            df = df.rename(columns={existing_id_col: 'customerID'})
        else:
            # Add dummy customerID - will be encoded but won't affect prediction
            df['customerID'] = 'pred_' + pd.Series(range(len(df))).astype(str)

    # Try to predict, and if columns are missing, add them with defaults
    try:
        preds = model.predict(df)
    except ValueError as e:
        error_msg = str(e)
        if "columns are missing" in error_msg:
            # Try to extract missing columns from error message or model
            # Get expected columns from the sklearn pipeline if available
            try:
                if hasattr(model, 'sklearn_model'):
                    pipeline = model.sklearn_model
                    if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
                        preprocessor = pipeline.named_steps['preprocessor']
                        if hasattr(preprocessor, 'feature_names_in_'):
                            expected_cols = set(preprocessor.feature_names_in_)
                            provided_cols = set(df.columns)
                            missing_cols = expected_cols - provided_cols
                            
                            print(f"Warning: Missing columns detected: {missing_cols}")
                            print(f"Provided columns: {provided_cols}")
                            print(f"Expected columns: {expected_cols}")
                            
                            # Add missing columns with default values
                            for col in missing_cols:
                                if col.lower() in ['customerid', 'customer_id']:
                                    df[col] = 'pred_' + pd.Series(range(len(df))).astype(str)
                                elif df.select_dtypes(include=['int64', 'float64']).empty:
                                    # If no numeric columns, assume it's numeric
                                    df[col] = 0
                                else:
                                    # Try to infer type from existing columns
                                    df[col] = 0  # Default to 0 for numeric
                            
                            # Retry prediction
                            preds = model.predict(df)
                        else:
                            raise
                    else:
                        raise
                else:
                    raise
            except Exception as e2:
                # If we can't fix it automatically, re-raise with better error message
                print(f"Error: {error_msg}")
                print(f"Provided columns: {list(df.columns)}")
                raise ValueError(f"Missing columns error. Original: {error_msg}. Provided columns: {list(df.columns)}") from e
        else:
            raise
    
    # Ensure list of floats (probabilities or labels depending on model)
    if hasattr(preds, "tolist"):
        return preds.tolist()
    return list(preds)


