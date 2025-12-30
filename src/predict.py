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
    import sys
    print("=" * 50, flush=True)
    print("predict_from_json called", flush=True)
    print(f"Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'not a dict'}", flush=True)
    sys.stdout.flush()
    
    if "records" in payload:
        df = pd.DataFrame(payload["records"])
    else:
        df = pd.DataFrame([payload])
    
    print(f"DataFrame created with shape: {df.shape}", flush=True)
    print(f"Initial columns: {list(df.columns)}", flush=True)
    sys.stdout.flush()

    # Handle ID columns: The model expects customerID if it was in training data.
    # The ColumnTransformer was fitted with customerID, so we need to provide it.
    id_column_variants = [
        "customerID",
        "customer_id",
        "CustomerID",
        "Customer_ID",
        "customerId",
        "CustomerId",
    ]

    # Check if any ID column variant exists
    existing_id_col = None
    for col in id_column_variants:
        if col in df.columns:
            existing_id_col = col
            break

    # If customerID is missing but model expects it, add/rename it.
    # The OneHotEncoder will handle unseen values (handle_unknown="ignore").
    if "customerID" not in df.columns:
        if existing_id_col:
            # Rename existing ID column to customerID
            df = df.rename(columns={existing_id_col: "customerID"})
        else:
            # Add dummy customerID - unique per row
            df["customerID"] = "pred_" + pd.Series(range(len(df))).astype(str)

    # CRITICAL: Convert types IMMEDIATELY to match training data types
    # The ColumnTransformer selects columns by dtype, so types must match exactly
    # IMPORTANT: Only handle INPUT columns - FeatureEngineer creates derived columns inside the pipeline
    
    # Define INPUT categorical columns (raw data from user - will be OneHotEncoded)
    # These must be object dtype (string) to match training
    input_categorical_cols = [
        "customerID",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "Partner",
        "Dependents",
        "gender",
    ]
    
    # Convert INPUT categorical columns to strings (object dtype)
    # This must happen BEFORE FeatureEngineer runs
    for col in input_categorical_cols:
        if col in df.columns:
            # Force to string, handling None/NaN and ensuring all values are strings
            df[col] = df[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
            # Ensure all values are actually strings (handle any mixed types)
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) and str(x) not in ['nan', 'None', ''] else 'Unknown')
        else:
            # If column is missing, add it with a default value
            df[col] = 'Unknown'
    
    # Ensure categorical columns are object dtype
    for col in input_categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')

    # Define INPUT numeric columns (raw data from user - will be StandardScaled)
    # These must be int64/float64 to match training
    input_numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
    ]
    
    # Convert INPUT numeric columns to proper numeric types
    for col in input_numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN, then fill NaN with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float64')
        else:
            # If column is missing, add it with default value 0
            df[col] = 0.0

    # Debug: Log column dtypes before prediction (helpful for troubleshooting)
    import sys
    print(f"DataFrame dtypes before prediction (input columns only):", flush=True)
    sys.stdout.flush()
    for col in input_categorical_cols + input_numeric_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype} (sample value: {df[col].iloc[0] if len(df) > 0 else 'N/A'})", flush=True)
    sys.stdout.flush()
    
    print(f"Total columns in DataFrame: {len(df.columns)}", flush=True)
    print(f"All columns: {list(df.columns)}", flush=True)
    sys.stdout.flush()

    # Now run prediction with error handling
    try:
        preds = model.predict(df)
    except Exception as e:
        # Log detailed error information
        print(f"ERROR during prediction: {type(e).__name__}: {str(e)}", flush=True)
        print(f"DataFrame shape: {df.shape}", flush=True)
        print(f"DataFrame dtypes:\n{df.dtypes}", flush=True)
        print(f"DataFrame info:\n{df.info()}", flush=True)
        sys.stdout.flush()
        raise

    # Ensure list of floats (probabilities or labels depending on model)
    if hasattr(preds, "tolist"):
        return preds.tolist()
    return list(preds)

