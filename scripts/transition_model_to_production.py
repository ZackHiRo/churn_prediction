#!/usr/bin/env python3
"""
Script to transition the latest model version to Production stage.
This is needed after retraining to ensure Lambda loads the new model.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlflow.tracking import MlflowClient
from src.config import get_mlflow_config

def transition_latest_model_to_production():
    """Transition the latest model version to Production stage."""
    cfg = get_mlflow_config()
    
    # Set up MLflow
    import mlflow
    mlflow.set_tracking_uri(cfg.tracking_uri)
    
    # Optional: if running against DagsHub, use token
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    
    model_name = os.getenv("MLFLOW_MODEL_NAME", "ChurnModel")
    client = MlflowClient()
    
    try:
        # Get all versions of the model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            print(f"No model versions found for '{model_name}'")
            print("The model may not be registered, or DagsHub doesn't support model registry.")
            return
        
        # Sort by version number (descending) to get the latest
        model_versions.sort(key=lambda x: x.version, reverse=True)
        latest_version = model_versions[0]
        
        print(f"Found {len(model_versions)} model version(s) for '{model_name}'")
        print(f"Latest version: {latest_version.version}")
        print(f"Current stage: {latest_version.current_stage}")
        print(f"Run ID: {latest_version.run_id}")
        
        # Transition to Production if not already there
        if latest_version.current_stage != "Production":
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Production",
                    archive_existing_versions=True  # Archive old Production versions
                )
                print(f"✓ Successfully transitioned version {latest_version.version} to Production")
            except Exception as e:
                print(f"✗ Failed to transition model to Production: {e}")
                print("This might be a DagsHub limitation. The model may need to be loaded by run ID instead.")
        else:
            print(f"Version {latest_version.version} is already in Production stage")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf model registry is not supported (e.g., DagsHub), you may need to:")
        print("1. Update Lambda environment variable MLFLOW_MODEL_STAGE to 'None'")
        print("2. Or modify predict.py to load from a specific run ID")

if __name__ == "__main__":
    transition_latest_model_to_production()

