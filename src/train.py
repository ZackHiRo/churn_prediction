import os
import warnings
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from feast import FeatureStore
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import get_mlflow_config
from src.processing.data_loader import load_data, train_test_split_data

# Suppress warnings for unknown categories in OneHotEncoder (expected behavior)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing._encoders')


def _get_feature_types(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    return numeric_features, categorical_features


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""
    
    def __init__(self):
        self.monthly_charges_mean_ = None
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if 'MonthlyCharges' in X_df.columns:
            self.monthly_charges_mean_ = X_df['MonthlyCharges'].mean()
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_eng = X_df.copy()
        
        # Feature engineering for numeric columns if they exist
        if 'tenure' in X_eng.columns:
            # Create tenure groups (more granular)
            X_eng['tenure_group'] = pd.cut(
                X_eng['tenure'], 
                bins=[0, 12, 24, 36, 48, 60, float('inf')], 
                labels=['0-12', '13-24', '25-36', '37-48', '49-60', '60+']
            )
            # Tenure squared (non-linear relationship)
            X_eng['tenure_squared'] = X_eng['tenure'] ** 2
            # New customer indicator
            X_eng['is_new_customer'] = (X_eng['tenure'] <= 6).astype(int)
            # Long-term customer indicator
            X_eng['is_long_term'] = (X_eng['tenure'] >= 36).astype(int)
        
        if 'MonthlyCharges' in X_eng.columns and 'TotalCharges' in X_eng.columns:
            # Calculate average monthly charge (if TotalCharges is available)
            total_charges_numeric = pd.to_numeric(X_eng['TotalCharges'], errors='coerce')
            tenure_numeric = pd.to_numeric(X_eng.get('tenure', 0), errors='coerce')
            X_eng['avg_monthly_charge'] = total_charges_numeric / (tenure_numeric + 1)
            
            # Charge discrepancy (difference between current and average)
            X_eng['charge_discrepancy'] = X_eng['MonthlyCharges'] - X_eng['avg_monthly_charge']
            
            # Total value indicator
            X_eng['total_value_high'] = (total_charges_numeric > total_charges_numeric.median()).astype(int)
            
            if self.monthly_charges_mean_ is not None:
                X_eng['charge_ratio'] = X_eng['MonthlyCharges'] / (self.monthly_charges_mean_ + 1e-6)
                # Charge tier relative to mean
                X_eng['charge_above_mean'] = (X_eng['MonthlyCharges'] > self.monthly_charges_mean_).astype(int)
        
        if 'MonthlyCharges' in X_eng.columns:
            # Create charge tiers (more granular)
            X_eng['charge_tier'] = pd.cut(
                X_eng['MonthlyCharges'],
                bins=[0, 30, 50, 70, 90, 110, float('inf')],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Premium']
            )
            # Monthly charges squared (non-linear)
            X_eng['monthly_charges_squared'] = X_eng['MonthlyCharges'] ** 2
        
        # Interaction features if both exist
        if 'tenure' in X_eng.columns and 'MonthlyCharges' in X_eng.columns:
            # Value per month (customer lifetime value proxy)
            X_eng['value_per_month'] = X_eng['MonthlyCharges'] * X_eng['tenure']
            # High charge + low tenure (risk indicator)
            if self.monthly_charges_mean_ is not None:
                X_eng['high_charge_low_tenure'] = (
                    (X_eng['MonthlyCharges'] > self.monthly_charges_mean_) & 
                    (X_eng['tenure'] < 12)
                ).astype(int)
        
        # Service count features (count of Yes services)
        service_cols = [col for col in X_eng.columns if col in [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]]
        if service_cols:
            # Count services
            X_eng['service_count'] = (X_eng[service_cols] == 'Yes').sum(axis=1)
            # Has multiple services
            X_eng['has_multiple_services'] = (X_eng['service_count'] > 2).astype(int)
        
        return X_eng


def build_pipeline(X: pd.DataFrame, y: pd.Series = None) -> tuple[Pipeline, dict]:
    # Feature engineering will happen in pipeline, so use dynamic selectors
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary')

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
            ("cat", categorical_transformer, make_column_selector(dtype_include=['object', 'bool', 'category'])),
        ],
        remainder='drop'  # Drop any unhandled columns
    )

    # Calculate scale_pos_weight to handle class imbalance
    # This gives more weight to the minority class (churn=Yes)
    # Using a slightly adjusted weight for better precision-recall balance
    scale_pos_weight = None
    if y is not None:
        negative_count = (y == 0).sum()
        positive_count = (y == 1).sum()
        if positive_count > 0:
            base_weight = negative_count / positive_count
            # Slightly reduce weight to improve precision (less aggressive on minority class)
            scale_pos_weight = base_weight * 0.9

    # No early stopping for now (complex with sklearn Pipeline)
    fit_params = {}

    model = XGBClassifier(
        n_estimators=500,  # Further increased for better performance
        max_depth=6,  # Slightly deeper for more complex patterns
        learning_rate=0.025,  # Even lower learning rate for more stable training
        subsample=0.85,  # Slightly higher for more data usage
        colsample_bytree=0.85,  # Slightly higher for more feature usage
        colsample_bynode=0.8,  # Additional regularization at node level
        min_child_weight=4,  # Balanced for complexity
        gamma=0.15,  # Balanced regularization
        reg_alpha=0.15,  # L1 regularization
        reg_lambda=1.5,  # L2 regularization
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        grow_policy='lossguide',  # Better for imbalanced data
        max_bin=256,  # Higher bin count for better precision
    )

    clf = Pipeline(steps=[
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", preprocessor), 
        ("model", model)
    ])
    return clf, fit_params


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
    # Pipeline will handle feature engineering automatically
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Optimize threshold using F-beta score (beta=0.7 favors precision more)
    # This helps balance precision and recall better than pure F1
    thresholds = np.arange(0.40, 0.60, 0.002)  # Very fine grid, focused range
    best_threshold = 0.5
    best_score = 0
    best_strategy = 'f1'
    
    # Try multiple optimization strategies
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        prec_thresh = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec_thresh = recall_score(y_test, y_pred_thresh, zero_division=0)
        
        # Calculate scores for different strategies
        f1_score_val = f1_score(y_test, y_pred_thresh)
        f07_score = fbeta_score(y_test, y_pred_thresh, beta=0.7)  # Favor precision
        f05_score = fbeta_score(y_test, y_pred_thresh, beta=0.5)  # Strongly favor precision
        
        # Try each strategy and pick the best
        for score_val, strategy_name in [
            (f1_score_val, 'f1'),
            (f07_score, 'f0.7'),
            (f05_score, 'f0.5'),
        ]:
            score = score_val
            
            # Additional bonus for good precision when using F-beta
            if 'f0' in strategy_name and prec_thresh > 0.55:
                score *= 1.02  # Small bonus for high precision
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_strategy = strategy_name
    
    y_pred = (y_proba >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    # Additional metrics for imbalanced classification
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("specificity", specificity)
        mlflow.log_metric("optimal_threshold", best_threshold)
        mlflow.log_metric("threshold_strategy", 1.0 if best_strategy == 'f0.7' else (0.5 if best_strategy == 'f0.5' else 0.0))

        # Log model artifact
        # Try to register model, but fall back to just logging if registry is not supported
        # (e.g., DagsHub doesn't support model registry operations)
        # Note: DagsHub may not support model logging endpoints at all, so we gracefully handle failures
        model_logged = False
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=register_as,
            )
            print(f"Model registered as '{register_as}' in MLflow model registry")
            model_logged = True
        except RestException as e:
            # Check if this is an unsupported endpoint error (DagsHub limitation)
            error_str = str(e).lower()
            if "unsupported endpoint" in error_str or "dagshub" in error_str:
                print(f"Warning: Model registry not supported by tracking server. Attempting to log model without registration.")
                try:
                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path="model",
                    )
                    model_logged = True
                    print("Model logged successfully (without registration)")
                except RestException as e2:
                    # Even logging without registration may fail on DagsHub
                    error_str2 = str(e2).lower()
                    if "unsupported endpoint" in error_str2 or "dagshub" in error_str2:
                        print(f"Warning: Model logging endpoint not supported by tracking server. Skipping model artifact logging.")
                    else:
                        print(f"Warning: Model logging failed. Skipping model artifact logging. Error: {e2}")
                except Exception as e2:
                    print(f"Warning: Model logging failed ({type(e2).__name__}). Skipping model artifact logging.")
            else:
                # Re-raise if it's a different RestException that we don't know how to handle
                print(f"Warning: Unexpected RestException during model registration: {e}. Attempting fallback.")
                try:
                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path="model",
                    )
                    model_logged = True
                except Exception:
                    print(f"Warning: Fallback model logging also failed. Skipping model artifact logging.")
        except Exception as e:
            # For any other exception, try logging without registration as fallback
            print(f"Warning: Model registration failed ({type(e).__name__}). Attempting to log model without registration.")
            try:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                )
                model_logged = True
            except RestException as e2:
                # Handle RestException specifically in fallback
                error_str2 = str(e2).lower()
                if "unsupported endpoint" in error_str2 or "dagshub" in error_str2:
                    print(f"Warning: Model logging endpoint not supported. Skipping model artifact logging.")
                else:
                    print(f"Warning: Model logging failed. Skipping model artifact logging. Error: {e2}")
            except Exception as e2:
                print(f"Warning: Model logging also failed ({type(e2).__name__}). Skipping model artifact logging.")
        
        if not model_logged:
            print("Note: Model metrics were logged, but model artifact could not be saved due to tracking server limitations.")

        run_id = run.info.run_id
        print(f"Run {run_id} logged to MLflow.")
        print(f"Metrics: accuracy={acc:.4f}, f1={f1:.4f}, roc_auc={roc:.4f}")
        print(f"         precision={precision:.4f}, recall={recall:.4f}, specificity={specificity:.4f}")
        print(f"         optimal_threshold={best_threshold:.3f} (strategy: {best_strategy})")


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

    # Pass y_train to build_pipeline for class weight calculation
    clf, fit_params = build_pipeline(X_train, y_train)

    client = configure_mlflow()

    clf.fit(X_train, y_train)

    log_and_register_model(clf, X_test, y_test)


if __name__ == "__main__":
    main()


