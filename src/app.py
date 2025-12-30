from typing import Any, Dict

from fastapi import FastAPI
from mangum import Mangum

from src.predict import load_production_model, predict_from_json

app = FastAPI(title="Churn Prediction API")

# Initialize model as None, will be loaded on startup
model = None


@app.on_event("startup")
def startup_event():
    # Load the latest Production model from MLflow on cold start
    global model
    model = load_production_model()


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    """
    POST /predict
    Body: JSON with either a single record or {"records": [ ... ]}
    """
    import sys
    print("=" * 50, flush=True)
    print("POST /predict endpoint called", flush=True)
    sys.stdout.flush()
    
    global model
    # Lazy load model if startup event hasn't run (e.g., in tests)
    if model is None:
        print("Loading model (lazy load)...", flush=True)
        sys.stdout.flush()
        model = load_production_model()
    
    try:
        preds = predict_from_json(model, payload)
        print(f"Prediction successful, returning {len(preds)} predictions", flush=True)
        sys.stdout.flush()
        return {"predictions": preds}
    except Exception as e:
        print(f"ERROR in predict endpoint: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        sys.stdout.flush()
        raise


handler = Mangum(app)


