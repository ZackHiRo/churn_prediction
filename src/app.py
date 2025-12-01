from typing import Any, Dict

from fastapi import FastAPI
from mangum import Mangum

from src.predict import load_production_model, predict_from_json

app = FastAPI(title="Churn Prediction API")


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
    preds = predict_from_json(model, payload)
    return {"predictions": preds}


handler = Mangum(app)


