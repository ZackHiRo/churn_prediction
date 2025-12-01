from fastapi.testclient import TestClient

from src.app import app


def test_predict_endpoint_smoke(monkeypatch):
    class DummyModel:
        def predict(self, df):
            return [0.1] * len(df)

    from src import predict as predict_module
    from src import app as app_module

    # Mock load_production_model before creating TestClient (which triggers startup)
    dummy_model = DummyModel()
    monkeypatch.setattr(predict_module, "load_production_model", lambda: dummy_model)
    # Also set the model directly in case startup already ran
    monkeypatch.setattr(app_module, "model", dummy_model)

    client = TestClient(app)

    response = client.post("/predict", json={"feature1": 1, "feature2": 2})
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)


