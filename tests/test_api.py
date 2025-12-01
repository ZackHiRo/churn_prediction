from fastapi.testclient import TestClient

from src.app import app


def test_predict_endpoint_smoke(monkeypatch):
    client = TestClient(app)

    class DummyModel:
        def predict(self, df):
            return [0.1] * len(df)

    from src import predict as predict_module

    monkeypatch.setattr(predict_module, "load_production_model", lambda: DummyModel())

    response = client.post("/predict", json={"feature1": 1, "feature2": 2})
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)


