from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_root_route() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Enterprise Predictive Maintenance Platform API is running."}


def test_health_route() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "enterprise-predictive-maintenance-platform"


def test_dashboard_ui_route() -> None:
    response = client.get("/dashboard-ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_machines_route() -> None:
    response = client.get("/machines")
    assert response.status_code == 200
    payload = response.json()
    assert "machines" in payload
    assert "count" in payload
    assert payload["count"] == len(payload["machines"])
    assert payload["machines"]


def test_dashboard_route() -> None:
    response = client.get("/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["machines"] > 0
    assert "average_health_score" in payload
    assert "critically_at_risk" in payload
    assert "maintenance_recommendations" in payload


def test_model_metrics_route() -> None:
    response = client.get("/model/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "training_metrics" in payload
    assert "feature_importance" in payload
    assert payload["status"] == "trained"


def test_predict_route_success() -> None:
    payload = {
        "machine_id": "M-001",
        "sensor_data": [
            {
                "temperature": 80.0,
                "pressure": 125.0,
                "vibration": 5.9,
                "rpm": 1800.0,
                "voltage": 240.0,
                "current": 24.0,
                "humidity": 52.0,
                "load": 68.0,
                "maintenance_history": 3.0,
                "failure_log": 1,
                "operating_hours": 5400.0,
            }
        ],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["machine_id"] == "M-001"
    assert {"machine_id", "risk_score", "health_score", "remaining_useful_life_days", "failure_probability", "recommended_action", "top_risk_factors", "feature_importance"}.issubset(body.keys())


def test_predict_route_validation_error() -> None:
    response = client.post("/predict", json={"machine_id": "M-001", "sensor_data": []})
    assert response.status_code == 422


def test_explainability_route_success() -> None:
    response = client.get("/explainability/M-001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["machine_id"] == "M-001"
    assert "top_risk_factors" in payload
    assert "feature_importance" in payload
    assert "recommended_action" in payload


def test_explainability_route_not_found() -> None:
    response = client.get("/explainability/UNKNOWN-MACHINE")
    assert response.status_code == 404
    assert "Machine UNKNOWN-MACHINE not found" in response.json()["detail"]
