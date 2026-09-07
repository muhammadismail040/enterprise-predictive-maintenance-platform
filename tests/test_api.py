from fastapi.testclient import TestClient

from src.api.main import DEFAULT_DATASET, ML_ENGINE, app

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


def test_machine_sensor_defaults_route() -> None:
    response = client.get("/machines/M-001/sensor-defaults")
    assert response.status_code == 200
    payload = response.json()
    assert {"temperature", "pressure", "vibration", "rpm", "voltage", "current", "humidity", "load", "maintenance_history", "failure_log", "operating_hours"}.issubset(payload)


def test_machine_sensor_defaults_route_not_found() -> None:
    response = client.get("/machines/UNKNOWN-MACHINE/sensor-defaults")
    assert response.status_code == 404


def test_dashboard_route() -> None:
    response = client.get("/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["machines"] > 0
    assert "average_health_score" in payload
    assert "critically_at_risk" in payload
    assert "maintenance_recommendations" in payload
    assert payload["fleet_health_percent"] == payload["average_health_score"]
    assert payload["critical_assets"] == payload["critically_at_risk"]
    assert "maintenance_queue" in payload


def test_dashboard_kpis_deduplicate_and_exclude_ad_hoc_predictions() -> None:
    dataset = DEFAULT_DATASET[
        DEFAULT_DATASET["machine_id"].isin(["M-001", "M-004"])
    ]
    summary = ML_ENGINE.summarize_dashboard(
        dataset,
        saved_predictions=[
            {
                "id": 1,
                "machine_id": "M-001",
                "predicted_status": "Healthy",
                "health_score": 90.0,
                "recommendation": "Continue Monitoring",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": 2,
                "machine_id": "M-001",
                "predicted_status": "Critical",
                "health_score": 20.0,
                "recommendation": "Immediate Repair",
                "created_at": "2026-01-02T00:00:00+00:00",
            },
            {
                "id": 3,
                "machine_id": "M-004",
                "predicted_status": "Warning",
                "health_score": 40.0,
                "recommendation": "Scheduled Maintenance",
                "created_at": "2026-01-03T00:00:00+00:00",
            },
            {
                "id": 4,
                "machine_id": "manual-test",
                "predicted_status": "Healthy",
                "health_score": 100.0,
                "recommendation": "Continue Monitoring",
                "created_at": "2026-01-04T00:00:00+00:00",
            },
        ],
    )
    assert summary["machines"] == 2
    assert summary["average_health_score"] == 30.0
    assert summary["critically_at_risk"] == 1
    assert summary["maintenance_queue"] == 2


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


def test_manual_prediction_is_saved_to_history() -> None:
    payload = {
        "machine_id": "manual-test",
        "sensor_data": [
            {
                "temperature": 70,
                "pressure": 100,
                "vibration": 2,
                "rpm": 1500,
                "voltage": 230,
                "current": 45,
                "humidity": 45,
                "load": 40,
                "maintenance_history": 2,
                "failure_log": 0,
                "operating_hours": 100,
            }
        ],
        "save_to_history": True,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["history"]["machine_id"] == "manual-test"

    history = client.get("/predictions/history?machine_id=manual-test")
    assert history.status_code == 200
    assert history.json()["count"] >= 1
    assert history.json()["predictions"][0]["predicted_status"] == "Healthy"


def test_prediction_history_delete() -> None:
    payload = {
        "machine_id": "delete-test",
        "sensor_data": [{
            "temperature": 70, "pressure": 100, "vibration": 2, "rpm": 1500,
            "voltage": 230, "current": 45, "humidity": 45, "load": 40,
            "maintenance_history": 2, "failure_log": 0, "operating_hours": 100,
        }],
        "save_to_history": True,
    }
    saved = client.post("/predict", json=payload).json()["history"]
    deleted = client.delete(f"/predictions/history/{saved['id']}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "id": saved["id"]}
    assert client.get("/predictions/history?machine_id=delete-test").json()["count"] == 0


def test_prediction_history_delete_not_found() -> None:
    response = client.delete("/predictions/history/999999999")
    assert response.status_code == 404


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
