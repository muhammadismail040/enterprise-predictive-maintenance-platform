from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_domain_endpoints_return_items() -> None:
    for path in ["/api/v1/assets", "/api/v1/predictions", "/api/v1/schedules", "/api/v1/alerts"]:
        response = client.get(path)
        assert response.status_code == 200
        payload = response.json()
        assert "items" in payload
        assert isinstance(payload["items"], list)
