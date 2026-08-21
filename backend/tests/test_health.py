from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200_ok():
    response = client.get("/health")

    assert response.status_code == 200


def test_health_payload_reports_service_name():
    response = client.get("/health")
    body = response.json()

    assert body["status"] == "ok"
    assert body["service"] == "prompt-security-framework"
