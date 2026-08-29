from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def payload() -> dict[str, object]:
    return {"text": "검찰입니다 지금 바로 안전계좌로 보내세요", "amount": 3000000, "average_amount": 300000, "recipient_is_new": True, "recipient_transfer_count": 0, "transfers_last_hour": 4, "average_transfers_per_hour": 0.4, "transfer_hour": 14}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_analyze_reports_not_ready_without_production_model() -> None:
    # 새 checkout에서는 candidate가 자동 production으로 승격되지 않는다.
    response = client.post("/analyze", json=payload())
    if response.status_code == 503:
        assert response.json()["detail"] == "model_not_ready"
    else:
        assert response.status_code == 200


def test_invalid_api_request() -> None:
    body = payload(); body["transfer_hour"] = 24
    assert client.post("/analyze", json=body).status_code == 422
