"""Unit and integration tests for FastAPI service in app/api.py."""

import pytest
from fastapi.testclient import TestClient
from app.api import app


@pytest.fixture(scope="module")
def client():
    """Create TestClient with lifespan triggers."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "operating_threshold" in data


def test_predict_legitimate_transaction(client):
    """Test POST /predict on typical legitimate transaction."""
    payload = {
        "TransactionID": 1000001,
        "TransactionDT": 100000,
        "TransactionAmt": 45.0,
        "ProductCD": "W",
        "card1": 10000,
        "card2": 111.0,
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == 1000001
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["decision"] in ["APPROVE", "CHALLENGE", "DECLINE"]
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["top_contributing_features"], list)


def test_predict_high_risk_anomaly(client):
    """Test POST /predict on anomalous high-risk transaction."""
    high_risk_payload = {
        "TransactionID": 9999999,
        "TransactionDT": 15000000,
        "TransactionAmt": 4500.0,
        "ProductCD": "C",
        "card1": 99999,
        "card4": "visa",
        "card6": "credit",
        "P_emaildomain": "anonymous.com",
        "R_emaildomain": "protonmail.com",
        "C1": 50.0,
        "C13": 120.0,
    }
    response = client.post("/predict", json=high_risk_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == 9999999
    assert 0.0 <= data["fraud_probability"] <= 1.0
    # Contributing features should be populated with explainability reasons
    assert len(data["top_contributing_features"]) <= 3
    if data["top_contributing_features"]:
        feat = data["top_contributing_features"][0]
        assert "feature" in feat
        assert "impact_score" in feat
        assert "description" in feat


def test_predict_validation_error(client):
    """Test POST /predict with invalid data (e.g. TransactionAmt <= 0) returns 422."""
    invalid_payload = {
        "TransactionID": 123,
        "TransactionAmt": -50.0,  # Invalid: gt=0
        "ProductCD": "W",
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_minimal_payload(client):
    """Test POST /predict with minimal payload succeeds and handles missing fields."""
    minimal_payload = {
        "TransactionAmt": 100.0,
        "ProductCD": "W",
    }
    response = client.post("/predict", json=minimal_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] in ["APPROVE", "CHALLENGE", "DECLINE"]
