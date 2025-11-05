"""
Health Endpoint Tests
"""

import pytest
from fastapi.testclient import TestClient


def test_health_check(client):
    """Test basic health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_ml_health_check(client):
    """Test ML models health check"""
    response = client.get("/health/ml")

    # Should return 200 if models are loaded, 503 if not
    assert response.status_code in [200, 503]

    data = response.json()

    if response.status_code == 200:
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert "available_resources" in data
        assert isinstance(data["available_resources"], list)
        assert len(data["available_resources"]) > 0
    else:
        # ML models not loaded
        assert "detail" in data
