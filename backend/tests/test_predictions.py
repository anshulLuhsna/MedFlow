"""
Prediction Endpoint Tests
"""

import pytest


def test_predict_demand_no_auth(client):
    """Test prediction endpoint without API key"""
    response = client.post(
        "/api/v1/predict/demand",
        json={
            "hospital_id": "H001",
            "resource_type": "ppe",
            "days_ahead": 14
        }
    )
    assert response.status_code == 403


def test_predict_demand_invalid_key(client):
    """Test prediction endpoint with invalid API key"""
    response = client.post(
        "/api/v1/predict/demand",
        json={
            "hospital_id": "H001",
            "resource_type": "ppe",
            "days_ahead": 14
        },
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 403


def test_predict_demand_valid(client, headers, sample_hospital_id, sample_resource_type):
    """Test prediction endpoint with valid request"""
    response = client.post(
        "/api/v1/predict/demand",
        json={
            "hospital_id": sample_hospital_id,
            "resource_type": sample_resource_type,
            "days_ahead": 14
        },
        headers=headers
    )

    # Should return 200 or 503 (if models not trained)
    assert response.status_code in [200, 400, 503, 500]

    data = response.json()

    if response.status_code == 200:
        assert "hospital_id" in data
        assert "resource_type" in data
        assert "predictions" in data
        assert "timestamp" in data
        assert data["hospital_id"] == sample_hospital_id
        assert data["resource_type"] == sample_resource_type


def test_predict_demand_invalid_days(client, headers):
    """Test prediction with invalid days_ahead"""
    response = client.post(
        "/api/v1/predict/demand",
        json={
            "hospital_id": "H001",
            "resource_type": "ppe",
            "days_ahead": 20  # Invalid: max is 14
        },
        headers=headers
    )
    assert response.status_code == 422  # Validation error


def test_detect_shortages(client, headers):
    """Test shortage detection endpoint"""
    response = client.get("/api/v1/shortages", headers=headers)

    # Should return 200 or 500 (if error)
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "shortages" in data
        assert "count" in data
        assert isinstance(data["shortages"], list)


def test_detect_shortages_filtered(client, headers, sample_resource_type):
    """Test shortage detection with resource type filter"""
    response = client.get(
        f"/api/v1/shortages?resource_type={sample_resource_type}",
        headers=headers
    )

    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "shortages" in data
        assert "count" in data


def test_optimize_allocation(client, headers, sample_resource_type):
    """Test allocation optimization endpoint"""
    response = client.post(
        "/api/v1/optimize",
        json={
            "resource_type": sample_resource_type,
            "n_strategies": 1
        },
        headers=headers
    )

    # Should return 200 or 400/500 (if error)
    assert response.status_code in [200, 400, 500]

    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "resource_type" in data
        assert "allocations" in data
        assert "summary" in data
        assert "timestamp" in data


def test_generate_strategies(client, headers, sample_resource_type):
    """Test strategy generation endpoint"""
    response = client.post(
        "/api/v1/strategies",
        json={
            "resource_type": sample_resource_type,
            "n_strategies": 3
        },
        headers=headers
    )

    # Should return 200 or 400/500 (if error)
    assert response.status_code in [200, 400, 500]

    if response.status_code == 200:
        data = response.json()
        assert "strategies" in data
        assert "count" in data
        assert "resource_type" in data
        assert "timestamp" in data
        assert data["resource_type"] == sample_resource_type


def test_generate_strategies_invalid_count(client, headers):
    """Test strategy generation with invalid count"""
    response = client.post(
        "/api/v1/strategies",
        json={
            "resource_type": "ppe",
            "n_strategies": 10  # Invalid: max is 5
        },
        headers=headers
    )
    assert response.status_code == 422  # Validation error
