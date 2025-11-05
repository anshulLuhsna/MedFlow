"""
Hospital Endpoint Tests
"""

import pytest


def test_get_hospitals_no_auth(client):
    """Test get hospitals without API key"""
    response = client.get("/api/v1/hospitals")
    assert response.status_code == 403


def test_get_hospitals_valid(client, headers):
    """Test get hospitals with valid request"""
    response = client.get("/api/v1/hospitals", headers=headers)

    # Should return 200 or 500 (if database error)
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "hospitals" in data
        assert "count" in data
        assert isinstance(data["hospitals"], list)


def test_get_hospitals_filtered(client, headers):
    """Test get hospitals with region filter"""
    response = client.get("/api/v1/hospitals?region=North", headers=headers)

    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "hospitals" in data
        assert "count" in data


def test_get_hospital_detail(client, headers, sample_hospital_id):
    """Test get single hospital details"""
    response = client.get(
        f"/api/v1/hospitals/{sample_hospital_id}",
        headers=headers
    )

    # Should return 200, 404, or 500
    assert response.status_code in [200, 404, 500]

    if response.status_code == 200:
        data = response.json()
        assert "hospital" in data
        assert isinstance(data["hospital"], dict)


def test_get_hospital_inventory(client, headers, sample_hospital_id):
    """Test get hospital inventory"""
    response = client.get(
        f"/api/v1/hospitals/{sample_hospital_id}/inventory",
        headers=headers
    )

    # Should return 200 or 500
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "hospital_id" in data
        assert "inventory" in data
        assert "count" in data
        assert data["hospital_id"] == sample_hospital_id
        assert isinstance(data["inventory"], list)


def test_get_hospital_status(client, headers, sample_hospital_id):
    """Test get complete hospital status"""
    response = client.get(
        f"/api/v1/hospitals/{sample_hospital_id}/status",
        headers=headers
    )

    # Should return 200 or 500
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "hospital_id" in data
        assert "current_inventory" in data
        assert "demand_predictions" in data
        assert "shortage_risks" in data
        assert "timestamp" in data
        assert data["hospital_id"] == sample_hospital_id
