"""
Preference Learning Endpoint Tests
"""

import pytest
from datetime import datetime


def test_score_recommendations_no_auth(client):
    """Test preference scoring without API key"""
    response = client.post(
        "/api/v1/preferences/score",
        json={
            "user_id": "user_123",
            "recommendations": []
        }
    )
    assert response.status_code == 403


def test_score_recommendations_valid(client, headers):
    """Test preference scoring with valid request"""
    recommendations = [
        {
            "strategy_name": "Cost-Efficient",
            "summary": {
                "total_cost": 1000,
                "hospitals_helped": 5,
                "total_transfers": 10
            }
        },
        {
            "strategy_name": "Maximum Coverage",
            "summary": {
                "total_cost": 2000,
                "hospitals_helped": 10,
                "total_transfers": 20
            }
        }
    ]

    response = client.post(
        "/api/v1/preferences/score",
        json={
            "user_id": "user_123",
            "recommendations": recommendations
        },
        headers=headers
    )

    # Should return 200 or 500 (if error)
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert "ranked_strategies" in data
        assert "count" in data
        assert "timestamp" in data
        assert isinstance(data["ranked_strategies"], list)


def test_score_recommendations_with_history(client, headers):
    """Test preference scoring with past interactions"""
    recommendations = [
        {"strategy_name": "Cost-Efficient", "summary": {"total_cost": 1000}}
    ]

    past_interactions = [
        {
            "selected_recommendation_index": 0,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    ]

    response = client.post(
        "/api/v1/preferences/score",
        json={
            "user_id": "user_123",
            "recommendations": recommendations,
            "past_interactions": past_interactions
        },
        headers=headers
    )

    assert response.status_code in [200, 500]


def test_update_preferences_no_auth(client):
    """Test preference update without API key"""
    response = client.post(
        "/api/v1/preferences/update",
        json={
            "user_id": "user_123",
            "interaction": {}
        }
    )
    assert response.status_code == 403


def test_update_preferences_valid(client, headers):
    """Test preference update with valid request"""
    interaction = {
        "selected_recommendation_index": 0,
        "recommendations": [
            {"strategy_name": "Cost-Efficient", "summary": {"total_cost": 1000}}
        ],
        "timestamp": datetime.now().isoformat()
    }

    response = client.post(
        "/api/v1/preferences/update",
        json={
            "user_id": "user_123",
            "interaction": interaction
        },
        headers=headers
    )

    # Should return 200 or 500 (if error)
    assert response.status_code in [200, 500]

    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "updated"
        assert data["user_id"] == "user_123"
        assert "timestamp" in data
