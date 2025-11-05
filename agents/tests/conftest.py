"""
Pytest Configuration for Agent Tests
Shared fixtures and test setup
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List
from agents.state import MedFlowState
from langchain_core.messages import AIMessage
from datetime import datetime
import uuid


@pytest.fixture
def sample_state() -> MedFlowState:
    """Create a sample MedFlowState for testing"""
    return {
        "resource_type": "ventilators",
        "user_id": "test_user",
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "workflow_status": "pending",
        "timestamp": datetime.now().isoformat(),
        "current_node": None,
        "shortage_count": 0,
        "shortage_hospitals": [],
        "active_outbreaks": [],
        "analysis_summary": "",
        "demand_forecasts": {},
        "forecast_summary": "",
        "allocation_strategies": [],
        "strategy_count": 0,
        "ranked_strategies": [],
        "preference_profile": {},
        "final_recommendation": {},
        "explanation": "",
        "reasoning_chain": "",
        "user_decision": None,
        "user_feedback": None,
        "feedback_stored": False,
        "error": None,
        "execution_time_seconds": None
    }


@pytest.fixture
def mock_shortages_response() -> Dict:
    """Mock response from shortages API"""
    return {
        "count": 3,
        "shortages": [
            {
                "hospital_id": "hospital_1",
                "hospital_name": "Test Hospital 1",
                "risk_level": "critical",
                "current_stock": 5.0,
                "predicted_shortage": 10.0,
                "days_until_critical": 2
            },
            {
                "hospital_id": "hospital_2",
                "hospital_name": "Test Hospital 2",
                "risk_level": "high",
                "current_stock": 15.0,
                "predicted_shortage": 8.0,
                "days_until_critical": 5
            },
            {
                "hospital_id": "hospital_3",
                "hospital_name": "Test Hospital 3",
                "risk_level": "medium",
                "current_stock": 25.0,
                "predicted_shortage": 5.0,
                "days_until_critical": 7
            }
        ],
        "summary": {
            "total_hospitals": 3,
            "critical_count": 1,
            "high_count": 1,
            "medium_count": 1
        }
    }


@pytest.fixture
def mock_outbreaks_response() -> Dict:
    """Mock response from outbreaks API"""
    return {
        "active_outbreaks": [
            {
                "outbreak_id": "outbreak_1",
                "outbreak_type": "COVID-19",
                "region": "North",
                "affected_hospitals": 5,
                "start_date": "2023-11-01"
            }
        ],
        "count": 1
    }


@pytest.fixture
def mock_demand_forecast_response() -> Dict:
    """Mock response from demand prediction API"""
    return {
        "predictions": {
            "point_forecast": [10.0, 11.0, 12.0, 13.0] * 4,  # 14 days
            "lower_bound": [8.0, 9.0, 10.0, 11.0] * 4,
            "upper_bound": [12.0, 13.0, 14.0, 15.0] * 4
        },
        "metadata": {
            "model_version": "v1.0",
            "mae": 1.5,
            "confidence_level": 0.8
        }
    }


@pytest.fixture
def mock_strategies_response() -> Dict:
    """Mock response from strategies API"""
    return {
        "strategies": [
            {
                "strategy_name": "Cost-Efficient",
                "allocations": [
                    {
                        "from_hospital_id": "surplus_1",
                        "to_hospital_id": "hospital_1",
                        "quantity": 10.0,
                        "distance_km": 50.0,
                        "cost": 500.0
                    }
                ],
                "summary": {
                    "total_cost": 500.0,
                    "total_transfers": 1,
                    "hospitals_helped": 1,
                    "shortage_reduction": 75.0,
                    "avg_distance": 50.0
                },
                "status": "optimal",
                "overall_score": 85.0
            },
            {
                "strategy_name": "Maximum Coverage",
                "allocations": [
                    {
                        "from_hospital_id": "surplus_1",
                        "to_hospital_id": "hospital_1",
                        "quantity": 10.0,
                        "distance_km": 50.0,
                        "cost": 500.0
                    },
                    {
                        "from_hospital_id": "surplus_2",
                        "to_hospital_id": "hospital_2",
                        "quantity": 8.0,
                        "distance_km": 75.0,
                        "cost": 600.0
                    }
                ],
                "summary": {
                    "total_cost": 1100.0,
                    "total_transfers": 2,
                    "hospitals_helped": 2,
                    "shortage_reduction": 90.0,
                    "avg_distance": 62.5
                },
                "status": "optimal",
                "overall_score": 90.0
            },
            {
                "strategy_name": "Balanced",
                "allocations": [
                    {
                        "from_hospital_id": "surplus_1",
                        "to_hospital_id": "hospital_1",
                        "quantity": 8.0,
                        "distance_km": 50.0,
                        "cost": 400.0
                    }
                ],
                "summary": {
                    "total_cost": 400.0,
                    "total_transfers": 1,
                    "hospitals_helped": 1,
                    "shortage_reduction": 70.0,
                    "avg_distance": 50.0
                },
                "status": "feasible",
                "overall_score": 80.0
            }
        ],
        "count": 3,
        "computation_time": 1.5
    }


@pytest.fixture
def mock_preferences_response() -> Dict:
    """Mock response from preferences scoring API"""
    return {
        "ranked_strategies": [
            {
                "strategy_name": "Maximum Coverage",
                "allocations": [],
                "summary": {
                    "total_cost": 1100.0,
                    "total_transfers": 2,
                    "hospitals_helped": 2,
                    "shortage_reduction": 90.0,
                    "avg_distance": 62.5
                },
                "status": "optimal",
                "overall_score": 90.0,
                "preference_score": 0.95,
                "llm_explanation": "This strategy maximizes coverage",
                "score_breakdown": {
                    "rf_score": 0.9,
                    "llm_score": 0.95,
                    "vector_score": 0.92
                }
            },
            {
                "strategy_name": "Cost-Efficient",
                "allocations": [],
                "summary": {
                    "total_cost": 500.0,
                    "total_transfers": 1,
                    "hospitals_helped": 1,
                    "shortage_reduction": 75.0,
                    "avg_distance": 50.0
                },
                "status": "optimal",
                "overall_score": 85.0,
                "preference_score": 0.85,
                "llm_explanation": "This strategy minimizes costs",
                "score_breakdown": {
                    "rf_score": 0.8,
                    "llm_score": 0.85,
                    "vector_score": 0.88
                }
            },
            {
                "strategy_name": "Balanced",
                "allocations": [],
                "summary": {
                    "total_cost": 400.0,
                    "total_transfers": 1,
                    "hospitals_helped": 1,
                    "shortage_reduction": 70.0,
                    "avg_distance": 50.0
                },
                "status": "feasible",
                "overall_score": 80.0,
                "preference_score": 0.75,
                "llm_explanation": "This strategy provides balance",
                "score_breakdown": {
                    "rf_score": 0.75,
                    "llm_score": 0.8,
                    "vector_score": 0.7
                }
            }
        ],
        "user_profile": {
            "preference_type": "coverage-focused",
            "confidence": 0.85,
            "key_patterns": ["Consistently chooses high coverage strategies"],
            "interaction_count": 5
        }
    }


@pytest.fixture
def mock_preferences_update_response() -> Dict:
    """Mock response from preferences update API"""
    return {
        "success": True,
        "message": "Preferences updated successfully"
    }


@pytest.fixture
def mock_api_client():
    """Create a mock API client with all methods"""
    client = Mock()
    
    # Mock all API methods
    client.get_shortages = Mock()
    client.get_active_outbreaks = Mock()
    client.predict_demand = Mock()
    client.generate_strategies = Mock()
    client.rank_strategies = Mock()
    client.update_preferences = Mock()
    client.close = Mock()
    
    return client

