"""
Pytest Configuration
Shared fixtures and test setup
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def api_key():
    """Valid API key fixture"""
    return "dev-key-123"


@pytest.fixture
def headers(api_key):
    """Request headers with API key"""
    return {"X-API-Key": api_key}


@pytest.fixture
def sample_hospital_id():
    """Sample hospital ID for testing"""
    return "H001"


@pytest.fixture
def sample_resource_type():
    """Sample resource type for testing"""
    return "ppe"
