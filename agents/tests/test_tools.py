"""
Tests for MedFlow API Client
"""

import pytest
import httpx
from unittest.mock import Mock, patch, MagicMock
from agents.tools.api_client import MedFlowAPIClient
from agents.tests.conftest import (
    mock_shortages_response,
    mock_outbreaks_response,
    mock_demand_forecast_response,
    mock_strategies_response,
    mock_preferences_response,
    mock_preferences_update_response
)


class TestMedFlowAPIClient:
    """Test suite for MedFlowAPIClient"""

    def test_init_default_values(self):
        """Test API client initialization with default values"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client:
            client = MedFlowAPIClient()
            
            assert client.base_url == "http://localhost:8000"
            assert client.timeout == 60.0
            mock_client.assert_called_once()

    def test_init_custom_values(self):
        """Test API client initialization with custom values"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client:
            client = MedFlowAPIClient(
                base_url="http://test:8000",
                api_key="test_key",
                timeout=30.0
            )
            
            assert client.base_url == "http://test:8000"
            assert client.api_key == "test_key"
            assert client.timeout == 30.0
            mock_client.assert_called_once()

    def test_get_shortages_success(self, mock_shortages_response):
        """Test successful shortage detection"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_shortages_response
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            result = client.get_shortages(resource_type="ventilators")
            
            assert result == mock_shortages_response
            mock_client.get.assert_called_once()
            mock_response.raise_for_status.assert_called_once()

    def test_get_shortages_with_params(self):
        """Test shortage detection with resource type filter"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"count": 0, "shortages": []}
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            client.get_shortages(resource_type="ppe", risk_level="critical")
            
            # Verify params were passed
            call_args = mock_client.get.call_args
            assert "params" in call_args.kwargs
            assert call_args.kwargs["params"]["resource_type"] == "ppe"
            assert call_args.kwargs["params"]["risk_level"] == "critical"

    def test_get_active_outbreaks_success(self, mock_outbreaks_response):
        """Test successful outbreak retrieval"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_outbreaks_response
            mock_response.raise_for_status = Mock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            result = client.get_active_outbreaks()
            
            assert result == mock_outbreaks_response
            mock_client.get.assert_called_once()

    def test_predict_demand_success(self, mock_demand_forecast_response):
        """Test successful demand prediction"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_demand_forecast_response
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            result = client.predict_demand(
                hospital_id="hospital_1",
                resource_type="ventilators",
                days_ahead=14
            )
            
            assert result == mock_demand_forecast_response
            mock_client.post.assert_called_once()
            
            # Verify payload
            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["hospital_id"] == "hospital_1"
            assert call_args.kwargs["json"]["resource_type"] == "ventilators"
            assert call_args.kwargs["json"]["days_ahead"] == 14

    def test_generate_strategies_success(self, mock_strategies_response):
        """Test successful strategy generation"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_strategies_response
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            result = client.generate_strategies(
                resource_type="ventilators",
                n_strategies=3,
                limit=50
            )
            
            assert result == mock_strategies_response
            assert len(result["strategies"]) == 3
            
            # Verify payload
            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["resource_type"] == "ventilators"
            assert call_args.kwargs["json"]["n_strategies"] == 3
            assert call_args.kwargs["json"]["limit"] == 50

    def test_rank_strategies_success(self, mock_preferences_response):
        """Test successful strategy ranking"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_preferences_response
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            strategies = [{"strategy_name": "Test"}]
            result = client.rank_strategies(
                user_id="test_user",
                strategies=strategies
            )
            
            assert result == mock_preferences_response
            assert "ranked_strategies" in result
            assert "user_profile" in result
            
            # Verify payload
            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["user_id"] == "test_user"
            assert call_args.kwargs["json"]["recommendations"] == strategies

    def test_update_preferences_success(self, mock_preferences_update_response):
        """Test successful preference update"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = mock_preferences_update_response
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            interaction = {
                "selected_recommendation_index": 0,
                "recommendations": [],
                "timestamp": "2023-11-05T10:00:00"
            }
            result = client.update_preferences(
                user_id="test_user",
                interaction=interaction
            )
            
            assert result == mock_preferences_update_response
            assert result["success"] is True
            
            # Verify payload
            call_args = mock_client.post.call_args
            assert call_args.kwargs["json"]["user_id"] == "test_user"
            assert call_args.kwargs["json"]["interaction"] == interaction

    def test_error_handling_http_error(self):
        """Test error handling for HTTP errors"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error", request=Mock(), response=mock_response
            )
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            
            with pytest.raises(httpx.HTTPStatusError):
                client.get_shortages()

    def test_close_client(self):
        """Test client close method"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            client.close()
            
            mock_client.close.assert_called_once()

    def test_retry_logic_on_timeout(self):
        """Test retry logic on timeout exceptions"""
        with patch('agents.tools.api_client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client_class.return_value = mock_client
            
            client = MedFlowAPIClient()
            
            # Should retry and eventually fail after 3 attempts
            with pytest.raises(httpx.TimeoutException):
                client.get_shortages()
            
            # Verify retry attempts (3 attempts)
            assert mock_client.get.call_count == 3

