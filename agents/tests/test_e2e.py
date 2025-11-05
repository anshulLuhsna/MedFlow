"""
End-to-End Tests for MedFlow Workflow
Tests the complete workflow with real backend integration (mocked)
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.graph import build_medflow_graph
from agents.tests.conftest import sample_state


class TestEndToEndWorkflow:
    """End-to-end workflow tests"""

    @pytest.mark.integration
    @patch('agents.nodes.feedback.MedFlowAPIClient')
    @patch('agents.nodes.reasoning.ChatGroq')
    @patch('agents.nodes.preference.MedFlowAPIClient')
    @patch('agents.nodes.optimization.MedFlowAPIClient')
    @patch('agents.nodes.forecasting.MedFlowAPIClient')
    @patch('agents.nodes.data_analyst.MedFlowAPIClient')
    @patch('builtins.input')
    def test_full_workflow_with_all_nodes(
        self,
        mock_input,
        mock_data_client,
        mock_forecast_client,
        mock_opt_client,
        mock_pref_client,
        mock_llm,
        mock_feedback_client,
        sample_state,
        mock_shortages_response,
        mock_outbreaks_response,
        mock_demand_forecast_response,
        mock_strategies_response,
        mock_preferences_response,
        mock_preferences_update_response
    ):
        """Test complete end-to-end workflow with all nodes"""
        
        mock_input.side_effect = ["0", "Excellent recommendation"]
        
        # Data Analyst
        data_client = MagicMock()
        data_client.get_shortages.return_value = mock_shortages_response
        data_client.get_active_outbreaks.return_value = mock_outbreaks_response
        mock_data_client.return_value = data_client
        
        # Forecasting
        forecast_client = MagicMock()
        forecast_client.predict_demand.return_value = mock_demand_forecast_response
        mock_forecast_client.return_value = forecast_client
        
        # Optimization
        opt_client = MagicMock()
        opt_client.generate_strategies.return_value = mock_strategies_response
        mock_opt_client.return_value = opt_client
        
        # Preference
        pref_client = MagicMock()
        pref_client.rank_strategies.return_value = mock_preferences_response
        mock_pref_client.return_value = pref_client
        
        # Reasoning
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "This recommendation maximizes coverage by helping 2 hospitals "
            "with a 90% shortage reduction at a cost of $1,100. "
            "It aligns with your coverage-focused preference profile."
        )
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        # Feedback
        feedback_client = MagicMock()
        feedback_client.update_preferences.return_value = mock_preferences_update_response
        mock_feedback_client.return_value = feedback_client
        
        # Build and run workflow
        graph = build_medflow_graph()
        
        state = sample_state.copy()
        state["shortage_hospitals"] = mock_shortages_response["shortages"][:3]
        
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": state["user_id"]}}
        )
        
        # Verify all nodes executed
        assert result["shortage_count"] > 0
        assert len(result["demand_forecasts"]) > 0
        assert len(result["allocation_strategies"]) > 0
        assert len(result["ranked_strategies"]) > 0
        assert result["final_recommendation"] is not None
        assert len(result["explanation"]) > 0
        assert result["user_decision"] is not None
        assert result["feedback_stored"] is True
        
        # Verify API calls
        data_client.get_shortages.assert_called_once()
        data_client.get_active_outbreaks.assert_called_once()
        assert forecast_client.predict_demand.call_count > 0
        opt_client.generate_strategies.assert_called_once()
        pref_client.rank_strategies.assert_called_once()
        mock_llm_instance.invoke.assert_called_once()
        feedback_client.update_preferences.assert_called_once()

    @pytest.mark.integration
    @patch('agents.nodes.data_analyst.MedFlowAPIClient')
    def test_workflow_early_termination_no_shortages(
        self,
        mock_data_client,
        sample_state,
        mock_outbreaks_response
    ):
        """Test workflow terminates early when no shortages"""
        
        data_client = MagicMock()
        data_client.get_shortages.return_value = {
            "count": 0,
            "shortages": [],
            "summary": {}
        }
        data_client.get_active_outbreaks.return_value = mock_outbreaks_response
        mock_data_client.return_value = data_client
        
        graph = build_medflow_graph()
        
        result = graph.invoke(
            sample_state,
            config={"configurable": {"thread_id": sample_state["user_id"]}}
        )
        
        # Should end after data analyst
        assert result["shortage_count"] == 0
        assert len(result["demand_forecasts"]) == 0
        assert len(result["allocation_strategies"]) == 0

    @pytest.mark.integration
    @patch('agents.nodes.optimization.MedFlowAPIClient')
    @patch('agents.nodes.forecasting.MedFlowAPIClient')
    @patch('agents.nodes.data_analyst.MedFlowAPIClient')
    def test_workflow_early_termination_infeasible_strategies(
        self,
        mock_data_client,
        mock_forecast_client,
        mock_opt_client,
        sample_state,
        mock_shortages_response,
        mock_outbreaks_response,
        mock_demand_forecast_response
    ):
        """Test workflow terminates when no feasible strategies"""
        
        data_client = MagicMock()
        data_client.get_shortages.return_value = mock_shortages_response
        data_client.get_active_outbreaks.return_value = mock_outbreaks_response
        mock_data_client.return_value = data_client
        
        forecast_client = MagicMock()
        forecast_client.predict_demand.return_value = mock_demand_forecast_response
        mock_forecast_client.return_value = forecast_client
        
        opt_client = MagicMock()
        opt_client.generate_strategies.return_value = {
            "strategies": [
                {
                    "strategy_name": "Infeasible",
                    "status": "infeasible",
                    "allocations": [],
                    "summary": {}
                }
            ],
            "count": 1
        }
        mock_opt_client.return_value = opt_client
        
        graph = build_medflow_graph()
        
        state = sample_state.copy()
        state["shortage_hospitals"] = mock_shortages_response["shortages"][:3]
        
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": state["user_id"]}}
        )
        
        # Should end after optimization
        assert len(result["allocation_strategies"]) == 1
        assert len(result["ranked_strategies"]) == 0

    @pytest.mark.integration
    @patch('agents.nodes.feedback.MedFlowAPIClient')
    @patch('agents.nodes.reasoning.ChatGroq')
    @patch('agents.nodes.preference.MedFlowAPIClient')
    @patch('agents.nodes.optimization.MedFlowAPIClient')
    @patch('agents.nodes.forecasting.MedFlowAPIClient')
    @patch('agents.nodes.data_analyst.MedFlowAPIClient')
    @patch('builtins.input')
    def test_workflow_with_error_handling(
        self,
        mock_input,
        mock_data_client,
        mock_forecast_client,
        mock_opt_client,
        mock_pref_client,
        mock_llm,
        mock_feedback_client,
        sample_state,
        mock_shortages_response,
        mock_outbreaks_response,
        mock_demand_forecast_response,
        mock_strategies_response,
        mock_preferences_response
    ):
        """Test workflow handles errors gracefully"""
        
        mock_input.side_effect = ["0", ""]
        
        # Data Analyst - success
        data_client = MagicMock()
        data_client.get_shortages.return_value = mock_shortages_response
        data_client.get_active_outbreaks.return_value = mock_outbreaks_response
        mock_data_client.return_value = data_client
        
        # Forecasting - one failure, one success
        forecast_client = MagicMock()
        forecast_client.predict_demand.side_effect = [
            Exception("API Error"),
            mock_demand_forecast_response
        ]
        mock_forecast_client.return_value = forecast_client
        
        # Optimization - success
        opt_client = MagicMock()
        opt_client.generate_strategies.return_value = mock_strategies_response
        mock_opt_client.return_value = opt_client
        
        # Preference - success
        pref_client = MagicMock()
        pref_client.rank_strategies.return_value = mock_preferences_response
        mock_pref_client.return_value = pref_client
        
        # Reasoning - success
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test explanation"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        # Feedback - failure
        feedback_client = MagicMock()
        feedback_client.update_preferences.side_effect = Exception("API Error")
        mock_feedback_client.return_value = feedback_client
        
        graph = build_medflow_graph()
        
        state = sample_state.copy()
        state["shortage_hospitals"] = mock_shortages_response["shortages"][:2]
        
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": state["user_id"]}}
        )
        
        # Should handle errors gracefully
        assert len(result["demand_forecasts"]) == 1  # One succeeded
        assert result["feedback_stored"] is False  # Failed but didn't crash

