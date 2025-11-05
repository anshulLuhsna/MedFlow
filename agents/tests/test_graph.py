"""
Integration Tests for LangGraph Workflow
Tests the complete workflow with all nodes
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.graph import build_medflow_graph, medflow_graph
from agents.tests.conftest import (
    sample_state,
    mock_shortages_response,
    mock_outbreaks_response,
    mock_demand_forecast_response,
    mock_strategies_response,
    mock_preferences_response,
    mock_preferences_update_response
)


class TestGraphWorkflow:
    """Integration tests for the complete LangGraph workflow"""

    @patch('agents.nodes.feedback.MedFlowAPIClient')
    @patch('agents.nodes.reasoning.ChatGroq')
    @patch('agents.nodes.preference.MedFlowAPIClient')
    @patch('agents.nodes.optimization.MedFlowAPIClient')
    @patch('agents.nodes.forecasting.MedFlowAPIClient')
    @patch('agents.nodes.data_analyst.MedFlowAPIClient')
    @patch('builtins.input')
    def test_complete_workflow_success(
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
        """Test complete workflow from start to finish"""
        # Setup mocks
        mock_input.side_effect = ["0", "Good recommendation"]
        
        # Data Analyst mocks
        data_client = MagicMock()
        data_client.get_shortages.return_value = mock_shortages_response
        data_client.get_active_outbreaks.return_value = mock_outbreaks_response
        mock_data_client.return_value = data_client
        
        # Forecasting mocks
        forecast_client = MagicMock()
        forecast_client.predict_demand.return_value = mock_demand_forecast_response
        mock_forecast_client.return_value = forecast_client
        
        # Optimization mocks
        opt_client = MagicMock()
        opt_client.generate_strategies.return_value = mock_strategies_response
        mock_opt_client.return_value = opt_client
        
        # Preference mocks
        pref_client = MagicMock()
        pref_client.rank_strategies.return_value = mock_preferences_response
        mock_pref_client.return_value = pref_client
        
        # Reasoning mocks
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a test explanation."
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        # Feedback mocks
        feedback_client = MagicMock()
        feedback_client.update_preferences.return_value = mock_preferences_update_response
        mock_feedback_client.return_value = feedback_client
        
        # Build graph
        graph = build_medflow_graph()
        
        # Update state with shortage hospitals for forecasting
        state = sample_state.copy()
        state["shortage_hospitals"] = mock_shortages_response["shortages"][:3]
        
        # Run workflow
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": state["user_id"]}}
        )
        
        # Verify final state
        assert result["shortage_count"] == 3
        assert len(result["demand_forecasts"]) > 0
        assert len(result["allocation_strategies"]) == 3
        assert len(result["ranked_strategies"]) == 3
        assert result["final_recommendation"] is not None
        assert len(result["explanation"]) > 0
        assert result["user_decision"] == 0
        assert result["feedback_stored"] is True

    @patch('agents.nodes.data_analyst.MedFlowAPIClient')
    def test_workflow_ends_if_no_shortages(
        self,
        mock_data_client,
        sample_state,
        mock_outbreaks_response
    ):
        """Test workflow ends early if no shortages detected"""
        # Setup mocks
        data_client = MagicMock()
        data_client.get_shortages.return_value = {
            "count": 0,
            "shortages": [],
            "summary": {}
        }
        data_client.get_active_outbreaks.return_value = mock_outbreaks_response
        mock_data_client.return_value = data_client
        
        # Build graph
        graph = build_medflow_graph()
        
        # Run workflow
        result = graph.invoke(
            sample_state,
            config={"configurable": {"thread_id": sample_state["user_id"]}}
        )
        
        # Verify workflow ended early
        assert result["shortage_count"] == 0
        assert len(result["demand_forecasts"]) == 0
        assert len(result["allocation_strategies"]) == 0
        # Should not reach forecasting node

    @patch('agents.nodes.optimization.MedFlowAPIClient')
    @patch('agents.nodes.forecasting.MedFlowAPIClient')
    @patch('agents.nodes.data_analyst.MedFlowAPIClient')
    def test_workflow_ends_if_no_feasible_strategies(
        self,
        mock_data_client,
        mock_forecast_client,
        mock_opt_client,
        sample_state,
        mock_shortages_response,
        mock_outbreaks_response,
        mock_demand_forecast_response
    ):
        """Test workflow ends if no feasible strategies generated"""
        # Setup mocks
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
        
        # Build graph
        graph = build_medflow_graph()
        
        # Update state
        state = sample_state.copy()
        state["shortage_hospitals"] = mock_shortages_response["shortages"][:3]
        
        # Run workflow
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": state["user_id"]}}
        )
        
        # Verify workflow ended after optimization
        assert len(result["allocation_strategies"]) == 1
        assert len(result["ranked_strategies"]) == 0
        # Should not reach preference node

    @patch('agents.nodes.feedback.MedFlowAPIClient')
    @patch('agents.nodes.reasoning.ChatGroq')
    @patch('agents.nodes.preference.MedFlowAPIClient')
    @patch('agents.nodes.optimization.MedFlowAPIClient')
    @patch('agents.nodes.forecasting.MedFlowAPIClient')
    @patch('agents.nodes.data_analyst.MedFlowAPIClient')
    @patch('builtins.input')
    def test_workflow_state_persistence(
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
        """Test workflow state persistence through checkpointer"""
        # Setup all mocks
        mock_input.side_effect = ["0", ""]
        
        data_client = MagicMock()
        data_client.get_shortages.return_value = mock_shortages_response
        data_client.get_active_outbreaks.return_value = mock_outbreaks_response
        mock_data_client.return_value = data_client
        
        forecast_client = MagicMock()
        forecast_client.predict_demand.return_value = mock_demand_forecast_response
        mock_forecast_client.return_value = forecast_client
        
        opt_client = MagicMock()
        opt_client.generate_strategies.return_value = mock_strategies_response
        mock_opt_client.return_value = opt_client
        
        pref_client = MagicMock()
        pref_client.rank_strategies.return_value = mock_preferences_response
        mock_pref_client.return_value = pref_client
        
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test explanation"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        feedback_client = MagicMock()
        feedback_client.update_preferences.return_value = mock_preferences_update_response
        mock_feedback_client.return_value = feedback_client
        
        # Build graph
        graph = build_medflow_graph()
        
        # Update state
        state = sample_state.copy()
        state["shortage_hospitals"] = mock_shortages_response["shortages"][:3]
        
        # Run workflow with checkpointing
        config = {"configurable": {"thread_id": state["user_id"]}}
        result = graph.invoke(state, config=config)
        
        # Verify state was persisted
        # (In real scenario, we could resume from checkpoint)
        assert result["session_id"] == state["session_id"]
        assert result["user_id"] == state["user_id"]
        assert result["resource_type"] == state["resource_type"]


class TestRouting:
    """Tests for routing logic"""

    def test_route_after_analysis_with_shortages(self, sample_state, mock_shortages_response):
        """Test routing continues to forecasting when shortages exist"""
        from agents.utils.routing import route_after_analysis
        
        state = sample_state.copy()
        state["shortage_count"] = 3
        
        result = route_after_analysis(state)
        
        assert result == "forecasting"

    def test_route_after_analysis_no_shortages(self, sample_state):
        """Test routing ends when no shortages"""
        from agents.utils.routing import route_after_analysis
        
        state = sample_state.copy()
        state["shortage_count"] = 0
        
        result = route_after_analysis(state)
        
        assert result == "END"

    def test_route_after_optimization_with_feasible_strategies(self, sample_state, mock_strategies_response):
        """Test routing continues to preference when feasible strategies exist"""
        from agents.utils.routing import route_after_optimization
        
        state = sample_state.copy()
        state["allocation_strategies"] = mock_strategies_response["strategies"]
        
        result = route_after_optimization(state)
        
        assert result == "preference"

    def test_route_after_optimization_no_strategies(self, sample_state):
        """Test routing ends when no strategies"""
        from agents.utils.routing import route_after_optimization
        
        state = sample_state.copy()
        state["allocation_strategies"] = []
        
        result = route_after_optimization(state)
        
        assert result == "END"

    def test_route_after_optimization_infeasible_only(self, sample_state):
        """Test routing ends when only infeasible strategies"""
        from agents.utils.routing import route_after_optimization
        
        state = sample_state.copy()
        state["allocation_strategies"] = [
            {
                "strategy_name": "Infeasible",
                "status": "infeasible",
                "allocations": [],
                "summary": {}
            }
        ]
        
        result = route_after_optimization(state)
        
        assert result == "END"

    def test_route_after_human_review_with_decision(self, sample_state):
        """Test routing continues to feedback when user makes decision"""
        from agents.utils.routing import route_after_human_review
        
        state = sample_state.copy()
        state["user_decision"] = 0
        
        result = route_after_human_review(state)
        
        assert result == "feedback"

    def test_route_after_human_review_no_decision(self, sample_state):
        """Test routing ends when no user decision"""
        from agents.utils.routing import route_after_human_review
        
        state = sample_state.copy()
        state["user_decision"] = None
        
        result = route_after_human_review(state)
        
        assert result == "END"

