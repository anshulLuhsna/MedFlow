"""
CrewAI Workflow Tests

Unit, integration, and E2E tests for CrewAI implementation.
"""

import pytest
import json
from pathlib import Path
from agents.crewai import MedFlowCrew
from agents.crewai.tools.medflow_tools import (
    DetectShortagesTool,
    GetActiveOutbreaksTool,
    ForecastDemandTool,
    GenerateStrategiesTool,
    ScorePreferencesTool,
    UpdatePreferencesTool,
    CreateAllocationTool,
    CreateUserInteractionTool,
)


class TestCrewAITools:
    """Unit tests for CrewAI tools"""
    
    def test_detect_shortages_tool(self):
        """Test DetectShortagesTool"""
        tool = DetectShortagesTool()
        assert tool.name == "detect_shortages"
        assert "shortage" in tool.description.lower()
        assert tool.args_schema is not None
    
    def test_get_active_outbreaks_tool(self):
        """Test GetActiveOutbreaksTool"""
        tool = GetActiveOutbreaksTool()
        assert tool.name == "get_active_outbreaks"
        assert "outbreak" in tool.description.lower()
        assert tool.args_schema is not None
    
    def test_forecast_demand_tool(self):
        """Test ForecastDemandTool"""
        tool = ForecastDemandTool()
        assert tool.name == "forecast_demand"
        assert "forecast" in tool.description.lower() or "demand" in tool.description.lower()
        assert tool.args_schema is not None
    
    def test_generate_strategies_tool(self):
        """Test GenerateStrategiesTool"""
        tool = GenerateStrategiesTool()
        assert tool.name == "generate_strategies"
        assert "strategy" in tool.description.lower()
        assert tool.args_schema is not None
    
    def test_score_preferences_tool(self):
        """Test ScorePreferencesTool"""
        tool = ScorePreferencesTool()
        assert tool.name == "score_preferences"
        assert "preference" in tool.description.lower() or "rank" in tool.description.lower()
        assert tool.args_schema is not None
    
    def test_update_preferences_tool(self):
        """Test UpdatePreferencesTool"""
        tool = UpdatePreferencesTool()
        assert tool.name == "update_preferences"
        assert "preference" in tool.description.lower() or "update" in tool.description.lower()
        assert tool.args_schema is not None
    
    def test_create_allocation_tool(self):
        """Test CreateAllocationTool"""
        tool = CreateAllocationTool()
        assert tool.name == "create_allocation"
        assert "allocation" in tool.description.lower()
        assert tool.args_schema is not None
    
    def test_create_user_interaction_tool(self):
        """Test CreateUserInteractionTool"""
        tool = CreateUserInteractionTool()
        assert tool.name == "create_user_interaction"
        assert "interaction" in tool.description.lower() or "user" in tool.description.lower()
        assert tool.args_schema is not None


class TestYAMLConfig:
    """Test YAML configuration loading"""
    
    def test_agents_yaml_exists(self):
        """Test that agents.yaml exists"""
        agents_path = Path(__file__).parent.parent / "config" / "agents.yaml"
        assert agents_path.exists(), f"agents.yaml not found at {agents_path}"
    
    def test_tasks_yaml_exists(self):
        """Test that tasks.yaml exists"""
        tasks_path = Path(__file__).parent.parent / "config" / "tasks.yaml"
        assert tasks_path.exists(), f"tasks.yaml not found at {tasks_path}"
    
    def test_yaml_structure(self):
        """Test YAML file structure"""
        import yaml
        
        agents_path = Path(__file__).parent.parent / "config" / "agents.yaml"
        tasks_path = Path(__file__).parent.parent / "config" / "tasks.yaml"
        
        # Load and parse agents.yaml
        with open(agents_path) as f:
            agents_config = yaml.safe_load(f)
        
        assert isinstance(agents_config, dict), "agents.yaml should be a dictionary"
        assert "data_analyst" in agents_config, "agents.yaml should contain data_analyst"
        assert "role" in agents_config["data_analyst"], "Agent should have role"
        assert "goal" in agents_config["data_analyst"], "Agent should have goal"
        assert "backstory" in agents_config["data_analyst"], "Agent should have backstory"
        
        # Load and parse tasks.yaml
        with open(tasks_path) as f:
            tasks_config = yaml.safe_load(f)
        
        assert isinstance(tasks_config, dict), "tasks.yaml should be a dictionary"
        assert "analyze_shortages_task" in tasks_config, "tasks.yaml should contain analyze_shortages_task"
        assert "description" in tasks_config["analyze_shortages_task"], "Task should have description"
        assert "expected_output" in tasks_config["analyze_shortages_task"], "Task should have expected_output"
        assert "agent" in tasks_config["analyze_shortages_task"], "Task should have agent"


class TestMedFlowCrew:
    """Test MedFlowCrew class"""
    
    def test_crew_initialization(self):
        """Test that MedFlowCrew can be initialized"""
        crew_instance = MedFlowCrew()
        assert crew_instance is not None
    
    def test_crew_has_agents(self):
        """Test that crew has agents defined"""
        crew_instance = MedFlowCrew()
        crew_obj = crew_instance.crew()
        assert crew_obj.agents is not None
        assert len(crew_obj.agents) > 0, "Crew should have at least one agent"
    
    def test_crew_has_tasks(self):
        """Test that crew has tasks defined"""
        crew_instance = MedFlowCrew()
        crew_obj = crew_instance.crew()
        assert crew_obj.tasks is not None
        assert len(crew_obj.tasks) > 0, "Crew should have at least one task"
    
    def test_crew_process_type(self):
        """Test that crew uses sequential process"""
        crew_instance = MedFlowCrew()
        crew_obj = crew_instance.crew()
        from crewai import Process
        assert crew_obj.process == Process.sequential, "Crew should use sequential process"


class TestCrewAIIntegration:
    """Integration tests for agent-task pairs"""
    
    @pytest.mark.skip(reason="Requires backend to be running")
    def test_shortage_detection_integration(self):
        """Test shortage detection agent-task integration"""
        # This would require a running backend
        # For now, we skip it
        pass
    
    @pytest.mark.skip(reason="Requires backend to be running")
    def test_strategy_generation_integration(self):
        """Test strategy generation agent-task integration"""
        # This would require a running backend
        # For now, we skip it
        pass


class TestCrewAIE2E:
    """End-to-end tests matching LangGraph functionality"""
    
    @pytest.mark.skip(reason="Requires backend and full workflow execution")
    def test_full_workflow_e2e(self):
        """
        E2E test matching LangGraph E2E test.
        
        This test should:
        1. Initialize crew with test inputs
        2. Run full workflow
        3. Verify all tasks complete
        4. Verify output structure matches expected format
        """
        # This would require:
        # - Backend running
        # - Test data in database
        # - Full workflow execution
        # For now, we skip it
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

