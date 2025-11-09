"""
MedFlow AI Crew - Main Implementation

Uses CrewAI @CrewBase pattern with structured output enforcement.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from logging.handlers import RotatingFileHandler

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent

from agents.crewai.tools.medflow_tools import (
    AnalyzeShortagesAndOutbreaksTool,
    ForecastDemandTool,
    GenerateStrategiesTool,
    ScorePreferencesTool,
    UpdatePreferencesTool,
)
from agents.crewai.models.output_models import (
    ShortageAnalysisOutput,
    DemandForecastOutput,
    StrategyGenerationOutput,
    RankedStrategiesOutput,
    RecommendationExplanation,
    UserDecisionOutput,
    PreferenceUpdateOutput,
)
from agents.crewai.config.crewai_config import Config

# Setup logging with file handler
def setup_logging():
    """Configure logging to both console and file"""
    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent.parent.parent.parent / "logs" / "crewai"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file with timestamp
    log_file = logs_dir / f"crewai_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Check if we've already set up logging (avoid duplicates)
    logger_name = 'medflow_crewai'
    if logger_name in [h.name for h in logging.root.handlers if hasattr(h, 'name')]:
        return log_file
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # More verbose in file
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    file_handler.name = logger_name  # Mark to avoid duplicates
    
    # Configure module logger (don't touch root logger to avoid conflicts)
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(logging.DEBUG)
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_file) 
               for h in module_logger.handlers):
        module_logger.addHandler(console_handler)
        module_logger.addHandler(file_handler)
        module_logger.propagate = False  # Prevent duplicate logs
    
    # Configure CrewAI's logger
    crewai_logger = logging.getLogger('crewai')
    crewai_logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_file) 
               for h in crewai_logger.handlers):
        crewai_logger.addHandler(console_handler)
        crewai_logger.addHandler(file_handler)
    
    # Configure tools logger
    tools_logger = logging.getLogger('agents.crewai.tools')
    tools_logger.setLevel(logging.DEBUG)
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_file) 
               for h in tools_logger.handlers):
        tools_logger.addHandler(console_handler)
        tools_logger.addHandler(file_handler)
    
    return log_file

# Initialize logging
_log_file = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"CrewAI logging initialized. Log file: {_log_file}")


@CrewBase
class MedFlowCrew:
    """
    MedFlow AI Resource Allocation Crew
    
    Implements complete end-to-end allocation workflow with:
    - Structured output enforcement (prevents hanging)
    - result_as_answer for first task (immediate return)
    - Low max_iter to prevent overthinking
    - Disabled memory and planning to reduce confusion
    """
    
    # Automatically collected by decorators
    agents: List[BaseAgent]
    tasks: List[Task]
    
    # YAML config paths - relative to agents/crewai/ directory
    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'
    
    @before_kickoff
    def prepare_inputs(self, inputs: Dict) -> Dict:
        """Validate and prepare inputs before crew execution"""
        # Set defaults
        if 'resource_type' not in inputs:
            inputs['resource_type'] = 'ventilators'
        if 'user_id' not in inputs:
            inputs['user_id'] = 'default_user'
        if 'simulation_date' not in inputs:
            inputs['simulation_date'] = None
        
        logger.info(f"[MedFlowCrew] Starting workflow:")
        logger.info(f"  - Resource Type: {inputs['resource_type']}")
        logger.info(f"  - User ID: {inputs['user_id']}")
        logger.info(f"  - Simulation Date: {inputs.get('simulation_date', 'Current')}")
        
        return inputs
    
    @after_kickoff
    def check_shortages_and_route(self, output: Any) -> Any:
        """
        Conditional routing: Skip allocation tasks if no shortages detected.
        
        This allows the workflow to exit early when there are no shortages,
        saving time and API calls.
        """
        if output.tasks and len(output.tasks) > 0:
            first_task = output.tasks[0]
            try:
                # Parse first task output
                if hasattr(first_task, 'output') and first_task.output:
                    if isinstance(first_task.output.raw, str):
                        data = json.loads(first_task.output.raw)
                    else:
                        data = first_task.output.raw
                    
                    shortage_count = data.get('shortage_count', 0)
                    
                    if shortage_count == 0:
                        logger.info("[MedFlowCrew] No shortages detected. Stopping workflow.")
                        # Mark remaining tasks as skipped
                        for task in output.tasks[1:]:
                            task.status = 'skipped'
                    else:
                        logger.info(f"[MedFlowCrew] {shortage_count} shortages detected. "
                                  f"Proceeding with allocation.")
            
            except Exception as e:
                logger.warning(f"[MedFlowCrew] Error checking shortages: {e}")
        
        return output
    
    # ========================================================================
    # AGENTS
    # ========================================================================
    
    @agent
    def data_analyst(self) -> Agent:
        """Data Analyst - Calls combined analysis tool"""
        return Agent(
            config=self.agents_config['data_analyst'],  # type: ignore
            tools=[AnalyzeShortagesAndOutbreaksTool()],
            max_iter=1,  # Reduced: Tool has result_as_answer=True, should return immediately
            allow_delegation=False,
            max_execution_time=90,  # Increased: API call can take 20-30s
        )
    
    @agent
    def forecasting_specialist(self) -> Agent:
        """Forecasting Specialist - Predicts demand"""
        return Agent(
            config=self.agents_config['forecasting_specialist'],  # type: ignore
            tools=[ForecastDemandTool()],
            max_iter=5,  # Multiple forecasts needed
            allow_delegation=False,
            max_execution_time=120,  # Increased: Multiple API calls (3+ forecasts)
        )
    
    @agent
    def optimization_specialist(self) -> Agent:
        """Optimization Specialist - Generates strategies"""
        return Agent(
            config=self.agents_config['optimization_specialist'],  # type: ignore
            tools=[GenerateStrategiesTool()],
            max_iter=3,
            allow_delegation=False,
            max_execution_time=120,  # Increased: Strategy generation can take longer
        )
    
    @agent
    def preference_analyst(self) -> Agent:
        """Preference Analyst - Ranks strategies"""
        return Agent(
            config=self.agents_config['preference_analyst'],  # type: ignore
            tools=[ScorePreferencesTool()],
            max_iter=2,  # Reduced: Simple tool call, should be fast
            allow_delegation=False,
            max_execution_time=120,  # Increased: Processing 3 strategies can take time
        )
    
    @agent
    def reasoning_specialist(self) -> Agent:
        """Reasoning Specialist - Explains recommendations"""
        return Agent(
            config=self.agents_config['reasoning_specialist'],  # type: ignore
            max_iter=3,  # Reduced from 5: Generate explanation, not iterate
            allow_delegation=False,
            max_execution_time=90,  # Increased: LLM explanation generation can take time
        )
    
    @agent
    def human_review_coordinator(self) -> Agent:
        """Human Review Coordinator - Facilitates HITL"""
        return Agent(
            config=self.agents_config['human_review_coordinator'],  # type: ignore
            max_iter=3,
            allow_delegation=False,
            max_execution_time=Config.MAX_EXECUTION_TIME,
        )
    
    @agent
    def feedback_specialist(self) -> Agent:
        """Feedback Specialist - Updates preferences"""
        return Agent(
            config=self.agents_config['feedback_specialist'],  # type: ignore
            tools=[UpdatePreferencesTool()],
            max_iter=5,
            allow_delegation=False,
            max_execution_time=Config.MAX_EXECUTION_TIME,
        )
    
    # ========================================================================
    # TASKS
    # ========================================================================
    
    @task
    def analyze_shortages_task(self) -> Task:
        """Task 1: Analyze shortages and outbreaks"""
        return Task(
            config=self.tasks_config['analyze_shortages_task'],  # type: ignore
            agent=self.data_analyst(),
            output_pydantic=ShortageAnalysisOutput,  # Enforce structured output
        )
    
    @task
    def forecast_demand_task(self) -> Task:
        """Task 2: Forecast demand for critical hospitals"""
        return Task(
            config=self.tasks_config['forecast_demand_task'],  # type: ignore
            agent=self.forecasting_specialist(),
            output_pydantic=DemandForecastOutput,
        )
    
    @task
    def generate_strategies_task(self) -> Task:
        """Task 3: Generate allocation strategies"""
        return Task(
            config=self.tasks_config['generate_strategies_task'],  # type: ignore
            agent=self.optimization_specialist(),
            output_pydantic=StrategyGenerationOutput,
        )
    
    @task
    def rank_strategies_task(self) -> Task:
        """Task 4: Rank strategies by preference"""
        return Task(
            config=self.tasks_config['rank_strategies_task'],  # type: ignore
            agent=self.preference_analyst(),
            output_pydantic=RankedStrategiesOutput,
        )
    
    @task
    def explain_recommendation_task(self) -> Task:
        """Task 5: Generate explanation"""
        return Task(
            config=self.tasks_config['explain_recommendation_task'],  # type: ignore
            agent=self.reasoning_specialist(),
            output_pydantic=RecommendationExplanation,
        )
    
    @task
    def review_strategies_task(self) -> Task:
        """Task 6: Human review (HITL)"""
        return Task(
            config=self.tasks_config['review_strategies_task'],  # type: ignore
            agent=self.human_review_coordinator(),
            output_pydantic=UserDecisionOutput,
            human_input=True,  # Enable human interaction
        )
    
    @task
    def update_preferences_task(self) -> Task:
        """Task 7: Update preferences"""
        return Task(
            config=self.tasks_config['update_preferences_task'],  # type: ignore
            agent=self.feedback_specialist(),
            output_pydantic=PreferenceUpdateOutput,
        )
    
    # ========================================================================
    # CREW
    # ========================================================================
    
    @crew
    def crew(self) -> Crew:
        """Assemble the MedFlow crew"""
        return Crew(
            agents=self.agents,  # Auto-collected by @agent decorator
            tasks=self.tasks,    # Auto-collected by @task decorator
            process=Process.sequential,
            verbose=Config.VERBOSE,
            memory=Config.MEMORY,  # Disabled
            planning=Config.PLANNING,  # Disabled
            max_rpm=10,  # Rate limiting
        )


# ============================================================================
# EXECUTION FUNCTIONS
# ============================================================================

def run_workflow_until_review(
    resource_type: str = "ventilators",
    user_id: str = "default_user",
    simulation_date: str = None
) -> Dict:
    """
    Run workflow up to (but not including) the human review task.
    Designed for Streamlit integration where review happens in UI.
    
    Args:
        resource_type: Type of resource to allocate
        user_id: User ID for preference learning
        simulation_date: Optional date for historical simulation
    
    Returns:
        Dict with workflow results up to review point
    """
    try:
        Config.validate()
        medflow_crew = MedFlowCrew()
        
        # Initialize crew to collect tasks and agents
        full_crew = medflow_crew.crew()
        
        # Get tasks from the crew instance
        # Task order: analyze_shortages, forecast_demand, generate_strategies, 
        # rank_strategies, explain_recommendation, review_strategies, update_preferences
        # We want tasks 0-4 (up to explain_recommendation)
        all_tasks = full_crew.tasks
        tasks_until_review = []
        
        for i, task in enumerate(all_tasks):
            # Stop before review_strategies_task (index 5)
            if i >= 5:  # review_strategies_task is the 6th task (index 5)
                break
            tasks_until_review.append(task)
        
        # Create a temporary crew with only tasks up to review
        from crewai import Crew
        review_crew = Crew(
            agents=full_crew.agents,
            tasks=tasks_until_review,
            process=Process.sequential,
            verbose=Config.VERBOSE,
            memory=Config.MEMORY,
            planning=Config.PLANNING,
            max_rpm=10,
        )
        
        inputs = {
            "resource_type": resource_type,
            "user_id": user_id,
            "simulation_date": simulation_date
        }
        
        logger.info("[CrewAI] Running workflow until review point...")
        result = review_crew.kickoff(inputs=inputs)
        
        # Extract tasks from the crew after execution
        # The tasks will have their outputs populated after kickoff
        executed_tasks = review_crew.tasks if hasattr(review_crew, 'tasks') else []
        
        return {
            "success": True,
            "result": result,
            "tasks": executed_tasks,  # Include tasks for data extraction
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def continue_workflow_after_review(
    resource_type: str = "ventilators",
    user_id: str = "default_user",
    simulation_date: str = None,
    selected_strategy_index: int = 0,
    user_feedback: Optional[str] = None,
    ranked_strategies: Optional[List[Dict]] = None
) -> Dict:
    """
    Continue workflow after user review, running update_preferences_task.
    
    Args:
        resource_type: Type of resource to allocate
        user_id: User ID for preference learning
        simulation_date: Optional date for historical simulation
        selected_strategy_index: User's selected strategy (0-2)
        user_feedback: Optional user feedback
        ranked_strategies: The ranked strategies from previous step
    
    Returns:
        Dict with final workflow results
    """
    try:
        Config.validate()
        medflow_crew = MedFlowCrew()
        
        # Initialize crew to collect tasks and agents
        full_crew = medflow_crew.crew()
        
        # Get only the update_preferences_task (last task, index 6)
        all_tasks = full_crew.tasks
        if len(all_tasks) < 7:
            raise ValueError(f"Expected 7 tasks, found {len(all_tasks)}")
        
        update_task = all_tasks[6]  # update_preferences_task is the 7th task (index 6)
        
        # Create a crew with just the update task
        from crewai import Crew
        from datetime import datetime as dt
        
        # Prepare inputs with user decision
        # Serialize complex types to JSON strings for CrewAI variable interpolation
        ranked_strategies_json = json.dumps(ranked_strategies or []) if ranked_strategies else "[]"
        
        # Build interaction dict in the format expected by the API (matching LangGraph feedback_node)
        # API expects: selected_recommendation_index, recommendations, timestamp, feedback_text, context
        interaction_dict = {
            "selected_recommendation_index": selected_strategy_index,  # API uses this name
            "recommendations": ranked_strategies or [],  # API expects this name
            "timestamp": dt.now().isoformat(),
            "feedback_text": user_feedback,  # Optional
            "context": {
                "resource_type": resource_type,
                "simulation_date": simulation_date
            }
        }
        interaction_json = json.dumps(interaction_dict)
        
        inputs = {
            "resource_type": resource_type,
            "user_id": user_id,
            "simulation_date": simulation_date,
            "selected_strategy_index": selected_strategy_index,
            "user_feedback": user_feedback or "",
            "ranked_strategies_json": ranked_strategies_json,
            "interaction_json": interaction_json,
            "decision_timestamp": dt.now().isoformat()
        }
        
        # Create minimal crew for final task
        final_crew = Crew(
            agents=full_crew.agents,
            tasks=[update_task],
            process=Process.sequential,
            verbose=Config.VERBOSE,
            memory=Config.MEMORY,
            planning=Config.PLANNING,
            max_rpm=10,
        )
        
        logger.info("[CrewAI] Continuing workflow after review...")
        result = final_crew.kickoff(inputs=inputs)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Workflow continuation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def run_allocation_workflow(
    resource_type: str = "ventilators",
    user_id: str = "default_user",
    simulation_date: str = None
) -> Dict:
    """
    Run the complete MedFlow allocation workflow.
    
    Args:
        resource_type: Type of resource to allocate
        user_id: User ID for preference learning
        simulation_date: Optional date for historical simulation
    
    Returns:
        Dict with workflow results
    """
    try:
        # Validate config
        Config.validate()
        
        # Initialize crew
        medflow_crew = MedFlowCrew()
        
        # Prepare inputs
        inputs = {
            "resource_type": resource_type,
            "user_id": user_id,
            "simulation_date": simulation_date
        }
        
        # Execute workflow
        logger.info("=" * 80)
        logger.info("STARTING MEDFLOW ALLOCATION WORKFLOW")
        logger.info("=" * 80)
        
        result = medflow_crew.crew().kickoff(inputs=inputs)
        
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETED")
        logger.info("=" * 80)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

