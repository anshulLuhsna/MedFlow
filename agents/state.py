"""
MedFlow Agent State Schema

Defines the central state structure for the LangGraph workflow.
State flows through 7 agent nodes with TypedDict and reducers.
"""

from typing import TypedDict, Annotated, Optional, List, Dict, Literal
from langgraph.graph.message import add_messages


class MedFlowState(TypedDict):
    """
    Central state for MedFlow agent workflow.

    Uses TypedDict with Annotated for reducer functions following
    LangGraph 2025 best practices.

    State flows through agent nodes:
    1. Data Analyst → shortage analysis
    2. Forecasting → demand predictions
    3. Optimization → allocation strategies
    4. Preference → ranking by user preferences
    5. Reasoning → LLM explanation generation
    6. Human Review → user decision (HITL)
    7. Feedback → preference learning update
    """

    # INPUT PARAMETERS
    resource_type: str
    user_id: str
    session_id: str
    hospital_ids: Optional[List[str]]  # Optional list of hospital IDs to process
    outbreak_id: Optional[str]  # Optional outbreak ID to use for context
    simulation_date: Optional[str]  # Optional date (YYYY-MM-DD) for historical data simulation
    regions: Optional[List[str]]  # Optional list of regions to filter by
    hospital_limit: Optional[int]  # Optional limit on number of hospitals to process

    # MESSAGES
    messages: Annotated[list, add_messages]

    # NODE 1: DATA ANALYST OUTPUTS
    shortage_count: int
    shortage_hospitals: List[Dict]
    active_outbreaks: List[Dict]
    affected_regions: Optional[List[str]]  # Regions from outbreak context
    analysis_summary: str

    # NODE 2: FORECASTING OUTPUTS
    demand_forecasts: Dict[str, Dict]
    forecast_summary: str

    # NODE 3: OPTIMIZATION OUTPUTS
    allocation_strategies: List[Dict]
    strategy_count: int

    # NODE 4: PREFERENCE LEARNING OUTPUTS
    ranked_strategies: List[Dict]
    preference_profile: Dict

    # NODE 5: REASONING OUTPUTS
    final_recommendation: Dict
    explanation: str
    reasoning_chain: str

    # HUMAN-IN-THE-LOOP
    user_decision: Optional[int]
    user_feedback: Optional[str]
    feedback_stored: bool

    # WORKFLOW METADATA
    timestamp: str
    workflow_status: Literal["pending", "in_progress", "completed", "failed"]
    current_node: Optional[str]
    error: Optional[str]
    execution_time_seconds: Optional[float]
