"""
MedFlow LangGraph Workflow

Orchestrates 7 agent nodes in a conditional workflow with state persistence.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.state import MedFlowState
from agents.nodes.data_analyst import data_analyst_node
from agents.nodes.forecasting import forecasting_node
from agents.nodes.optimization import optimization_node
from agents.nodes.preference import preference_node
from agents.nodes.reasoning import reasoning_node
from agents.nodes.human_review import human_review_node
from agents.nodes.feedback import feedback_node
import logging
import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

# LangSmith tracing setup
# Note: Tracing is now handled via @traceable decorators on agent nodes
# Set LANGCHAIN_API_KEY or LANGSMITH_API_KEY in .env to enable
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_analysis(state: MedFlowState) -> str:
    """Route after data analysis based on shortage count"""
    shortage_count = state.get("shortage_count", 0)

    if shortage_count == 0:
        logger.info("[Router] No shortages detected. Ending workflow.")
        return "END"
    else:
        logger.info(f"[Router] {shortage_count} shortages detected. Proceeding to forecasting.")
        return "forecasting"


def route_after_optimization(state: MedFlowState) -> str:
    """Route after optimization based on strategy availability"""
    strategies = state.get("allocation_strategies", [])

    if not strategies:
        logger.info("[Router] No strategies generated. Ending workflow.")
        return "END"

    # Check if any strategy is optimal or feasible
    feasible = [s for s in strategies if s.get("status") in ["optimal", "feasible"]]

    if not feasible:
        logger.info("[Router] No feasible strategies. Ending workflow.")
        return "END"
    else:
        logger.info(f"[Router] {len(feasible)} feasible strategies. Proceeding to preference ranking.")
        return "preference"


def route_after_human_review(state: MedFlowState) -> str:
    """Route after human review based on decision"""
    user_decision = state.get("user_decision")

    if user_decision is not None:
        logger.info("[Router] User made selection. Updating preferences.")
        return "feedback"
    else:
        logger.info("[Router] No user decision. Ending workflow.")
        return "END"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_medflow_graph() -> StateGraph:
    """
    Build the MedFlow LangGraph workflow.

    Returns:
        Compiled StateGraph with checkpointing enabled
    """
    logger.info("Building MedFlow agent graph")

    # Initialize StateGraph with our schema
    builder = StateGraph(MedFlowState)

    # ADD NODES
    builder.add_node("data_analyst", data_analyst_node)
    builder.add_node("forecasting", forecasting_node)
    builder.add_node("optimization", optimization_node)
    builder.add_node("preference", preference_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("feedback", feedback_node)

    # ADD EDGES - Define workflow structure

    # Entry point
    builder.add_edge(START, "data_analyst")

    # Conditional: After data analysis
    builder.add_conditional_edges(
        "data_analyst",
        route_after_analysis,
        {
            "forecasting": "forecasting",
            "END": END
        }
    )

    # Linear: Forecasting → Optimization
    builder.add_edge("forecasting", "optimization")

    # Conditional: After optimization
    builder.add_conditional_edges(
        "optimization",
        route_after_optimization,
        {
            "preference": "preference",
            "END": END
        }
    )

    # Linear: Preference → Reasoning → Human Review
    builder.add_edge("preference", "reasoning")
    builder.add_edge("reasoning", "human_review")

    # Conditional: After human review
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "feedback": "feedback",
            "END": END
        }
    )

    # Linear: Feedback → END
    builder.add_edge("feedback", END)

    # COMPILE WITH CHECKPOINTING
    # Ensure checkpoints directory exists
    os.makedirs("agents/checkpoints", exist_ok=True)
    
    # Create SQLite connection
    conn = sqlite3.connect("agents/checkpoints/workflows.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    # LangSmith tracing setup
    # Tracing is handled via @traceable decorators on agent nodes
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "medflow")
    
    if langsmith_api_key:
        if LANGSMITH_AVAILABLE:
            logger.info(f"✅ LangSmith tracing enabled via @traceable decorators")
            logger.info(f"   Project: {langsmith_project}")
            logger.info(f"   All agent nodes will be traced to LangSmith")
        else:
            logger.warning("⚠️ LangSmith API key set but langsmith package not installed")
            logger.warning("   Install with: pip install langsmith")
    else:
        logger.info("ℹ️ LangSmith tracing disabled")
        logger.info("   Set LANGCHAIN_API_KEY or LANGSMITH_API_KEY in .env to enable")
    
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("MedFlow agent graph compiled successfully")

    return graph


# Create singleton graph instance
medflow_graph = build_medflow_graph()
