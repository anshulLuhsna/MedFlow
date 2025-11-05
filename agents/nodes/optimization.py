"""Optimization Agent Node"""

from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from agents.config import AgentConfig
from langchain_core.messages import AIMessage
import logging
import os

# LangSmith tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)

_langsmith_project = os.getenv("LANGSMITH_PROJECT", "medflow")


@traceable(name="optimization_node", project_name=_langsmith_project)
def optimization_node(state: MedFlowState) -> Dict:
    """
    Optimization Agent - Generate allocation strategies.

    Uses Linear Programming to create 3 strategies:
    1. Cost-Efficient: Minimize transfer costs
    2. Maximum Coverage: Help as many hospitals as possible
    3. Balanced: Trade-off between cost, coverage, urgency
    """
    logger.info("[Optimization] Generating allocation strategies")

    api_client = MedFlowAPIClient()

    # Call strategy generation API
    hospital_ids = state.get("hospital_ids")
    affected_regions = state.get("affected_regions")  # From outbreak context
    
    # If outbreak provided, use affected regions; otherwise use hospital_ids or limit
    if affected_regions:
        limit = AgentConfig.DEMO_HOSPITAL_LIMIT  # Use demo limit for performance
        hospital_ids = None  # Use regions instead
    else:
        limit = 1 if hospital_ids else AgentConfig.DEMO_HOSPITAL_LIMIT
    
    # Use fewer strategies for faster demos (configurable)
    n_strategies = int(os.getenv("DEMO_N_STRATEGIES", "2"))  # Default to 2 for speed
    result = api_client.generate_strategies(
        resource_type=state["resource_type"],
        n_strategies=n_strategies,
        limit=limit,
        hospital_ids=hospital_ids,
        regions=affected_regions
    )

    strategies = result["strategies"]
    summary = f"Generated {len(strategies)} allocation strategies"

    logger.info(f"[Optimization] {summary}")

    # Log key metrics
    for strategy in strategies:
        logger.info(
            f"  - {strategy['strategy_name']}: "
            f"${strategy['summary']['total_cost']:.0f}, "
            f"{strategy['summary']['hospitals_helped']} hospitals"
        )

    return {
        "allocation_strategies": strategies,
        "strategy_count": len(strategies),
        "messages": [AIMessage(content=summary)],
        "current_node": "optimization"
    }
