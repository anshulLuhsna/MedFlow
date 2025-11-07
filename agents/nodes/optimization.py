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
    simulation_date = state.get("simulation_date")  # For time-travel simulations
    
    # AGGRESSIVE DEBUG: Log with error level
    logger.error(f"[Optimization] ⚠️ STATE DEBUG:")
    logger.error(f"[Optimization]   simulation_date from state = {simulation_date} (type: {type(simulation_date)})")
    logger.error(f"[Optimization]   State keys: {list(state.keys())}")
    print(f"[Optimization ERROR] simulation_date = {simulation_date}")
    
    # FALLBACK: If None, use current date and log error
    if simulation_date is None:
        from datetime import datetime
        logger.error(f"[Optimization] ❌ CRITICAL: simulation_date is None! Using current date as fallback.")
        simulation_date = datetime.now().strftime("%Y-%m-%d")
        logger.error(f"[Optimization] ✅ Using fallback simulation_date: {simulation_date}")
    
    # If outbreak provided, use affected regions; otherwise use hospital_ids or limit
    if affected_regions:
        limit = AgentConfig.get_hospital_limit()  # Use demo limit for performance
        hospital_ids = None  # Use regions instead
    else:
        limit = 1 if hospital_ids else AgentConfig.get_hospital_limit()
    
    logger.info(f"[Optimization] Using hospital limit: {limit} (from DEMO_HOSPITAL_LIMIT={os.getenv('DEMO_HOSPITAL_LIMIT', '5')})")
    if simulation_date:
        logger.info(f"[Optimization] Using simulation date: {simulation_date}")
    
    # Use fewer strategies for faster demos (configurable)
    n_strategies = int(os.getenv("DEMO_N_STRATEGIES", "2"))  # Default to 2 for speed
    logger.info(f"[Optimization] Generating {n_strategies} strategies...")
    
    import time
    opt_start = time.time()
    result = api_client.generate_strategies(
        resource_type=state["resource_type"],
        n_strategies=n_strategies,
        limit=limit,
        hospital_ids=hospital_ids,
        regions=affected_regions,
        simulation_date=simulation_date
    )
    opt_elapsed = time.time() - opt_start
    logger.info(f"[Optimization] Strategy generation completed in {opt_elapsed:.2f}s")

    strategies = result["strategies"]
    summary = f"Generated {len(strategies)} allocation strategies"

    logger.info(f"[Optimization] {summary}")

    # Log key metrics
    for strategy in strategies:
        strategy_summary = strategy.get('summary', {})
        total_cost = strategy_summary.get('total_cost', 0)
        hospitals_helped = strategy_summary.get('hospitals_helped', 0)
        logger.info(
            f"  - {strategy.get('strategy_name', 'Unknown')}: "
            f"${total_cost:.0f}, "
            f"{hospitals_helped} hospitals"
        )

    return {
        "allocation_strategies": strategies,
        "strategy_count": len(strategies),
        "messages": [AIMessage(content=summary)],
        "current_node": "optimization"
    }
