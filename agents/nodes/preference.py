"""Preference Learning Agent Node"""

from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
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


@traceable(name="preference_node", project_name=_langsmith_project)
def preference_node(state: MedFlowState) -> Dict:
    """
    Preference Learning Agent - Rank strategies by user preferences.

    Uses hybrid ML system (40% RF + 30% LLM + 30% Vector)
    """
    logger.info("[Preference] Ranking strategies by user preferences")

    api_client = MedFlowAPIClient()

    # Call preference scoring API
    result = api_client.rank_strategies(
        user_id=state["user_id"],
        strategies=state["allocation_strategies"]
    )

    ranked = result["ranked_strategies"]
    profile = result.get("user_profile", {})

    summary = f"Ranked strategies. Top: {ranked[0]['strategy_name']}"

    logger.info(
        f"[Preference] Top strategy: {ranked[0]['strategy_name']} "
        f"(score: {ranked[0].get('preference_score', 0):.3f})"
    )

    return {
        "ranked_strategies": ranked,
        "preference_profile": profile,
        "messages": [AIMessage(content=summary)],
        "current_node": "preference"
    }
