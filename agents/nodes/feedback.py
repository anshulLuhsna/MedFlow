"""Feedback Node"""

from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from langchain_core.messages import AIMessage
from datetime import datetime
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


@traceable(name="feedback_node", project_name=_langsmith_project)
def feedback_node(state: MedFlowState) -> Dict:
    """
    Feedback Node - Update preference learning from user decision.

    Sends interaction data to preference learning API.
    """
    logger.info("[Feedback] Updating user preferences")

    api_client = MedFlowAPIClient()

    # Build interaction object
    interaction = {
        "selected_recommendation_index": state["user_decision"],
        "recommendations": state["ranked_strategies"],
        "timestamp": datetime.now().isoformat(),
        "feedback_text": state.get("user_feedback"),
        "context": {
            "resource_type": state["resource_type"],
            "shortage_count": state["shortage_count"],
            "session_id": state["session_id"]
        }
    }

    # Call preference update API
    try:
        result = api_client.update_preferences(
            user_id=state["user_id"],
            interaction=interaction
        )

        feedback_stored = True
        message = "Preferences updated successfully"

        logger.info(f"[Feedback] {message}")

    except Exception as e:
        logger.error(f"[Feedback] Failed to update preferences: {e}")
        feedback_stored = False
        message = f"Failed to update preferences: {str(e)}"

    return {
        "feedback_stored": feedback_stored,
        "messages": [AIMessage(content=message)],
        "current_node": "feedback"
    }
