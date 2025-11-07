"""Feedback Node"""

from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from langchain_core.messages import AIMessage
from datetime import datetime
import logging
import os
import json

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
    import time
    feedback_start = time.time()
    try:
        result = api_client.update_preferences(
            user_id=state["user_id"],
            interaction=interaction
        )
        feedback_elapsed = time.time() - feedback_start
        logger.info(f"[Feedback] Preference update completed in {feedback_elapsed:.2f}s")

        feedback_stored = True
        message = "Preferences updated successfully"

        logger.info(f"[Feedback] {message}")

        # ✅ NEW CODE: Persist allocation and interaction to database
        try:
            # Get selected strategy from ranked_strategies
            ranked_strategies = state.get("ranked_strategies", [])
            user_decision = state.get("user_decision")
            
            selected_strategy = None
            if user_decision is not None and ranked_strategies:
                if 0 <= user_decision < len(ranked_strategies):
                    selected_strategy = ranked_strategies[user_decision]
                elif state.get("final_recommendation"):
                    selected_strategy = state["final_recommendation"]
            
            # If no selected strategy found, try final_recommendation
            if not selected_strategy and state.get("final_recommendation"):
                selected_strategy = state["final_recommendation"]
            
            # Extract strategy details
            if selected_strategy:
                strategy_summary = selected_strategy.get('summary', {})
                strategy_name = selected_strategy.get('strategy_name', 'Unknown')
                allocations_list = selected_strategy.get('allocations', [])
                
                # Get shortage reduction (handle both keys)
                shortage_reduction = strategy_summary.get('shortage_reduction', 
                                                         strategy_summary.get('shortage_reduction_percent', 0))
                shortage_reduction_percent = strategy_summary.get('shortage_reduction_percent', 
                                                                  strategy_summary.get('shortage_reduction', 0))
                hospitals_helped = strategy_summary.get('hospitals_helped', 0)
                total_cost = strategy_summary.get('total_cost', 0)
                
                # Create allocation record
                allocation_payload = {
                    "user_id": state.get("user_id", "unknown_user"),
                    "resource_type": state.get("resource_type", "unknown"),
                    "strategy_name": strategy_name,
                    "shortage_reduction": float(shortage_reduction),
                    "shortage_reduction_percent": float(shortage_reduction_percent) if shortage_reduction_percent else None,
                    "hospitals_helped": int(hospitals_helped),
                    "total_cost": float(total_cost),
                    "allocations": allocations_list,
                    "summary": strategy_summary
                }
                
                # Create user interaction record
                interaction_payload = {
                    "user_id": state.get("user_id", "unknown_user"),
                    "session_id": state.get("session_id", state.get("user_id", "unknown")),
                    "selected_strategy": strategy_name,
                    "feedback_text": state.get("user_feedback"),
                    "recommendations": ranked_strategies,
                    "context": {
                        "resource_type": state.get("resource_type"),
                        "shortage_count": state.get("shortage_count", 0),
                        "session_id": state.get("session_id")
                    }
                }
                
                # Persist to database
                db_start = time.time()
                api_client.create_allocation(allocation_payload)
                api_client.create_user_interaction(interaction_payload)
                db_elapsed = time.time() - db_start
                logger.info(f"[Feedback] Allocation + Interaction logged to database in {db_elapsed:.2f}s")
                
                # Optional: Create local backup
                try:
                    os.makedirs("reports/allocations", exist_ok=True)
                    backup_data = {
                        "allocation": allocation_payload,
                        "interaction": interaction_payload,
                        "timestamp": datetime.now().isoformat()
                    }
                    backup_path = f"reports/allocations/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(backup_path, "w") as f:
                        json.dump(backup_data, f, indent=2)
                    logger.debug(f"[Feedback] Backup saved to {backup_path}")
                except Exception as backup_err:
                    logger.warning(f"[Feedback] Failed to create backup: {backup_err}")
                    
            else:
                logger.warning("[Feedback] No selected strategy found - skipping database persistence")
                
        except Exception as db_err:
            # Log warning but don't fail the workflow
            logger.warning(f"[Feedback] Failed to log to DB: {db_err}")
            logger.debug(f"[Feedback] DB error details: {type(db_err).__name__}: {str(db_err)}")

    except Exception as e:
        logger.error(f"[Feedback] Failed to update preferences: {e}")
        feedback_stored = False
        message = f"Failed to update preferences: {str(e)}"

    return {
        "feedback_stored": feedback_stored,
        "messages": [AIMessage(content=message)],
        "current_node": "feedback"
    }
