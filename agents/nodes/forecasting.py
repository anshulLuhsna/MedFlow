"""Forecasting Agent Node"""

from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from agents.config import AgentConfig
from langchain_core.messages import AIMessage
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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


@traceable(name="forecasting_node", project_name=_langsmith_project)
def forecasting_node(state: MedFlowState) -> Dict:
    """
    Forecasting Agent - Predict 14-day demand for at-risk hospitals.
    """
    logger.info("[Forecasting] Generating 14-day demand predictions")

    api_client = MedFlowAPIClient()
    shortage_hospitals = state.get("shortage_hospitals", [])
    resource_type = state.get("resource_type")

    forecasts = {}

    # Generate predictions for hospitals up to DEMO_HOSPITAL_LIMIT (for consistency with optimization)
    # Parallelize predictions for faster execution
    forecast_limit = min(len(shortage_hospitals), AgentConfig.DEMO_HOSPITAL_LIMIT)
    top_hospitals = shortage_hospitals[:forecast_limit]
    logger.info(f"[Forecasting] Forecasting for {forecast_limit} hospitals (limit: {AgentConfig.DEMO_HOSPITAL_LIMIT})")
    
    def predict_single_hospital(hospital):
        """Predict demand for a single hospital"""
        hospital_id = hospital.get("hospital_id")
        try:
            prediction = api_client.predict_demand(
                hospital_id=hospital_id,
                resource_type=resource_type,
                days_ahead=14
            )
            return hospital_id, prediction
        except Exception as e:
            logger.warning(f"[Forecasting] Failed to predict for {hospital_id}: {e}")
            return hospital_id, None
    
    # Use ThreadPoolExecutor to parallelize predictions
    max_workers = min(len(top_hospitals), 10)  # Parallelize up to 10 hospitals for better performance
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(predict_single_hospital, hospital): hospital 
                   for hospital in top_hospitals}
        
        for future in as_completed(futures):
            hospital_id, prediction = future.result()
            if prediction:
                forecasts[hospital_id] = prediction
                logger.debug(f"[Forecasting] Predicted demand for {hospital_id}")

    summary = f"Generated demand forecasts for {len(forecasts)} hospitals"
    logger.info(f"[Forecasting] {summary}")

    return {
        "demand_forecasts": forecasts,
        "forecast_summary": summary,
        "messages": [AIMessage(content=summary)],
        "current_node": "forecasting"
    }
