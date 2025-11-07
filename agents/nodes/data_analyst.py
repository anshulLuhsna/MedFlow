"""Data Analyst Agent Node"""

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
    # Fallback if langsmith not available
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Get project name from env
_langsmith_project = os.getenv("LANGSMITH_PROJECT", "medflow")


@traceable(name="data_analyst_node", project_name=_langsmith_project)
def data_analyst_node(state: MedFlowState) -> Dict:
    """
    Data Analyst Agent - Assess current resource situation.

    Responsibilities:
    1. Detect hospitals with shortage risks
    2. Identify active outbreak events
    3. Generate situational summary
    """
    logger.info(f"[Data Analyst] Analyzing shortages for resource: {state.get('resource_type', 'all')}")

    # Initialize API client
    api_client = MedFlowAPIClient()

    # Get outbreak context if provided
    outbreak_id = state.get("outbreak_id")
    hospital_ids = state.get("hospital_ids")
    affected_regions = None
    
    # If outbreak provided, get affected regions and hospitals
    if outbreak_id:
        try:
            outbreak = api_client.get_outbreak(outbreak_id)
            outbreak_data = outbreak.get("outbreak", {})
            affected_regions_str = outbreak_data.get("affected_region", "")
            
            if affected_regions_str:
                # Parse regions (comma-separated)
                affected_regions = [r.strip() for r in affected_regions_str.split(",") if r.strip()]
                logger.info(f"[Data Analyst] Outbreak affects regions: {affected_regions}")
                
                # Get hospitals from affected regions (will be used in optimization)
                # Use demo limit for performance, backend will filter by regions
                hospital_ids = None  # Let backend handle region filtering
                limit = 1 if hospital_ids else AgentConfig.get_hospital_limit()
            else:
                limit = 1 if hospital_ids else AgentConfig.get_hospital_limit()
        except Exception as e:
            logger.warning(f"[Data Analyst] Failed to fetch outbreak {outbreak_id}: {e}")
            limit = 1 if hospital_ids else AgentConfig.get_hospital_limit()
    else:
        limit = 1 if hospital_ids else AgentConfig.get_hospital_limit()
    
    logger.info(f"[Data Analyst] Using hospital limit: {limit} (from DEMO_HOSPITAL_LIMIT={os.getenv('DEMO_HOSPITAL_LIMIT', '5')})")
    
    import time
    from datetime import datetime
    start_time = time.time()
    simulation_date = state.get("simulation_date")
    
    # AGGRESSIVE DEBUG: Log with error level to ensure visibility
    logger.error(f"[Data Analyst] ⚠️ STATE DEBUG:")
    logger.error(f"[Data Analyst]   simulation_date from state = {simulation_date} (type: {type(simulation_date)})")
    logger.error(f"[Data Analyst]   State keys: {list(state.keys())}")
    print(f"[Data Analyst ERROR] simulation_date = {simulation_date}")
    
    # FALLBACK: If None, use current date and log error
    if simulation_date is None:
        logger.error(f"[Data Analyst] ❌ CRITICAL: simulation_date is None! Using current date as fallback.")
        simulation_date = datetime.now().strftime("%Y-%m-%d")
        logger.error(f"[Data Analyst] ✅ Using fallback simulation_date: {simulation_date}")
    
    shortages = api_client.get_shortages(
        resource_type=state.get("resource_type"),
        limit=limit,
        hospital_ids=hospital_ids,
        simulation_date=simulation_date
    )
    elapsed = time.time() - start_time
    logger.info(f"[Data Analyst] Shortage detection completed in {elapsed:.2f}s - found {shortages['count']} hospitals")

    # Get active outbreaks (or specific outbreak if provided)
    # Ensure simulation_date is set (use fallback if still None)
    if simulation_date is None:
        from datetime import datetime
        simulation_date = datetime.now().strftime("%Y-%m-%d")
        logger.error(f"[Data Analyst] Using fallback simulation_date for outbreaks: {simulation_date}")
    
    if outbreak_id:
        try:
            outbreak = api_client.get_outbreak(outbreak_id)
            outbreaks = {"active_outbreaks": [outbreak.get("outbreak", {})]}
        except Exception as e:
            logger.warning(f"[Data Analyst] Failed to fetch outbreak: {e}")
            outbreaks = api_client.get_active_outbreaks(simulation_date=simulation_date)
    else:
        outbreaks = api_client.get_active_outbreaks(simulation_date=simulation_date)

    # Generate summary
    summary = (
        f"Analysis complete: Found {shortages['count']} hospitals with shortages. "
        f"{len(outbreaks.get('active_outbreaks', []))} active outbreaks detected."
    )

    logger.info(f"[Data Analyst] {summary}")

    # Return state updates (include affected_regions for optimization node)
    return {
        "shortage_count": shortages["count"],
        "shortage_hospitals": shortages["shortages"],
        "active_outbreaks": outbreaks.get("active_outbreaks", []),
        "affected_regions": affected_regions,  # Pass to optimization node
        "analysis_summary": summary,
        "messages": [AIMessage(content=summary)],
        "current_node": "data_analyst"
    }
