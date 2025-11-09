"""
CrewAI Tools for MedFlow
"""
import json
import logging
from typing import Type, Optional, List, Dict
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from agents.tools.api_client import MedFlowAPIClient
from agents.crewai.config.crewai_config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# COMBINED ANALYSIS TOOL (CRITICAL - Uses result_as_answer=True)
# ============================================================================

class AnalyzeShortagesInput(BaseModel):
    """Input for combined shortage and outbreak analysis"""
    resource_type: str = Field(..., description="Resource type (ventilators, medications, beds, o2_cylinders, ppe)")
    limit: Optional[int] = Field(None, description="Optional limit on number of hospitals")
    simulation_date: Optional[str] = Field(None, description="Optional date for simulation (YYYY-MM-DD)")


class AnalyzeShortagesAndOutbreaksTool(BaseTool):
    """
    COMBINED TOOL: Gets both shortage data AND outbreak information in ONE call.
    
    CRITICAL: This tool uses result_as_answer=True to force the agent to 
    return the tool output immediately without additional processing or thinking.
    This prevents the hanging issue.
    """
    name: str = "analyze_shortages_and_outbreaks"
    description: str = (
        "Get complete shortage analysis including both hospital shortages and active outbreaks. "
        "Returns structured JSON with shortage_count, shortage_hospitals, active_outbreaks, "
        "affected_regions, and analysis_summary. USE THIS TOOL ONCE AND RETURN ITS OUTPUT."
    )
    args_schema: Type[BaseModel] = AnalyzeShortagesInput
    
    # CRITICAL: This forces the tool output to be the final answer
    result_as_answer: bool = True
    
    def _run(
        self,
        resource_type: str,
        limit: Optional[int] = None,
        simulation_date: Optional[str] = None
    ) -> str:
        """Execute combined analysis"""
        try:
            client = MedFlowAPIClient(base_url=Config.MEDFLOW_API_BASE)
            
            # Call both APIs
            shortages = client.get_shortages(
                resource_type=resource_type,
                limit=limit,
                simulation_date=simulation_date
            )
            
            outbreaks = client.get_active_outbreaks(simulation_date=simulation_date)
            
            # Combine into structured output
            result = {
                "shortage_count": shortages.get("count", 0),
                "shortage_hospitals": shortages.get("shortages", []),
                "active_outbreaks": outbreaks.get("active_outbreaks", []),
                "affected_regions": outbreaks.get("affected_regions", []),
                "analysis_summary": (
                    f"Detected {shortages.get('count', 0)} hospitals with {resource_type} shortages. "
                    f"Active outbreaks: {len(outbreaks.get('active_outbreaks', []))}."
                )
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error in AnalyzeShortagesAndOutbreaksTool: {e}")
            return json.dumps({
                "shortage_count": 0,
                "shortage_hospitals": [],
                "active_outbreaks": [],
                "affected_regions": [],
                "analysis_summary": f"Error: {str(e)}"
            }, indent=2)


# ============================================================================
# FORECASTING TOOL
# ============================================================================

class ForecastDemandInput(BaseModel):
    """Input for demand forecasting"""
    hospital_id: str = Field(..., description="Hospital ID")
    resource_type: str = Field(..., description="Resource type")
    days_ahead: int = Field(14, description="Days to forecast (default: 14)")
    simulation_date: Optional[str] = Field(None, description="Optional simulation date (YYYY-MM-DD)")


class ForecastDemandTool(BaseTool):
    """
    Forecast future demand for a hospital.
    
    Returns a single HospitalForecast object that can be aggregated into a list.
    The agent should call this tool for each hospital and collect the results.
    """
    name: str = "forecast_demand"
    description: str = (
        "Predict future resource consumption for a specific hospital. "
        "Returns a HospitalForecast object with hospital_id, hospital_name, "
        "predicted_consumption_7d, predicted_consumption_14d, confidence_score, and forecast_details. "
        "Call this tool once per hospital and collect all results into a list."
    )
    args_schema: Type[BaseModel] = ForecastDemandInput
    
    def _run(self, hospital_id: str, resource_type: str, days_ahead: int = 14, simulation_date: Optional[str] = None) -> str:
        """Execute demand forecast and transform to HospitalForecast format"""
        try:
            client = MedFlowAPIClient(base_url=Config.MEDFLOW_API_BASE)
            result = client.predict_demand(
                hospital_id=hospital_id,
                resource_type=resource_type,
                days_ahead=days_ahead,
                simulation_date=simulation_date
            )
            
            # Transform API response to HospitalForecast format
            predictions = result.get("predictions", {})
            predicted_demand = predictions.get("predicted_demand", [])
            
            # Calculate 7-day and 14-day consumption sums
            predicted_consumption_7d = sum(predicted_demand[:7]) if len(predicted_demand) >= 7 else sum(predicted_demand)
            predicted_consumption_14d = sum(predicted_demand[:14]) if len(predicted_demand) >= 14 else sum(predicted_demand)
            
            # Calculate confidence score from confidence intervals
            confidence_lower = predictions.get("confidence_lower", [])
            confidence_upper = predictions.get("confidence_upper", [])
            if confidence_lower and confidence_upper:
                # Confidence score based on interval width (narrower = higher confidence)
                avg_interval_width = sum(u - l for u, l in zip(confidence_upper[:7], confidence_lower[:7])) / min(7, len(confidence_lower))
                max_demand = max(predicted_demand[:7]) if predicted_demand else 1.0
                confidence_score = max(0.0, min(1.0, 1.0 - (avg_interval_width / max(max_demand, 1.0))))
            else:
                confidence_score = 0.8  # Default confidence
            
            # Create forecast details string
            current_stock = predictions.get("current_stock", 0.0)
            avg_daily = predictions.get("avg_daily_consumption", 0.0)
            forecast_details = (
                f"Forecast period {days_ahead} days with expected daily consumption "
                f"ranging between {min(predicted_demand[:7]):.2f} to {max(predicted_demand[:7]):.2f} {resource_type}. "
                f"Current stock: {current_stock:.1f}, Average daily consumption: {avg_daily:.2f}."
            )
            
            # Return as HospitalForecast-compatible JSON
            forecast = {
                "hospital_id": hospital_id,
                "hospital_name": "Unknown",  # API doesn't provide name, agent can look it up if needed
                "predicted_consumption_7d": predicted_consumption_7d,
                "predicted_consumption_14d": predicted_consumption_14d,
                "confidence_score": confidence_score,
                "forecast_details": forecast_details
            }
            
            return json.dumps(forecast, indent=2)
            
        except Exception as e:
            logger.error(f"Error in ForecastDemandTool: {e}")
            return json.dumps({
                "hospital_id": hospital_id,
                "hospital_name": "Unknown",
                "predicted_consumption_7d": 0.0,
                "predicted_consumption_14d": 0.0,
                "confidence_score": 0.0,
                "forecast_details": f"Error: {str(e)}"
            }, indent=2)


# ============================================================================
# STRATEGY GENERATION TOOL
# ============================================================================

class GenerateStrategiesInput(BaseModel):
    """Input for strategy generation"""
    resource_type: str = Field(..., description="Resource type")
    n_strategies: int = Field(3, description="Number of strategies (default: 3)")
    limit: Optional[int] = Field(None, description="Optional hospital limit")
    simulation_date: Optional[str] = Field(None, description="Optional simulation date")


class GenerateStrategiesTool(BaseTool):
    """
    Generate multiple allocation strategies.
    
    CRITICAL: Uses result_as_answer=True to ensure clean output for next task.
    """
    name: str = "generate_strategies"
    description: str = (
        "Generate 3 allocation strategies (Cost-Efficient, Maximum Coverage, Balanced). "
        "Returns strategies array with allocations, costs, and metrics for each strategy. "
        "CALL THIS TOOL ONCE AND RETURN ITS OUTPUT."
    )
    args_schema: Type[BaseModel] = GenerateStrategiesInput
    
    # CRITICAL: Force immediate return for clean output
    result_as_answer: bool = True
    
    def _run(
        self,
        resource_type: str,
        n_strategies: int = 3,
        limit: Optional[int] = None,
        simulation_date: Optional[str] = None
    ) -> str:
        """Execute strategy generation"""
        try:
            client = MedFlowAPIClient(base_url=Config.MEDFLOW_API_BASE)
            result = client.generate_strategies(
                resource_type=resource_type,
                n_strategies=n_strategies,
                limit=limit,
                simulation_date=simulation_date
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error in GenerateStrategiesTool: {e}")
            return json.dumps({"strategies": [], "strategy_count": 0}, indent=2)


# ============================================================================
# PREFERENCE RANKING TOOL
# ============================================================================

class ScorePreferencesInput(BaseModel):
    """Input for preference scoring"""
    user_id: str = Field(..., description="User ID")
    strategies: List[Dict] = Field(..., description="List of strategies to rank")


class ScorePreferencesTool(BaseTool):
    """
    Rank strategies by user preferences.
    
    CRITICAL: Uses result_as_answer=True to force immediate return of tool output.
    """
    name: str = "score_preferences"
    description: str = (
        "Rank allocation strategies based on user preferences. "
        "Returns ranked_strategies sorted by preference_score and preference_profile. "
        "CALL THIS TOOL ONCE AND RETURN ITS OUTPUT IMMEDIATELY."
    )
    args_schema: Type[BaseModel] = ScorePreferencesInput
    
    # CRITICAL: Force immediate return of tool output
    result_as_answer: bool = True
    
    def _run(self, user_id: str, strategies: List[Dict]) -> str:
        """Execute preference scoring"""
        try:
            client = MedFlowAPIClient(base_url=Config.MEDFLOW_API_BASE)
            result = client.rank_strategies(user_id=user_id, strategies=strategies)
            
            # Transform preference_profile to match PreferenceProfile model if needed
            if "preference_profile" in result and isinstance(result["preference_profile"], dict):
                profile = result["preference_profile"]
                # Ensure it has required fields
                if "user_id" not in profile:
                    profile["user_id"] = user_id
            
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error in ScorePreferencesTool: {e}")
            return json.dumps({"ranked_strategies": strategies, "preference_profile": {"user_id": user_id}}, indent=2)


# ============================================================================
# PREFERENCE UPDATE TOOL
# ============================================================================

class UpdatePreferencesInput(BaseModel):
    """Input for preference update"""
    user_id: str = Field(..., description="User ID")
    interaction: Dict = Field(..., description="Interaction data")


class UpdatePreferencesTool(BaseTool):
    """Update user preferences based on feedback"""
    name: str = "update_preferences"
    description: str = (
        "Update user preference model based on their decision and feedback. "
        "Returns success status and confirmation message."
    )
    args_schema: Type[BaseModel] = UpdatePreferencesInput
    
    def _run(self, user_id: str, interaction: Dict) -> str:
        """Execute preference update"""
        try:
            client = MedFlowAPIClient(base_url=Config.MEDFLOW_API_BASE)
            result = client.update_preferences(user_id=user_id, interaction=interaction)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error in UpdatePreferencesTool: {e}")
            return json.dumps({"success": False, "error": str(e)}, indent=2)
