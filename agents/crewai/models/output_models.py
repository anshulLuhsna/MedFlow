"""
Pydantic Models for Structured CrewAI Task Outputs
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional


class HospitalShortage(BaseModel):
    """Individual hospital shortage details"""
    hospital_id: str
    hospital_name: Optional[str] = None
    resource_type: str
    risk_level: str
    current_stock: float
    predicted_demand_7d: float
    days_of_supply: float


class ActiveOutbreak(BaseModel):
    """Active outbreak event"""
    id: str
    event_name: str
    severity: str
    affected_region: str
    start_date: str
    end_date: Optional[str] = None


class ShortageAnalysisOutput(BaseModel):
    """Output from shortage analysis task"""
    model_config = ConfigDict(extra='forbid')
    
    shortage_count: int = Field(description="Total hospitals with shortages")
    shortage_hospitals: List[Dict] = Field(description="List of hospitals with shortages")
    active_outbreaks: List[Dict] = Field(default_factory=list, description="Active outbreaks")
    affected_regions: List[str] = Field(default_factory=list, description="Affected regions")
    analysis_summary: str = Field(description="2-3 sentence summary")


class HospitalForecast(BaseModel):
    """Forecast for single hospital"""
    hospital_id: str
    hospital_name: str
    predicted_consumption_7d: float
    predicted_consumption_14d: float
    confidence_score: float
    forecast_details: str


class DemandForecastOutput(BaseModel):
    """Output from demand forecasting task"""
    model_config = ConfigDict(extra='forbid')
    
    forecasts: List[HospitalForecast] = Field(description="List of hospital forecasts")
    forecast_summary: str = Field(description="Summary of forecasts")


class StrategySummary(BaseModel):
    """Summary metrics for an allocation strategy"""
    hospitals_helped: float
    total_cost: float
    shortage_reduction_percent: float
    avg_transfer_cost: Optional[float] = None


class AllocationItem(BaseModel):
    """Single allocation transfer"""
    model_config = ConfigDict(extra='forbid')
    
    from_hospital_id: str
    to_hospital_id: str
    quantity: float
    distance_km: Optional[float] = None
    cost: Optional[float] = None


class AllocationStrategy(BaseModel):
    """Single allocation strategy"""
    model_config = ConfigDict(extra='forbid')
    
    strategy_name: str
    summary: StrategySummary
    allocations: List[AllocationItem]
    status: Optional[str] = None
    overall_score: Optional[float] = None


class StrategyGenerationOutput(BaseModel):
    """Output from strategy generation task"""
    model_config = ConfigDict(extra='forbid')
    
    strategies: List[AllocationStrategy] = Field(description="List of 3 strategies")
    strategy_count: int = Field(description="Number of strategies")


class RankedStrategy(BaseModel):
    """Strategy with preference ranking"""
    model_config = ConfigDict(extra='forbid')
    
    strategy_name: str
    preference_score: float
    summary: StrategySummary
    allocations: List[AllocationItem]
    status: Optional[str] = None
    overall_score: Optional[float] = None


class PreferenceProfile(BaseModel):
    """User preference profile information"""
    model_config = ConfigDict(extra='forbid')
    
    user_id: str
    preference_weights: Optional[Dict[str, float]] = None
    decision_count: Optional[int] = None
    last_updated: Optional[str] = None


class RankedStrategiesOutput(BaseModel):
    """Output from ranking task"""
    model_config = ConfigDict(extra='forbid')
    
    ranked_strategies: List[RankedStrategy] = Field(description="Strategies sorted by score")
    preference_profile: PreferenceProfile = Field(description="User preference info")


class FinalRecommendation(BaseModel):
    """Final recommendation details"""
    strategy_name: str
    strategy_index: int
    key_metrics: StrategySummary


class RecommendationExplanation(BaseModel):
    """Output from explanation task"""
    model_config = ConfigDict(extra='forbid')
    
    explanation: str = Field(description="200-300 word explanation")
    reasoning_chain: str = Field(description="Brief reasoning summary")
    final_recommendation: FinalRecommendation = Field(description="Recommended strategy details")


class UserDecisionOutput(BaseModel):
    """Output from human review task"""
    model_config = ConfigDict(extra='forbid')
    
    selected_strategy_index: int = Field(description="Index 0-2 of selected strategy")
    user_feedback: Optional[str] = Field(None, description="Optional user feedback")
    decision_timestamp: str = Field(description="ISO timestamp")


class PreferenceUpdateOutput(BaseModel):
    """Output from preference update task"""
    model_config = ConfigDict(extra='forbid')
    
    preferences_updated: bool
    allocation_stored: bool
    interaction_logged: bool
    summary: str
