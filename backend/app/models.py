"""
Request/Response Models
Pydantic models for API validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================
# REQUEST MODELS
# ============================================

class DemandRequest(BaseModel):
    """Request model for demand prediction"""
    hospital_id: str = Field(..., description="Hospital UUID", example="H001")
    resource_type: str = Field(
        ...,
        description="Resource type to predict",
        example="ppe"
    )
    days_ahead: int = Field(
        14,
        ge=1,
        le=14,
        description="Forecast horizon in days"
    )


class OptimizeRequest(BaseModel):
    """Request model for allocation optimization"""
    resource_type: str = Field(..., description="Resource type", example="ppe")
    n_strategies: int = Field(
        3,
        ge=1,
        le=5,
        description="Number of strategies to generate"
    )
    shortage_hospital_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of hospitals with shortages"
    )
    hospital_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of hospital IDs to process (filters to only these hospitals)"
    )
    regions: Optional[List[str]] = Field(
        None,
        description="Optional list of regions to filter hospitals by (for outbreak scenarios)"
    )
    limit: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Optional limit on number of hospitals to process (max 100)"
    )
    simulation_date: Optional[str] = Field(
        None,
        description="Simulation date (YYYY-MM-DD) - use historical data as 'today'"
    )


class ScoreRequest(BaseModel):
    """Request model for scoring recommendations"""
    user_id: str = Field(..., description="User identifier")
    recommendations: List[Dict[str, Any]] = Field(
        ...,
        description="List of recommendations to score"
    )
    past_interactions: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional past user interactions for personalization"
    )


class PreferenceUpdate(BaseModel):
    """Request model for updating user preferences"""
    user_id: str = Field(..., description="User identifier")
    interaction: Dict[str, Any] = Field(
        ...,
        description="Interaction data including selected recommendation"
    )


# ============================================
# RESPONSE MODELS
# ============================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., example="healthy")
    timestamp: str = Field(..., example="2025-11-04T10:00:00")
    version: str = Field(..., example="1.0.0")


class MLHealthResponse(BaseModel):
    """ML models health check response"""
    status: str = Field(..., example="healthy")
    models_loaded: bool = Field(..., example=True)
    available_resources: List[str] = Field(
        ...,
        example=["ppe", "o2_cylinders", "ventilators", "medications", "beds"]
    )
    timestamp: str = Field(..., example="2025-11-04T10:00:00")


class PredictionResponse(BaseModel):
    """Demand prediction response"""
    hospital_id: str
    resource_type: str
    predictions: Dict[str, Any]
    timestamp: str


class ShortageResponse(BaseModel):
    """Shortage detection response"""
    shortages: List[Dict[str, Any]]
    count: int
    summary: Optional[Dict[str, Any]] = None


class OptimizationResponse(BaseModel):
    """Optimization result response"""
    status: str = Field(..., example="optimal")
    resource_type: str
    allocations: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str
    shortage_count: Optional[int] = Field(None, description="Number of hospitals with shortages")
    surplus_count: Optional[int] = Field(None, description="Number of hospitals with surplus")
    diagnostics: Optional[Dict[str, Any]] = Field(None, description="Diagnostic information when status is 'no_feasible_transfers'")


class StrategiesResponse(BaseModel):
    """Multiple strategies response"""
    strategies: List[Dict[str, Any]]
    count: int
    resource_type: str
    timestamp: str


class PreferenceScoreResponse(BaseModel):
    """Preference scoring response"""
    ranked_strategies: List[Dict[str, Any]]
    count: int
    timestamp: str


class PreferenceUpdateResponse(BaseModel):
    """Preference update response"""
    status: str = Field(..., example="updated")
    user_id: str
    timestamp: str


class HospitalResponse(BaseModel):
    """Hospital data response"""
    hospitals: List[Dict[str, Any]]
    count: int


class HospitalDetailResponse(BaseModel):
    """Single hospital detail response"""
    hospital: Dict[str, Any]


# ============================================
# OUTBREAK/EVENT MODELS
# ============================================

class OutbreakResponse(BaseModel):
    """Single outbreak/event response"""
    outbreak: Dict[str, Any]
    timestamp: str


class OutbreaksListResponse(BaseModel):
    """List of outbreaks response"""
    outbreaks: List[Dict[str, Any]]
    count: int
    filters: Optional[Dict[str, Any]] = None
    timestamp: str


class ActiveOutbreakResponse(BaseModel):
    """Currently active outbreaks response"""
    active_outbreaks: List[Dict[str, Any]]
    count: int
    timestamp: str


class OutbreakImpactResponse(BaseModel):
    """Outbreak impact analysis response"""
    outbreak_id: str
    outbreak_name: str
    impact_period: Dict[str, str]
    affected_hospitals: List[Dict[str, Any]]
    shortages_during_event: List[Dict[str, Any]]
    demand_increase: Optional[Dict[str, Any]] = None
    timestamp: str


class InventoryResponse(BaseModel):
    """Hospital inventory response"""
    hospital_id: str
    inventory: List[Dict[str, Any]]
    count: int


class HospitalStatusResponse(BaseModel):
    """Complete hospital status response"""
    hospital_id: str
    current_inventory: List[Dict[str, Any]]
    demand_predictions: Dict[str, Any]
    shortage_risks: List[Dict[str, Any]]
    timestamp: str


class InteractionsResponse(BaseModel):
    """User interactions response"""
    user_id: str
    interactions: List[Dict[str, Any]]
    count: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    path: Optional[str] = None
    timestamp: str
