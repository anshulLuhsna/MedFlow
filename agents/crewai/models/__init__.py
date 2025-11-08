"""
Pydantic Models for CrewAI Task Outputs
"""
from agents.crewai.models.output_models import (
    ShortageAnalysisOutput,
    DemandForecastOutput,
    StrategyGenerationOutput,
    RankedStrategiesOutput,
    RecommendationExplanation,
    UserDecisionOutput,
    PreferenceUpdateOutput,
)

__all__ = [
    "ShortageAnalysisOutput",
    "DemandForecastOutput",
    "StrategyGenerationOutput",
    "RankedStrategiesOutput",
    "RecommendationExplanation",
    "UserDecisionOutput",
    "PreferenceUpdateOutput",
]
