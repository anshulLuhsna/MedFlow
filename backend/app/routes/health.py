"""
Health Check Endpoints
System health and ML model status checks
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from ..models import HealthResponse, MLHealthResponse
from ..database import get_ml_core
from ..config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint

    Returns:
        API status and timestamp
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.app_version
    )


@router.get("/health/ml", response_model=MLHealthResponse)
async def ml_health_check():
    """
    ML models health check

    Verifies that ML Core is loaded and models are available

    Returns:
        ML system status and available resource types
    """
    try:
        ml_core = get_ml_core()

        # Check available resource types
        from ml_core.config import RESOURCE_TYPES

        return MLHealthResponse(
            status="healthy",
            models_loaded=True,
            available_resources=RESOURCE_TYPES,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"ML models not available: {str(e)}"
        )
