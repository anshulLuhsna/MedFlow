"""
Prediction Endpoints
Demand forecasting, shortage detection, and optimization
"""

from fastapi import APIRouter, HTTPException, Security, Query
from datetime import datetime
from typing import Optional
from ..models import (
    DemandRequest,
    OptimizeRequest,
    PredictionResponse,
    ShortageResponse,
    OptimizationResponse,
    StrategiesResponse
)
from ..database import get_ml_core
from ..auth import verify_api_key

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


@router.post("/predict/demand", response_model=PredictionResponse)
async def predict_demand(
    request: DemandRequest,
    api_key: str = Security(verify_api_key)
):
    """
    Predict resource demand for a specific hospital

    Predicts future resource consumption using LSTM models.

    - **hospital_id**: Hospital identifier (e.g., H001)
    - **resource_type**: One of: ppe, o2_cylinders, ventilators, medications, beds
    - **days_ahead**: Forecast horizon (1-14 days)

    Returns predictions with confidence intervals.
    """
    try:
        ml_core = get_ml_core()

        prediction = ml_core.predict_demand(
            hospital_id=request.hospital_id,
            resource_type=request.resource_type,
            days_ahead=request.days_ahead
        )

        return PredictionResponse(
            hospital_id=request.hospital_id,
            resource_type=request.resource_type,
            predictions=prediction,
            timestamp=datetime.now().isoformat()
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"ML models not loaded. Please train models first: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data structure error - missing column {str(e)}. This likely means historical data is not properly formatted in the database."
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}\n\nFull error:\n{error_details}"
        )


@router.get("/shortages", response_model=ShortageResponse)
async def detect_shortages(
    resource_type: Optional[str] = Query(
        None,
        description="Optional filter by resource type"
    ),
    limit: Optional[int] = Query(
        None,
        ge=1,
        le=100,
        description="Optional limit on number of hospitals to process (max 100)"
    ),
    hospital_ids: Optional[str] = Query(
        None,
        description="Optional comma-separated list of hospital IDs to process"
    ),
    simulation_date: Optional[str] = Query(
        None,
        description="Simulation date (YYYY-MM-DD) - use historical data as 'today'"
    ),
    api_key: str = Security(verify_api_key)
):
    """
    Detect shortage risks across all hospitals

    Query params:
    - **resource_type**: Optional filter (e.g., "ppe")

    Returns:
        List of hospitals with shortage risks and summary statistics
    """
    try:
        ml_core = get_ml_core()
        
        # Log simulation date if provided  
        if simulation_date:
            print(f"[Shortage API] ✅ USING SIMULATION DATE: {simulation_date}")

        # Parse hospital_ids if provided
        hospital_ids_list = None
        if hospital_ids:
            hospital_ids_list = [h.strip() for h in hospital_ids.split(",") if h.strip()]

        shortages_df = ml_core.detect_shortages(
            resource_type=resource_type, 
            hospital_limit=limit,
            hospital_ids=hospital_ids_list,
            simulation_date=simulation_date
        )
        
        # Handle empty DataFrame case
        if shortages_df is None or shortages_df.empty:
            return ShortageResponse(
                shortages=[],
                count=0,
                summary={
                    "total_hospitals": 0,
                    "shortage_hospitals": 0,
                    "risk_levels": {}
                }
            )

        # Get summary from already-computed shortages (avoid re-computation)
        summary = ml_core.get_shortage_summary(shortages_df=shortages_df)

        return ShortageResponse(
            shortages=shortages_df.to_dict('records'),
            count=len(shortages_df),
            summary=summary
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Shortage detection failed: {str(e)}\n\nFull error:\n{error_details}"
        )


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_allocation(
    request: OptimizeRequest,
    api_key: str = Security(verify_api_key)
):
    """
    Generate optimal resource allocation strategy

    Body:
    - **resource_type**: Type of resource to allocate
    - **n_strategies**: Number of strategies to generate (1-5)
    - **shortage_hospital_ids**: Optional list of hospitals with shortages
    - **simulation_date**: Optional simulation date (YYYY-MM-DD) for historical data

    Returns:
        Optimal allocation strategy with transfers and costs
    """
    try:
        ml_core = get_ml_core()
        
        # Log simulation date if provided
        if request.simulation_date:
            print(f"[Optimize API] ✅ USING SIMULATION DATE: {request.simulation_date}")

        result = ml_core.optimize_allocation(
            resource_type=request.resource_type,
            shortage_hospital_ids=request.shortage_hospital_ids,
            hospital_limit=request.limit,
            simulation_date=request.simulation_date
        )

        return OptimizationResponse(
            status=result.get('status', 'unknown'),
            resource_type=request.resource_type,
            allocations=result.get('allocations', []),
            summary=result.get('summary', {}),
            timestamp=datetime.now().isoformat(),
            shortage_count=result.get('shortage_count'),
            surplus_count=result.get('surplus_count'),
            diagnostics=result.get('diagnostics')
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/strategies", response_model=StrategiesResponse)
async def generate_strategies(
    request: OptimizeRequest,
    api_key: str = Security(verify_api_key)
):
    """
    Generate multiple allocation strategies

    Body:
    - **resource_type**: Type of resource to allocate
    - **n_strategies**: Number of strategies to generate (1-5)
    - **simulation_date**: Optional simulation date (YYYY-MM-DD) for historical data

    Returns:
        Multiple ranked allocation strategies
    """
    try:
        ml_core = get_ml_core()
        
        # Log simulation date if provided
        if request.simulation_date:
            print(f"[Strategies API] ✅ USING SIMULATION DATE: {request.simulation_date}")

        strategies = ml_core.generate_allocation_strategies(
            resource_type=request.resource_type,
            n_strategies=request.n_strategies,
            hospital_limit=request.limit,
            hospital_ids=request.hospital_ids,
            regions=request.regions,
            simulation_date=request.simulation_date
        )

        return StrategiesResponse(
            strategies=strategies,
            count=len(strategies),
            resource_type=request.resource_type,
            timestamp=datetime.now().isoformat()
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Strategy generation failed: {str(e)}"
        )
