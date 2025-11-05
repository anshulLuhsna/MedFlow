"""
Hospital Data Endpoints
Hospital information, inventory, and status
"""

from fastapi import APIRouter, HTTPException, Security, Query, Path
from typing import Optional
from ..models import (
    HospitalResponse,
    HospitalDetailResponse,
    InventoryResponse,
    HospitalStatusResponse
)
from ..database import get_supabase, get_ml_core
from ..auth import verify_api_key

router = APIRouter(prefix="/api/v1/hospitals", tags=["Hospitals"])


@router.get("", response_model=HospitalResponse)
async def get_hospitals(
    region: Optional[str] = Query(None, description="Optional filter by region"),
    api_key: str = Security(verify_api_key)
):
    """
    Get all hospitals

    Query params:
    - **region**: Optional filter by region

    Returns:
        List of hospitals with their details
    """
    try:
        supabase = get_supabase()

        query = supabase.table('hospitals').select('*')

        if region:
            query = query.eq('region', region)

        result = query.execute()

        return HospitalResponse(
            hospitals=result.data,
            count=len(result.data)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch hospitals: {str(e)}"
        )


@router.get("/{hospital_id}", response_model=HospitalDetailResponse)
async def get_hospital(
    hospital_id: str = Path(..., description="Hospital identifier"),
    api_key: str = Security(verify_api_key)
):
    """
    Get single hospital details

    Path params:
    - **hospital_id**: Hospital identifier

    Returns:
        Hospital details
    """
    try:
        supabase = get_supabase()

        result = supabase.table('hospitals')\
            .select('*')\
            .eq('id', hospital_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(
                status_code=404,
                detail=f"Hospital {hospital_id} not found"
            )

        return HospitalDetailResponse(hospital=result.data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch hospital: {str(e)}"
        )


@router.get("/{hospital_id}/inventory", response_model=InventoryResponse)
async def get_hospital_inventory(
    hospital_id: str = Path(..., description="Hospital identifier"),
    api_key: str = Security(verify_api_key)
):
    """
    Get hospital's current inventory

    Path params:
    - **hospital_id**: Hospital identifier

    Returns:
        Current inventory for all resource types
    """
    try:
        supabase = get_supabase()

        result = supabase.table('inventory')\
            .select('*')\
            .eq('hospital_id', hospital_id)\
            .execute()

        return InventoryResponse(
            hospital_id=hospital_id,
            inventory=result.data,
            count=len(result.data)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch inventory: {str(e)}"
        )


@router.get("/{hospital_id}/status", response_model=HospitalStatusResponse)
async def get_hospital_status(
    hospital_id: str = Path(..., description="Hospital identifier"),
    api_key: str = Security(verify_api_key)
):
    """
    Get complete hospital status

    Uses ML Core to generate comprehensive status including:
    - Current inventory
    - Demand predictions for all resources
    - Shortage risk assessment

    Path params:
    - **hospital_id**: Hospital identifier

    Returns:
        Complete hospital status with predictions
    """
    try:
        ml_core = get_ml_core()

        status = ml_core.get_hospital_status(hospital_id)

        return HospitalStatusResponse(**status)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get hospital status: {str(e)}"
        )
