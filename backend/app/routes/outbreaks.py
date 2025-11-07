"""
Outbreak Endpoints
Query outbreak and event data
"""

from fastapi import APIRouter, HTTPException, Security, Query
from datetime import datetime
from typing import Optional
from ..models import (
    OutbreakResponse,
    OutbreaksListResponse,
    ActiveOutbreakResponse,
    OutbreakImpactResponse
)
from ..database import get_ml_core
from ..auth import verify_api_key

router = APIRouter(prefix="/api/v1", tags=["Outbreaks"])


@router.get("/outbreaks", response_model=OutbreaksListResponse)
async def list_outbreaks(
    start_date: Optional[str] = Query(
        None,
        description="Filter events starting on/after this date (ISO format)"
    ),
    end_date: Optional[str] = Query(
        None,
        description="Filter events ending on/before this date (ISO format)"
    ),
    region: Optional[str] = Query(
        None,
        description="Filter by affected region"
    ),
    event_type: Optional[str] = Query(
        None,
        description="Filter by event type (outbreak, supply_disruption)"
    ),
    severity: Optional[str] = Query(
        None,
        description="Filter by severity (low, medium, high, critical)"
    ),
    limit: Optional[int] = Query(
        None,
        ge=1,
        le=100,
        description="Limit number of results (1-100)"
    ),
    api_key: str = Security(verify_api_key)
):
    """
    List all outbreaks/events with optional filters
    
    Query Parameters:
    - **start_date**: Filter events starting on/after this date
    - **end_date**: Filter events ending on/before this date
    - **region**: Filter by affected region
    - **event_type**: Filter by type (outbreak, supply_disruption)
    - **severity**: Filter by severity (low, medium, high, critical)
    - **limit**: Limit number of results (1-100)
    
    Returns:
        List of outbreaks/events matching the filters
    """
    try:
        ml_core = get_ml_core()
        
        # Fetch events with filters
        events_df = ml_core.data_loader.get_events(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
            severity=severity,
            region=region,
            active_only=False,
            limit=limit
        )
        
        # Convert to dict format
        outbreaks = events_df.to_dict('records') if not events_df.empty else []
        
        # Build filters dict for response
        filters = {}
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        if region:
            filters['region'] = region
        if event_type:
            filters['event_type'] = event_type
        if severity:
            filters['severity'] = severity
        if limit:
            filters['limit'] = limit
        
        return OutbreaksListResponse(
            outbreaks=outbreaks,
            count=len(outbreaks),
            filters=filters if filters else None,
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
            detail=f"Failed to fetch outbreaks: {str(e)}"
        )


@router.get("/outbreaks/active", response_model=ActiveOutbreakResponse)
async def get_active_outbreaks(
    region: Optional[str] = Query(
        None,
        description="Filter by affected region"
    ),
    simulation_date: Optional[str] = Query(
        None,
        description="Simulation date (YYYY-MM-DD) - check for active outbreaks as of this date"
    ),
    api_key: str = Security(verify_api_key)
):
    """
    Get currently active outbreaks/events
    
    Query Parameters:
    - **region**: Optional filter by affected region
    - **simulation_date**: Optional simulation date (YYYY-MM-DD) to check for active outbreaks
    
    Returns:
        List of currently active outbreaks (where simulation_date or current date is between start_date and end_date)
    """
    try:
        ml_core = get_ml_core()
        
        # Log simulation date if provided
        if simulation_date:
            print(f"[Outbreaks API] ✅ CHECKING ACTIVE OUTBREAKS AS OF: {simulation_date}")
        
        # Fetch active events only (as of simulation_date if provided)
        events_df = ml_core.data_loader.get_events(
            region=region,
            active_only=True,
            as_of_date=simulation_date
        )
        
        active_outbreaks = events_df.to_dict('records') if not events_df.empty else []
        
        return ActiveOutbreakResponse(
            active_outbreaks=active_outbreaks,
            count=len(active_outbreaks),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch active outbreaks: {str(e)}"
        )


@router.get("/outbreaks/{outbreak_id}", response_model=OutbreakResponse)
async def get_outbreak(
    outbreak_id: str,
    api_key: str = Security(verify_api_key)
):
    """
    Get specific outbreak/event by ID
    
    Path Parameters:
    - **outbreak_id**: UUID of the outbreak
    
    Returns:
        Outbreak/event details
    """
    try:
        ml_core = get_ml_core()
        
        # Fetch all events and filter by ID
        events_df = ml_core.data_loader.get_events()
        
        if events_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Outbreak {outbreak_id} not found (no events in database)"
            )
        
        # Check what column name is used for ID (could be 'id' or something else)
        id_column = None
        for col in ['id', 'event_id', 'outbreak_id']:
            if col in events_df.columns:
                id_column = col
                break
        
        if id_column is None:
            raise HTTPException(
                status_code=500,
                detail=f"Cannot find ID column in events. Available columns: {list(events_df.columns)}"
            )
        
        # Filter by ID
        outbreak_df = events_df[events_df[id_column] == outbreak_id]
        
        if outbreak_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Outbreak {outbreak_id} not found"
            )
        
        outbreak = outbreak_df.iloc[0].to_dict()
        
        return OutbreakResponse(
            outbreak=outbreak,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch outbreak: {str(e)}\n\nFull error:\n{error_details}"
        )


@router.get("/outbreaks/impact/{outbreak_id}", response_model=OutbreakImpactResponse)
async def get_outbreak_impact(
    outbreak_id: str,
    resource_type: Optional[str] = Query(
        None,
        description="Filter by resource type"
    ),
    api_key: str = Security(verify_api_key)
):
    """
    Get impact analysis for a specific outbreak
    
    Path Parameters:
    - **outbreak_id**: UUID of the outbreak
    
    Query Parameters:
    - **resource_type**: Optional filter by resource type
    
    Returns:
        Impact analysis including shortages during the event period
    """
    try:
        ml_core = get_ml_core()
        
        # Get outbreak details
        events_df = ml_core.data_loader.get_events()
        outbreak_df = events_df[events_df['id'] == outbreak_id]
        
        if outbreak_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Outbreak {outbreak_id} not found"
            )
        
        outbreak = outbreak_df.iloc[0]
        outbreak_name = outbreak.get('event_name', 'Unknown')
        start_date = outbreak.get('start_date')
        end_date = outbreak.get('end_date')
        affected_regions = outbreak.get('affected_region', '')
        
        # Get affected hospitals
        affected_hospital_ids = outbreak.get('affected_hospitals', [])
        if not affected_hospital_ids:
            # If no specific hospitals, get hospitals from affected regions
            hospitals_df = ml_core.data_loader.get_hospitals()
            if affected_regions:
                regions_list = [r.strip() for r in affected_regions.split(',')]
                affected_hospitals_df = hospitals_df[hospitals_df['region'].isin(regions_list)]
                affected_hospital_ids = affected_hospitals_df['hospital_id'].tolist()
            else:
                affected_hospital_ids = []
        
        # Get shortages during the event period
        shortages_during_event = []
        if start_date and end_date:
            # Get inventory history for the event period
            inventory_history = ml_core.data_loader.get_inventory_history(
                start_date=start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
                end_date=end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date),
                hospital_ids=affected_hospital_ids if affected_hospital_ids else None
            )
            
            # Filter by resource type if specified
            if resource_type and not inventory_history.empty:
                resource_types_df = ml_core.data_loader.get_resource_types()
                resource_type_id = resource_types_df[resource_types_df['name'] == resource_type]
                if not resource_type_id.empty:
                    resource_type_id = resource_type_id.iloc[0]['id']
                    inventory_history = inventory_history[inventory_history['resource_type_id'] == resource_type_id]
            
            # Calculate shortages (inventory below critical threshold)
            if not inventory_history.empty:
                # Get critical thresholds
                resource_types_df = ml_core.data_loader.get_resource_types()
                critical_thresholds = dict(zip(
                    resource_types_df['id'],
                    resource_types_df['critical_threshold']
                ))
                
                # Find shortages
                for _, row in inventory_history.iterrows():
                    resource_type_id = row.get('resource_type_id')
                    threshold = critical_thresholds.get(resource_type_id, 0)
                    
                    if row.get('quantity', 0) < threshold:
                        shortages_during_event.append({
                            'hospital_id': row.get('hospital_id'),
                            'resource_type_id': resource_type_id,
                            'quantity': row.get('quantity'),
                            'critical_threshold': threshold,
                            'date': row.get('date').isoformat() if hasattr(row.get('date'), 'isoformat') else str(row.get('date')),
                            'shortage_amount': threshold - row.get('quantity', 0)
                        })
        
        # Format impact period
        impact_period = {
            'start_date': start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
            'end_date': end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date) if end_date else None
        }
        
        # Get affected hospitals info
        affected_hospitals = []
        if affected_hospital_ids:
            hospitals_df = ml_core.data_loader.get_hospitals()
            affected_hospitals_df = hospitals_df[hospitals_df['hospital_id'].isin(affected_hospital_ids)]
            affected_hospitals = affected_hospitals_df.to_dict('records') if not affected_hospitals_df.empty else []
        
        return OutbreakImpactResponse(
            outbreak_id=outbreak_id,
            outbreak_name=outbreak_name,
            impact_period=impact_period,
            affected_hospitals=affected_hospitals,
            shortages_during_event=shortages_during_event,
            demand_increase=None,  # TODO: Calculate from admissions data during event period
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch outbreak impact: {str(e)}"
        )

