"""
Preference Learning Endpoints
User preference scoring and updates
"""

from fastapi import APIRouter, HTTPException, Security
from datetime import datetime
from ..models import (
    ScoreRequest,
    PreferenceUpdate,
    PreferenceScoreResponse,
    PreferenceUpdateResponse
)
from ..database import get_ml_core
from ..auth import verify_api_key

router = APIRouter(prefix="/api/v1/preferences", tags=["Preferences"])


@router.post("/score", response_model=PreferenceScoreResponse)
async def score_recommendations(
    request: ScoreRequest,
    api_key: str = Security(verify_api_key)
):
    """
    Rank strategies by user preferences

    Uses hybrid scoring (40% RF + 30% LLM + 30% Vector DB)

    Body:
    - **user_id**: User identifier
    - **recommendations**: List of recommendation dictionaries
    - **past_interactions**: Optional list of past interactions for personalization

    Returns:
        Recommendations ranked by preference score with explanations
    """
    try:
        ml_core = get_ml_core()
        
        # Fetch past interactions from Supabase if user_id is provided
        past_interactions = request.past_interactions
        if request.user_id and not past_interactions:
            try:
                from ..database import get_supabase
                supabase = get_supabase()
                
                # Fetch last 20 interactions for this user
                result = supabase.table("user_interactions")\
                    .select("*")\
                    .eq("session_id", request.user_id)\
                    .order("interaction_timestamp", desc=True)\
                    .limit(20)\
                    .execute()
                
                if result.data:
                    past_interactions = result.data
                    print(f"[Preferences] Retrieved {len(past_interactions)} past interactions for user {request.user_id}")
            except Exception as db_err:
                print(f"[Preferences] Warning: Could not fetch past interactions from DB: {db_err}")
                # Continue without past interactions

        ranked = ml_core.score_recommendations(
            recommendations=request.recommendations,
            user_id=request.user_id,
            past_interactions=past_interactions
        )

        return PreferenceScoreResponse(
            ranked_strategies=ranked,
            count=len(ranked),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preference scoring failed: {str(e)}"
        )


@router.post("/update", response_model=PreferenceUpdateResponse)
async def update_preferences(
    request: PreferenceUpdate,
    api_key: str = Security(verify_api_key)
):
    """
    Update user preferences after selection

    Body:
    - **user_id**: User identifier
    - **interaction**: Interaction data including:
        - selected_recommendation_index: Index of chosen recommendation
        - recommendations: List of recommendations shown
        - timestamp: ISO timestamp

    Returns:
        Update confirmation
    """
    try:
        ml_core = get_ml_core()

        ml_core.update_preferences(
            user_id=request.user_id,
            interaction=request.interaction
        )

        return PreferenceUpdateResponse(
            status="updated",
            user_id=request.user_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preference update failed: {str(e)}"
        )
