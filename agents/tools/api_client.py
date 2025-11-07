"""
MedFlow API Client

HTTP client for MedFlow FastAPI backend with retry logic and error handling.
"""

import os
import httpx
from typing import Optional, Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

logger = logging.getLogger(__name__)


class MedFlowAPIClient:
    """
    Client for MedFlow FastAPI backend with automatic retries.

    Features:
    - Retry logic with exponential backoff
    - Timeout handling
    - Error logging
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0  # Increased to 120s for slow operations
    ):
        self.base_url = base_url or os.getenv("MEDFLOW_API_BASE", "http://localhost:8000")
        # Default to dev-key-123 if not provided (backend default)
        self.api_key = api_key or os.getenv("MEDFLOW_API_KEY", "dev-key-123")
        self.timeout = timeout

        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"X-API-Key": self.api_key}
        )

        logger.info(f"Initialized MedFlow API client: {self.base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def get_shortages(
        self, 
        resource_type: Optional[str] = None,
        limit: Optional[int] = None,
        hospital_ids: Optional[List[str]] = None,
        simulation_date: Optional[str] = None
    ) -> Dict:
        """
        Detect hospitals with shortage risks
        
        Args:
            resource_type: Optional filter by resource type
            limit: Optional limit on number of hospitals to process (for demo/testing)
            hospital_ids: Optional list of hospital IDs to process
            simulation_date: Optional date (YYYY-MM-DD) to use historical data as 'today'
        """
        params = {}
        if resource_type:
            params["resource_type"] = resource_type
        if limit:
            params["limit"] = limit
        if hospital_ids:
            params["hospital_ids"] = ",".join(hospital_ids)
        if simulation_date:
            params["simulation_date"] = simulation_date

        # AGGRESSIVE DEBUG: Log params with error level
        logger.error(f"[API Client] ⚠️ get_shortages params: {params}")
        logger.error(f"[API Client]   simulation_date in params: {params.get('simulation_date')}")
        print(f"[API Client ERROR] Final params dict: {params}")
        
        logger.debug(f"GET /api/v1/shortages with params: {params}")
        response = self.client.get("/api/v1/shortages", params=params)
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def get_active_outbreaks(self, simulation_date: Optional[str] = None) -> Dict:
        """Get active outbreak events (optionally as of a historical date)"""
        params = {}
        if simulation_date:
            params["simulation_date"] = simulation_date
        
        # AGGRESSIVE DEBUG: Log params with error level
        logger.error(f"[API Client] ⚠️ get_active_outbreaks params: {params}")
        logger.error(f"[API Client]   simulation_date in params: {params.get('simulation_date')}")
        print(f"[API Client ERROR] get_active_outbreaks params: {params}")
        
        logger.debug(f"GET /api/v1/outbreaks/active params={params}")
        response = self.client.get("/api/v1/outbreaks/active", params=params)
        response.raise_for_status()
        return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def get_outbreak(self, outbreak_id: str) -> Dict:
        """Get specific outbreak by ID"""
        logger.debug(f"GET /api/v1/outbreaks/{outbreak_id}")
        response = self.client.get(f"/api/v1/outbreaks/{outbreak_id}")
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def predict_demand(
        self,
        hospital_id: str,
        resource_type: str,
        days_ahead: int = 14
    ) -> Dict:
        """Predict future demand using LSTM"""
        payload = {
            "hospital_id": hospital_id,
            "resource_type": resource_type,
            "days_ahead": days_ahead
        }

        logger.debug(f"POST /api/v1/predict/demand for {hospital_id}")
        response = self.client.post("/api/v1/predict/demand", json=payload)
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def generate_strategies(
        self,
        resource_type: str,
        n_strategies: int = 3,
        limit: Optional[int] = None,
        hospital_ids: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        simulation_date: Optional[str] = None
    ) -> Dict:
        """Generate allocation strategies"""
        payload = {
            "resource_type": resource_type,
            "n_strategies": n_strategies
        }
        if limit:
            payload["limit"] = limit
        if hospital_ids:
            payload["hospital_ids"] = hospital_ids
        if regions:
            payload["regions"] = regions
        if simulation_date:
            payload["simulation_date"] = simulation_date

        # AGGRESSIVE DEBUG: Log payload with error level
        logger.error(f"[API Client] ⚠️ generate_strategies payload: {payload}")
        logger.error(f"[API Client]   simulation_date in payload: {payload.get('simulation_date')}")
        print(f"[API Client ERROR] generate_strategies payload: {payload}")

        logger.debug(f"POST /api/v1/strategies for {resource_type}")
        response = self.client.post("/api/v1/strategies", json=payload, timeout=120.0)
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def rank_strategies(self, user_id: str, strategies: List[Dict]) -> Dict:
        """Rank strategies by user preferences"""
        payload = {
            "user_id": user_id,
            "recommendations": strategies
        }

        logger.debug(f"POST /api/v1/preferences/score for user {user_id}")
        response = self.client.post("/api/v1/preferences/score", json=payload)
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def update_preferences(self, user_id: str, interaction: Dict) -> Dict:
        """Update preference model from user interaction"""
        payload = {
            "user_id": user_id,
            "interaction": interaction
        }

        logger.debug(f"POST /api/v1/preferences/update for user {user_id}")
        response = self.client.post("/api/v1/preferences/update", json=payload)
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def create_allocation(self, allocation_data: Dict) -> Dict:
        """
        Create allocation record in Supabase
        
        Args:
            allocation_data: Dictionary containing allocation details:
                - user_id: User identifier
                - resource_type: Resource type
                - strategy_name: Strategy name
                - shortage_reduction: Shortage reduction percentage
                - hospitals_helped: Number of hospitals helped
                - total_cost: Total transfer cost
                - allocations: List of allocation transfers
                - summary: Optional strategy summary
        """
        logger.debug(f"POST /api/v1/allocations")
        response = self.client.post("/api/v1/allocations", json=allocation_data)
        response.raise_for_status()
        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def create_user_interaction(self, interaction_data: Dict) -> Dict:
        """
        Create user interaction record in Supabase
        
        Args:
            interaction_data: Dictionary containing interaction details:
                - user_id: User identifier
                - session_id: Session identifier
                - selected_strategy: Selected strategy name
                - feedback_text: User feedback text
                - recommendations: List of recommendations shown
                - context: Context information
        """
        logger.debug(f"POST /api/v1/user_interactions")
        response = self.client.post("/api/v1/user_interactions", json=interaction_data)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close HTTP client"""
        self.client.close()
