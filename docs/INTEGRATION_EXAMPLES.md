# Integration Examples

Practical code examples for integrating MedFlow API with LangGraph agents.

---

## Python Client

```python
import requests
from typing import Optional, List, Dict
from datetime import datetime

class MedFlowClient:
    """Simple client for MedFlow API"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}

    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()["status"] == "healthy"
        except:
            return False

    def get_shortages(self, resource_type: Optional[str] = None) -> Dict:
        """Detect shortages"""
        params = {}
        if resource_type:
            params["resource_type"] = resource_type

        response = requests.get(
            f"{self.base_url}/api/v1/shortages",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    def generate_strategies(
        self,
        resource_type: str,
        n_strategies: int = 3,
        limit: Optional[int] = None
    ) -> Dict:
        """Generate allocation strategies"""
        payload = {
            "resource_type": resource_type,
            "n_strategies": n_strategies
        }
        if limit:
            payload["limit"] = limit

        response = requests.post(
            f"{self.base_url}/api/v1/strategies",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def rank_strategies(
        self,
        user_id: str,
        strategies: List[Dict]
    ) -> Dict:
        """Rank strategies by user preferences"""
        response = requests.post(
            f"{self.base_url}/api/v1/preferences/score",
            headers=self.headers,
            json={
                "user_id": user_id,
                "recommendations": strategies
            }
        )
        response.raise_for_status()
        return response.json()

    def update_preferences(
        self,
        user_id: str,
        selected_index: int,
        strategies: List[Dict]
    ) -> Dict:
        """Learn from user decision"""
        response = requests.post(
            f"{self.base_url}/api/v1/preferences/update",
            headers=self.headers,
            json={
                "user_id": user_id,
                "interaction": {
                    "selected_recommendation_index": selected_index,
                    "recommendations": strategies,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
        response.raise_for_status()
        return response.json()


# Usage
client = MedFlowClient("http://localhost:8000", "your_api_key")

# Complete workflow
if client.health_check():
    shortages = client.get_shortages("ventilators")
    strategies = client.generate_strategies("ventilators")
    ranked = client.rank_strategies("user_123", strategies["strategies"])
    print(f"Top: {ranked['ranked_strategies'][0]['strategy_name']}")
```

---

## LangChain Tools

```python
from langchain.tools import tool
import os

API_BASE = os.getenv("MEDFLOW_API_BASE", "http://localhost:8000")
API_KEY = os.getenv("MEDFLOW_API_KEY")


@tool
def detect_shortages(resource_type: str = None) -> dict:
    """
    Detect hospitals with shortage risks.

    Args:
        resource_type: Optional filter (ppe, ventilators, etc.)

    Returns:
        Dictionary with shortage information
    """
    import requests

    params = {}
    if resource_type:
        params["resource_type"] = resource_type

    response = requests.get(
        f"{API_BASE}/api/v1/shortages",
        headers={"X-API-Key": API_KEY},
        params=params
    )
    response.raise_for_status()
    return response.json()


@tool
def generate_allocation_strategies(resource_type: str, n_strategies: int = 3) -> dict:
    """
    Generate resource allocation strategies.

    Args:
        resource_type: Type of resource (ppe, ventilators, etc.)
        n_strategies: Number of strategies to generate (1-5)

    Returns:
        Dictionary with allocation strategies
    """
    import requests

    response = requests.post(
        f"{API_BASE}/api/v1/strategies",
        headers={"X-API-Key": API_KEY},
        json={
            "resource_type": resource_type,
            "n_strategies": n_strategies
        }
    )
    response.raise_for_status()
    return response.json()


@tool
def rank_by_user_preferences(user_id: str, strategies: list) -> dict:
    """
    Rank strategies by learned user preferences.

    Args:
        user_id: User identifier
        strategies: List of strategy dictionaries

    Returns:
        Ranked strategies with preference scores
    """
    import requests

    response = requests.post(
        f"{API_BASE}/api/v1/preferences/score",
        headers={"X-API-Key": API_KEY},
        json={
            "user_id": user_id,
            "recommendations": strategies
        }
    )
    response.raise_for_status()
    return response.json()


# Usage with LangChain agent
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

tools = [detect_shortages, generate_allocation_strategies, rank_by_user_preferences]

agent = initialize_agent(
    tools,
    ChatOpenAI(model="gpt-4"),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Find ventilator shortages and generate 3 allocation strategies"
)
```

---

## LangGraph Complete Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import requests
import os

# Configuration
API_BASE = os.getenv("MEDFLOW_API_BASE", "http://localhost:8000")
API_KEY = os.getenv("MEDFLOW_API_KEY")
HEADERS = {"X-API-Key": API_KEY}


# State definition
class MedFlowState(TypedDict):
    """State for MedFlow agent workflow"""
    resource_type: str
    user_id: str
    shortage_count: int
    shortage_hospitals: Annotated[list, operator.add]
    allocation_strategies: Annotated[list, operator.add]
    ranked_strategies: Annotated[list, operator.add]
    final_recommendation: dict
    messages: Annotated[list, operator.add]


# Node 1: Data Analyst
def analyze_situation(state: MedFlowState) -> MedFlowState:
    """Analyze current shortage situation"""

    # Detect shortages
    response = requests.get(
        f"{API_BASE}/api/v1/shortages",
        headers=HEADERS,
        params={"resource_type": state.get("resource_type")}
    )
    shortages = response.json()

    return {
        "shortage_count": shortages["count"],
        "shortage_hospitals": shortages["shortages"],
        "messages": [f"Found {shortages['count']} hospitals with shortages"]
    }


# Node 2: Optimization Agent
def optimize_allocation(state: MedFlowState) -> MedFlowState:
    """Generate allocation strategies"""

    if state["shortage_count"] == 0:
        return {
            "messages": ["No shortages detected. No action needed."]
        }

    # Generate strategies
    response = requests.post(
        f"{API_BASE}/api/v1/strategies",
        headers=HEADERS,
        json={
            "resource_type": state.get("resource_type", "ventilators"),
            "n_strategies": 3
        }
    )
    strategies = response.json()

    return {
        "allocation_strategies": strategies["strategies"],
        "messages": [f"Generated {strategies['count']} allocation strategies"]
    }


# Node 3: Preference Learning
def learn_preferences(state: MedFlowState) -> MedFlowState:
    """Rank strategies by user preferences"""

    # Rank strategies
    response = requests.post(
        f"{API_BASE}/api/v1/preferences/score",
        headers=HEADERS,
        json={
            "user_id": state.get("user_id", "user_123"),
            "recommendations": state["allocation_strategies"]
        }
    )
    ranked = response.json()

    return {
        "ranked_strategies": ranked["ranked_strategies"],
        "messages": [f"Ranked strategies by user preferences"]
    }


# Node 4: Decision
def make_recommendation(state: MedFlowState) -> MedFlowState:
    """Make final recommendation"""

    top_strategy = state["ranked_strategies"][0]

    return {
        "final_recommendation": top_strategy,
        "messages": [
            f"Top recommendation: {top_strategy['strategy_name']}",
            f"Preference score: {top_strategy['preference_score']:.2f}"
        ]
    }


# Conditional routing
def should_continue(state: MedFlowState) -> str:
    """Decide next node based on state"""
    if state.get("shortage_count", 0) == 0:
        return "END"
    elif not state.get("allocation_strategies"):
        return "optimizer"
    elif not state.get("ranked_strategies"):
        return "preference"
    else:
        return "decision"


# Build graph
workflow = StateGraph(MedFlowState)

# Add nodes
workflow.add_node("analyst", analyze_situation)
workflow.add_node("optimizer", optimize_allocation)
workflow.add_node("preference", learn_preferences)
workflow.add_node("decision", make_recommendation)

# Add edges
workflow.set_entry_point("analyst")
workflow.add_conditional_edges(
    "analyst",
    should_continue,
    {
        "optimizer": "optimizer",
        "END": END
    }
)
workflow.add_edge("optimizer", "preference")
workflow.add_edge("preference", "decision")
workflow.add_edge("decision", END)

# Compile
app = workflow.compile()


# Run workflow
if __name__ == "__main__":
    result = app.invoke({
        "resource_type": "ventilators",
        "user_id": "user_123",
        "shortage_count": 0,
        "shortage_hospitals": [],
        "allocation_strategies": [],
        "ranked_strategies": [],
        "messages": []
    })

    # Print results
    print("\n=== Workflow Complete ===")
    for msg in result["messages"]:
        print(f"→ {msg}")

    if result.get("final_recommendation"):
        rec = result["final_recommendation"]
        print(f"\n✅ Recommendation: {rec['strategy_name']}")
        print(f"   Score: {rec['preference_score']:.2f}")
```

---

## Async Example

```python
import asyncio
import aiohttp


class AsyncMedFlowClient:
    """Async client for parallel API calls"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key}

    async def get_shortages(self, resource_type: str = None):
        """Async shortage detection"""
        async with aiohttp.ClientSession() as session:
            params = {}
            if resource_type:
                params["resource_type"] = resource_type

            async with session.get(
                f"{self.base_url}/api/v1/shortages",
                headers=self.headers,
                params=params
            ) as response:
                return await response.json()

    async def generate_strategies(self, resource_type: str):
        """Async strategy generation"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/strategies",
                headers=self.headers,
                json={"resource_type": resource_type, "n_strategies": 3}
            ) as response:
                return await response.json()

    async def complete_workflow(self, resource_type: str, user_id: str):
        """Run complete workflow with parallel calls where possible"""

        # Step 1: Get shortages
        shortages = await self.get_shortages(resource_type)

        if shortages["count"] == 0:
            return {"status": "no_action_needed"}

        # Step 2: Generate strategies
        strategies = await self.generate_strategies(resource_type)

        # Step 3: Rank strategies
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/preferences/score",
                headers=self.headers,
                json={
                    "user_id": user_id,
                    "recommendations": strategies["strategies"]
                }
            ) as response:
                ranked = await response.json()

        return {
            "status": "complete",
            "shortages": shortages,
            "strategies": strategies,
            "ranked": ranked
        }


# Usage
async def main():
    client = AsyncMedFlowClient("http://localhost:8000", "your_api_key")
    result = await client.complete_workflow("ventilators", "user_123")
    print(result)

asyncio.run(main())
```

---

## Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import requests


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def robust_api_call(endpoint: str, method: str = "GET", **kwargs):
    """API call with retry logic"""

    if method == "GET":
        response = requests.get(endpoint, **kwargs)
    else:
        response = requests.post(endpoint, **kwargs)

    response.raise_for_status()
    return response.json()


# Usage
try:
    result = robust_api_call(
        "http://localhost:8000/api/v1/shortages",
        headers={"X-API-Key": "your_key"}
    )
except Exception as e:
    print(f"Failed after 3 retries: {e}")
```

---

## Testing

```python
import pytest
from unittest.mock import patch, Mock


def test_shortage_detection():
    """Test shortage detection integration"""

    mock_response = Mock()
    mock_response.json.return_value = {
        "shortages": [{"hospital_id": "uuid", "risk_level": "critical"}],
        "count": 1
    }
    mock_response.raise_for_status = Mock()

    with patch('requests.get', return_value=mock_response):
        client = MedFlowClient("http://localhost:8000", "test_key")
        result = client.get_shortages("ventilators")

        assert result["count"] == 1
        assert result["shortages"][0]["risk_level"] == "critical"


def test_strategy_generation():
    """Test strategy generation integration"""

    mock_response = Mock()
    mock_response.json.return_value = {
        "strategies": [
            {"strategy_name": "Cost-Efficient", "status": "optimal"}
        ],
        "count": 1
    }
    mock_response.raise_for_status = Mock()

    with patch('requests.post', return_value=mock_response):
        client = MedFlowClient("http://localhost:8000", "test_key")
        result = client.generate_strategies("ventilators")

        assert result["count"] == 1
        assert result["strategies"][0]["status"] == "optimal"
```

---

## Environment Setup

```bash
# .env file
MEDFLOW_API_BASE=http://localhost:8000
MEDFLOW_API_KEY=your_secret_api_key_here

# Load in Python
from dotenv import load_dotenv
load_dotenv()

API_BASE = os.getenv("MEDFLOW_API_BASE")
API_KEY = os.getenv("MEDFLOW_API_KEY")
```

---

See `AGENT_WORKFLOW.md` for architectural guidance.
