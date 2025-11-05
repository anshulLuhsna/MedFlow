# Agent Workflow Integration Guide

Guide for integrating MedFlow API with LangGraph agents (Phase 5).

---

## Workflow Overview

```
START
  ↓
[1. Data Analyst] → Assess situation
  ↓
[2. Forecasting] → Predict future demand
  ↓
[3. Optimization] → Generate strategies
  ↓
[4. Preference Learning] → Rank by user preferences
  ↓
[5. Reasoning] → Explain recommendations
  ↓
PRESENT TO USER
  ↓
[Feedback Loop] → Learn from decision
```

---

## Agent Nodes

### 1. Data Analyst Agent

**Role:** Situational awareness - understand current state

**Endpoints Used:**
- `GET /api/v1/hospitals` - Get hospital list
- `GET /api/v1/outbreaks/active` - Check ongoing events
- `GET /api/v1/shortages` - Identify at-risk hospitals

**State Updates:**
```python
{
  "hospitals": [...],
  "active_outbreaks": [...],
  "shortage_hospitals": [...],
  "shortage_count": 12
}
```

**Example Tool:**
```python
@tool
def analyze_situation(resource_type: str = None) -> dict:
    """Analyze current resource situation"""
    # Get shortages
    shortages = requests.get(
        f"{API_BASE}/api/v1/shortages",
        headers={"X-API-Key": API_KEY},
        params={"resource_type": resource_type}
    ).json()

    # Get active outbreaks
    outbreaks = requests.get(
        f"{API_BASE}/api/v1/outbreaks/active",
        headers={"X-API-Key": API_KEY}
    ).json()

    return {
        "shortages": shortages["shortages"],
        "shortage_count": shortages["count"],
        "active_outbreaks": outbreaks["active_outbreaks"],
        "summary": shortages["summary"]
    }
```

**Conditional Routing:**
```python
def route_after_analysis(state):
    if state["shortage_count"] == 0:
        return "END"  # No action needed
    else:
        return "forecasting"  # Continue to forecasting
```

---

### 2. Forecasting Agent

**Role:** Predict future demand for at-risk hospitals

**Endpoints Used:**
- `POST /api/v1/predict/demand` (⚠️ Currently unavailable)

**State Updates:**
```python
{
  "demand_forecasts": {
    "hospital_uuid_1": {"ppe": [120, 125, ...], "ventilators": [2, 3, ...]},
    "hospital_uuid_2": {...}
  }
}
```

**Note:** Skip this node until demand prediction is fixed. Use shortage detection data directly for optimization.

---

### 3. Optimization Agent

**Role:** Generate allocation strategies

**Endpoints Used:**
- `POST /api/v1/strategies` - Generate multiple strategies

**State Updates:**
```python
{
  "allocation_strategies": [
    {"strategy_name": "Cost-Efficient", "allocations": [...], "summary": {...}},
    {"strategy_name": "Maximum Coverage", ...},
    {"strategy_name": "Balanced", ...}
  ]
}
```

**Example Tool:**
```python
@tool
def generate_strategies(resource_type: str, n_strategies: int = 3) -> dict:
    """Generate allocation strategies"""
    response = requests.post(
        f"{API_BASE}/api/v1/strategies",
        headers={"X-API-Key": API_KEY},
        json={
            "resource_type": resource_type,
            "n_strategies": n_strategies,
            "limit": 50
        }
    )
    return response.json()
```

**Conditional Routing:**
```python
def route_after_optimization(state):
    strategies = state["allocation_strategies"]

    # Check if any strategy is optimal
    has_optimal = any(s["status"] == "optimal" for s in strategies)

    if has_optimal:
        return "preference_learning"
    else:
        return "END"  # No feasible strategies
```

---

### 4. Preference Learning Agent

**Role:** Rank strategies by user preferences

**Endpoints Used:**
- `POST /api/v1/preferences/score` - Rank strategies
- `POST /api/v1/preferences/update` - Learn from feedback (after user decision)

**State Updates:**
```python
{
  "ranked_strategies": [
    {
      "strategy_name": "Cost-Efficient",
      "preference_score": 0.85,
      "llm_explanation": "..."
    }
  ]
}
```

**Example Tools:**
```python
@tool
def rank_by_preferences(user_id: str, strategies: list) -> dict:
    """Rank strategies by user preferences"""
    response = requests.post(
        f"{API_BASE}/api/v1/preferences/score",
        headers={"X-API-Key": API_KEY},
        json={
            "user_id": user_id,
            "recommendations": strategies
        }
    )
    return response.json()

@tool
def learn_from_feedback(user_id: str, selected_index: int, strategies: list) -> dict:
    """Update preferences after user decision"""
    response = requests.post(
        f"{API_BASE}/api/v1/preferences/update",
        headers={"X-API-Key": API_KEY},
        json={
            "user_id": user_id,
            "interaction": {
                "selected_recommendation_index": selected_index,
                "recommendations": strategies,
                "timestamp": datetime.now().isoformat()
            }
        }
    )
    return response.json()
```

---

### 5. Reasoning Agent

**Role:** Generate natural language explanations

**Endpoints Used:** None (uses LLM directly)

**State Updates:**
```python
{
  "final_recommendation": "...",
  "explanation": "...",
  "reasoning": "..."
}
```

**Example:**
```python
def generate_explanation(state):
    """Generate natural language explanation"""

    top_strategy = state["ranked_strategies"][0]
    shortages = state["shortage_hospitals"]

    prompt = f"""
    Based on the analysis:
    - {len(shortages)} hospitals have shortages
    - Top recommended strategy: {top_strategy['strategy_name']}
    - This will help {top_strategy['summary']['hospitals_helped']} hospitals
    - Cost: ${top_strategy['summary']['total_cost']}

    Explain why this is the best strategy in simple terms.
    """

    explanation = llm.invoke(prompt)

    return {
        "final_recommendation": top_strategy,
        "explanation": explanation.content
    }
```

---

## Complete LangGraph Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
import requests

# Define state
class AgentState(TypedDict):
    shortage_count: int
    shortage_hospitals: list
    allocation_strategies: list
    ranked_strategies: list
    final_recommendation: dict

# Define nodes
def data_analyst(state):
    shortages = requests.get(
        f"{API_BASE}/api/v1/shortages",
        headers={"X-API-Key": API_KEY}
    ).json()

    return {
        "shortage_count": shortages["count"],
        "shortage_hospitals": shortages["shortages"]
    }

def optimizer(state):
    if state["shortage_count"] == 0:
        return state

    strategies = requests.post(
        f"{API_BASE}/api/v1/strategies",
        headers={"X-API-Key": API_KEY},
        json={"resource_type": "ventilators", "n_strategies": 3}
    ).json()

    return {"allocation_strategies": strategies["strategies"]}

def preference_learner(state):
    ranked = requests.post(
        f"{API_BASE}/api/v1/preferences/score",
        headers={"X-API-Key": API_KEY},
        json={
            "user_id": "user_123",
            "recommendations": state["allocation_strategies"]
        }
    ).json()

    return {"ranked_strategies": ranked["ranked_strategies"]}

def reasoning_agent(state):
    top = state["ranked_strategies"][0]
    return {"final_recommendation": top}

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyst", data_analyst)
workflow.add_node("optimizer", optimizer)
workflow.add_node("preference", preference_learner)
workflow.add_node("reasoning", reasoning_agent)

# Add edges
workflow.set_entry_point("analyst")
workflow.add_edge("analyst", "optimizer")
workflow.add_edge("optimizer", "preference")
workflow.add_edge("preference", "reasoning")
workflow.add_edge("reasoning", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({})
print(result["final_recommendation"])
```

---

## Error Handling for Agents

```python
def safe_api_call(endpoint, method="GET", **kwargs):
    """Wrapper with retry logic"""
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call():
        if method == "GET":
            response = requests.get(endpoint, **kwargs)
        else:
            response = requests.post(endpoint, **kwargs)

        response.raise_for_status()
        return response.json()

    try:
        return {"success": True, "data": _call()}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## State Management

**Typical State Object:**
```python
{
  # Data Analyst
  "hospitals": [...],
  "active_outbreaks": [...],
  "shortage_hospitals": [...],
  "shortage_count": 12,

  # Optimizer
  "allocation_strategies": [...],

  # Preference Learning
  "ranked_strategies": [...],
  "user_id": "user_123",

  # Reasoning
  "final_recommendation": {...},
  "explanation": "...",

  # Metadata
  "timestamp": "2025-11-05T10:00:00",
  "resource_type": "ventilators"
}
```

---

## Next Steps for Phase 5

1. **Set up LangGraph project structure**
2. **Implement 5 agent nodes** (analyst, forecasting, optimizer, preference, reasoning)
3. **Add conditional routing** based on shortage count and strategy status
4. **Implement feedback loop** for preference learning
5. **Add monitoring and logging** for agent decisions
6. **Test end-to-end workflow** with real data

See Phase 5 plan for detailed implementation steps.
