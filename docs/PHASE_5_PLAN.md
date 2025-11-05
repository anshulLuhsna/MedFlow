# Phase 5 Implementation Plan: LangGraph Agentic Layer

**Status:** Ready to Implement
**Date:** 2025-11-05
**Phase 4 Status:** ✅ Complete - All 16 endpoints functional (22/22 tests passing)

---

## 📋 Executive Summary

Phase 5 implements an intelligent multi-agent orchestration system using **LangGraph** that automates resource allocation decisions through specialized AI agents with human-in-the-loop oversight and adaptive preference learning.

**Key Features:**
- 🤖 **5 Specialized Agents**: Data analysis, forecasting, optimization, preference learning, reasoning
- 🔄 **LangGraph Orchestration**: State-based workflow with conditional routing
- 👤 **Human-in-the-Loop**: Review recommendations before execution
- 🧠 **Adaptive Learning**: System learns from user decisions over time
- 💾 **State Persistence**: Resume workflows across sessions
- 🖥️ **CLI Interface**: Beautiful terminal UI with Rich
- 📊 **Optional Dashboard**: Streamlit web interface

---

## 🎯 Phase 5 Objectives

### Primary Goals
1. **Build LangGraph Workflow** - Orchestrate 5 agents in a coherent decision pipeline
2. **Enable HITL Interaction** - Allow users to review and approve recommendations
3. **Implement Feedback Loop** - Learn from user decisions to improve future recommendations
4. **Create CLI Interface** - Provide intuitive command-line interaction
5. **Ensure State Persistence** - Save and resume workflows seamlessly

### Success Metrics
- ✅ Complete workflow executes in < 60 seconds (excluding human review)
- ✅ All 5 agent nodes functional with proper error handling
- ✅ State persistence working across sessions
- ✅ Preference learning improves recommendations over time (measurable via preference scores)
- ✅ 80%+ test coverage
- ✅ Comprehensive documentation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                            │
│  ┌──────────────────┐              ┌──────────────────────┐        │
│  │  CLI (Typer)     │              │ Streamlit Dashboard  │        │
│  │  - Rich UI       │              │ (Optional Phase 5.4) │        │
│  │  - Interactive   │              │                      │        │
│  └────────┬─────────┘              └──────────┬───────────┘        │
└───────────┼────────────────────────────────────┼────────────────────┘
            │                                    │
┌───────────▼────────────────────────────────────▼────────────────────┐
│                    LANGGRAPH ORCHESTRATION LAYER                     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    StateGraph Workflow                      │    │
│  │                                                             │    │
│  │   START                                                     │    │
│  │     │                                                       │    │
│  │     ▼                                                       │    │
│  │  ┌─────────────────┐                                       │    │
│  │  │ Data Analyst    │  "Assess current resource situation"  │    │
│  │  │ Agent (Node 1)  │  Endpoints: /shortages, /outbreaks    │    │
│  │  └────────┬────────┘                                       │    │
│  │           │                                                 │    │
│  │           │ shortage_count > 0?                            │    │
│  │           ├─── NO ──────────────────────────► END          │    │
│  │           │                                                 │    │
│  │           └─── YES                                         │    │
│  │                │                                            │    │
│  │                ▼                                            │    │
│  │  ┌─────────────────┐                                       │    │
│  │  │ Forecasting     │  "Predict 14-day demand"             │    │
│  │  │ Agent (Node 2)  │  Endpoint: POST /predict/demand       │    │
│  │  └────────┬────────┘  ✅ NOW WORKING (17 features)        │    │
│  │           │                                                 │    │
│  │           ▼                                                 │    │
│  │  ┌─────────────────┐                                       │    │
│  │  │ Optimization    │  "Generate 3 allocation strategies"   │    │
│  │  │ Agent (Node 3)  │  Endpoint: POST /strategies           │    │
│  │  └────────┬────────┘                                       │    │
│  │           │                                                 │    │
│  │           ▼                                                 │    │
│  │  ┌─────────────────┐                                       │    │
│  │  │ Preference      │  "Rank by learned preferences"        │    │
│  │  │ Agent (Node 4)  │  Endpoint: POST /preferences/score    │    │
│  │  └────────┬────────┘                                       │    │
│  │           │                                                 │    │
│  │           ▼                                                 │    │
│  │  ┌─────────────────┐                                       │    │
│  │  │ Reasoning       │  "Generate LLM explanation"           │    │
│  │  │ Agent (Node 5)  │  LLM: GPT-4 / Groq Llama 3.3 70B     │    │
│  │  └────────┬────────┘                                       │    │
│  │           │                                                 │    │
│  │           ▼                                                 │    │
│  │  ┌─────────────────┐                                       │    │
│  │  │ Human Review    │  "User reviews top 3 recommendations" │    │
│  │  │ (HITL Node)     │  Select strategy + optional feedback  │    │
│  │  └────────┬────────┘                                       │    │
│  │           │                                                 │    │
│  │           │ User provided feedback?                        │    │
│  │           ├─── YES ─► Update Preferences Node              │    │
│  │           │              │                                  │    │
│  │           │              ▼                                  │    │
│  │           │           Endpoint: POST /preferences/update   │    │
│  │           │              │                                  │    │
│  │           └─── NO ───────┴─────────────────► END          │    │
│  │                                                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  State Persistence: SQLite Checkpointer                             │
│  Config: {"configurable": {"thread_id": user_id}}                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                      FASTAPI BACKEND (Phase 4)                       │
│                                                                      │
│  Health (2) | Hospitals (4) | Predictions (4) | Preferences (2)    │
│  Outbreaks (4)                                                      │
│                                                                      │
│  ✅ All 16 endpoints functional                                     │
│  ✅ 22/22 tests passing                                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                    ML CORE + SUPABASE DATABASE                       │
│                                                                      │
│  • LSTM Demand Forecasting (5 resources)                            │
│  • Random Forest Shortage Detection                                 │
│  • Linear Programming Optimization                                  │
│  • Hybrid Preference Learning (RF + LLM + Vector DB)                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
MedFlow/
├── agents/                          # ⭐ NEW: Phase 5 Agentic Layer
│   ├── __init__.py
│   ├── config.py                   # Agent configuration & LLM settings
│   ├── state.py                    # State definitions (TypedDict)
│   ├── graph.py                    # Main LangGraph workflow builder
│   │
│   ├── nodes/                      # Agent node implementations
│   │   ├── __init__.py
│   │   ├── data_analyst.py         # Node 1: Situational analysis
│   │   ├── forecasting.py          # Node 2: Demand predictions (WORKING!)
│   │   ├── optimization.py         # Node 3: Strategy generation
│   │   ├── preference.py           # Node 4: Preference ranking
│   │   ├── reasoning.py            # Node 5: LLM explanation
│   │   ├── human_review.py         # HITL node
│   │   └── feedback.py             # Preference update node
│   │
│   ├── tools/                      # LangChain tool wrappers for API
│   │   ├── __init__.py
│   │   ├── api_client.py           # FastAPI client with retry logic
│   │   ├── shortage_tools.py       # @tool wrappers for shortage detection
│   │   ├── forecasting_tools.py    # @tool wrappers for demand prediction
│   │   ├── optimization_tools.py   # @tool wrappers for optimization
│   │   └── preference_tools.py     # @tool wrappers for preference learning
│   │
│   ├── prompts/                    # LLM prompt templates
│   │   ├── __init__.py
│   │   ├── reasoning_prompts.py    # System prompts for reasoning agent
│   │   └── few_shot_examples.py    # Few-shot examples for better LLM output
│   │
│   ├── utils/                      # Helper utilities
│   │   ├── __init__.py
│   │   ├── formatters.py           # Output formatting (tables, summaries)
│   │   ├── validators.py           # Input validation
│   │   └── routing.py              # Conditional routing logic
│   │
│   ├── checkpoints/                # State persistence (SQLite)
│   │   ├── .gitkeep
│   │   └── workflows.db            # Auto-created by SqliteSaver
│   │
│   └── tests/                      # Agent tests
│       ├── __init__.py
│       ├── test_nodes.py           # Unit tests for each node
│       ├── test_graph.py           # Integration tests for workflow
│       ├── test_tools.py           # Tool wrapper tests
│       └── test_e2e.py             # End-to-end workflow tests
│
├── cli/                            # ⭐ NEW: Command-line interface
│   ├── __init__.py
│   ├── main.py                     # Typer CLI app entry point
│   ├── commands/                   # CLI command modules
│   │   ├── __init__.py
│   │   ├── allocate.py             # Main allocation workflow command
│   │   ├── history.py              # View decision history
│   │   └── preferences.py          # View/manage user preferences
│   │
│   └── ui/                         # Rich UI components
│       ├── __init__.py
│       ├── tables.py               # Rich table formatters
│       ├── panels.py               # Rich panel components
│       └── prompts.py              # Interactive prompts
│
├── dashboard/                      # 🔮 OPTIONAL: Streamlit web UI (Phase 5.4)
│   ├── app.py                      # Main Streamlit app
│   ├── pages/
│   │   ├── 1_workflow.py           # Workflow execution & visualization
│   │   ├── 2_history.py            # Decision history viewer
│   │   └── 3_preferences.py        # Preference analytics
│   └── components/                 # Reusable UI components
│       ├── workflow_viz.py         # LangGraph visualization
│       └── metrics.py              # Metrics display
│
├── backend/                        # ✅ Phase 4: Existing FastAPI backend
├── ml_core/                        # ✅ Phase 4: Existing ML models
├── data/                           # ✅ Phase 4: Existing data generators
├── docs/                           # Documentation
│   ├── PHASE_5_PLAN.md            # ⭐ This document
│   ├── PHASE_5_DOCUMENTATION.md   # Final implementation docs (to be created)
│   ├── AGENT_ARCHITECTURE.md      # Agent design details (to be created)
│   └── CLI_GUIDE.md               # CLI usage guide (to be created)
├── tests/                          # Existing integration tests
└── requirements.txt                # ⭐ Updated with LangGraph dependencies
```

---

## 🛠️ Technology Stack & Dependencies

### New Dependencies for Phase 5

```python
# requirements.txt additions

# ============================================================================
# LangGraph & LangChain - Agent Orchestration
# ============================================================================
langgraph>=0.2.28                   # Graph-based agent orchestration
langchain>=0.3.0                    # Agent framework
langchain-core>=0.3.0               # Core abstractions
langchain-openai>=0.2.0             # OpenAI LLM integration
langchain-groq>=0.2.0               # Groq LLM integration (Llama 3.3 70B)

# ============================================================================
# State Persistence & Checkpointing
# ============================================================================
langgraph-checkpoint>=1.0.0         # Checkpoint interface
langgraph-checkpoint-sqlite>=1.0.0  # SQLite-based state persistence

# ============================================================================
# LLM Providers
# ============================================================================
openai>=1.50.0                      # OpenAI API (GPT-4 for reasoning)
groq>=0.11.0                        # Groq API (already in use for preferences)

# ============================================================================
# Vector Store (already installed for preferences)
# ============================================================================
qdrant-client>=1.11.0               # Vector similarity search

# ============================================================================
# CLI Interface
# ============================================================================
typer>=0.12.0                       # Modern CLI framework with type hints
rich>=13.8.0                        # Beautiful terminal UI
prompt-toolkit>=3.0.47              # Interactive prompts & auto-completion

# ============================================================================
# Async HTTP Client
# ============================================================================
httpx>=0.27.0                       # Async HTTP client for API calls
aiohttp>=3.10.0                     # Alternative async HTTP client

# ============================================================================
# Retry & Resilience
# ============================================================================
tenacity>=9.0.0                     # Retry logic with exponential backoff

# ============================================================================
# Optional: Monitoring & Observability
# ============================================================================
langsmith>=0.1.0                    # LangChain monitoring (optional)

# ============================================================================
# Optional: Web Dashboard (Phase 5.4)
# ============================================================================
streamlit>=1.39.0                   # Web UI framework
plotly>=5.24.0                      # Interactive visualizations
streamlit-agraph>=0.0.45            # Graph visualization for Streamlit
```

### Environment Variables

```bash
# .env additions for Phase 5

# ============================================================================
# LLM API Keys
# ============================================================================
OPENAI_API_KEY=sk-...                      # OpenAI API key for GPT-4
GROQ_API_KEY=gsk_...                       # Groq API key (already configured)

# ============================================================================
# MedFlow Backend
# ============================================================================
MEDFLOW_API_BASE=http://localhost:8000     # Backend API URL
MEDFLOW_API_KEY=your_secret_api_key        # API authentication key

# ============================================================================
# LangSmith Monitoring (Optional)
# ============================================================================
LANGCHAIN_TRACING_V2=true                  # Enable tracing
LANGCHAIN_API_KEY=ls__...                  # LangSmith API key
LANGCHAIN_PROJECT=medflow-agents           # Project name

# ============================================================================
# Agent Configuration
# ============================================================================
DEFAULT_LLM_MODEL=gpt-4o                   # Model for reasoning agent
DEFAULT_LLM_TEMPERATURE=0.3                # Temperature (0.0-1.0)
MAX_RETRIES=3                              # API retry attempts
TIMEOUT_SECONDS=60                         # Request timeout
```

---

## 📊 State Schema Design

### Latest LangGraph Syntax (2025)

```python
# agents/state.py
from typing import TypedDict, Annotated, Optional, List, Dict, Literal
from langgraph.graph.message import add_messages
from datetime import datetime

class MedFlowState(TypedDict):
    """
    Central state for MedFlow agent workflow.

    Uses TypedDict with Annotated for reducer functions following
    LangGraph 2025 best practices.

    State flows through 5 agent nodes:
    1. Data Analyst → shortage analysis
    2. Forecasting → demand predictions
    3. Optimization → allocation strategies
    4. Preference → ranking by user preferences
    5. Reasoning → LLM explanation generation
    """

    # ========================================================================
    # INPUT PARAMETERS (Set by user at workflow start)
    # ========================================================================
    resource_type: str
    """Resource to allocate: 'ppe', 'ventilators', 'o2', 'beds', 'medications'"""

    user_id: str
    """User identifier for preference learning"""

    session_id: str
    """Unique session ID (UUID) for this workflow execution"""

    # ========================================================================
    # MESSAGES (For LLM context and conversation history)
    # ========================================================================
    messages: Annotated[list, add_messages]
    """Chat-style messages for LLM context. Uses add_messages reducer."""

    # ========================================================================
    # NODE 1: DATA ANALYST OUTPUTS
    # ========================================================================
    shortage_count: int
    """Number of hospitals with detected shortages"""

    shortage_hospitals: List[Dict]
    """
    List of hospitals with shortages. Each dict contains:
    {
        "hospital_id": str,
        "hospital_name": str,
        "risk_level": "critical" | "high" | "medium" | "low",
        "current_stock": float,
        "predicted_shortage": float,
        "days_until_critical": int
    }
    """

    active_outbreaks: List[Dict]
    """
    Currently active outbreak events. Each dict contains:
    {
        "outbreak_id": str,
        "outbreak_type": str,
        "region": str,
        "affected_hospitals": int,
        "start_date": str
    }
    """

    analysis_summary: str
    """Human-readable summary of current situation"""

    # ========================================================================
    # NODE 2: FORECASTING OUTPUTS
    # ========================================================================
    demand_forecasts: Dict[str, Dict]
    """
    14-day demand predictions per hospital. Structure:
    {
        "hospital_uuid_1": {
            "predictions": {
                "point_forecast": [float, ...],  # 14 values
                "lower_bound": [float, ...],     # P10 percentile
                "upper_bound": [float, ...]      # P90 percentile
            },
            "metadata": {
                "model_version": str,
                "mae": float,
                "confidence_level": 0.8
            }
        },
        ...
    }
    """

    forecast_summary: str
    """Summary of key forecast insights"""

    # ========================================================================
    # NODE 3: OPTIMIZATION OUTPUTS
    # ========================================================================
    allocation_strategies: List[Dict]
    """
    Generated allocation strategies (typically 3). Each dict contains:
    {
        "strategy_name": str,  # "Cost-Efficient", "Maximum Coverage", "Balanced"
        "allocations": [
            {
                "from_hospital_id": str,
                "to_hospital_id": str,
                "quantity": float,
                "distance_km": float,
                "cost": float
            },
            ...
        ],
        "summary": {
            "total_cost": float,
            "total_transfers": int,
            "hospitals_helped": int,
            "shortage_reduction": float,  # percentage
            "avg_distance": float
        },
        "status": "optimal" | "feasible" | "infeasible",
        "overall_score": float  # 0-100
    }
    """

    strategy_count: int
    """Number of strategies generated"""

    # ========================================================================
    # NODE 4: PREFERENCE LEARNING OUTPUTS
    # ========================================================================
    ranked_strategies: List[Dict]
    """
    Strategies ranked by learned user preferences. Same structure as
    allocation_strategies but with additional fields:
    {
        ...all fields from allocation_strategies...,
        "preference_score": float,  # 0.0 - 1.0
        "llm_explanation": str,
        "score_breakdown": {
            "rf_score": float,      # 40% weight
            "llm_score": float,     # 30% weight
            "vector_score": float   # 30% weight
        }
    }
    Sorted by preference_score descending.
    """

    preference_profile: Dict
    """
    User preference analysis:
    {
        "preference_type": "cost-conscious" | "coverage-focused" |
                          "urgency-driven" | "balanced",
        "confidence": float,  # 0.0 - 1.0
        "key_patterns": [str, ...],
        "interaction_count": int
    }
    """

    # ========================================================================
    # NODE 5: REASONING OUTPUTS
    # ========================================================================
    final_recommendation: Dict
    """Top-ranked strategy (ranked_strategies[0])"""

    explanation: str
    """Natural language explanation of recommendation"""

    reasoning_chain: str
    """Step-by-step reasoning process"""

    # ========================================================================
    # HUMAN-IN-THE-LOOP (HITL)
    # ========================================================================
    user_decision: Optional[int]
    """Index of strategy selected by user (0-2 typically)"""

    user_feedback: Optional[str]
    """Optional text feedback from user"""

    feedback_stored: bool
    """Whether feedback was successfully stored"""

    # ========================================================================
    # WORKFLOW METADATA
    # ========================================================================
    timestamp: str
    """ISO 8601 timestamp of workflow start"""

    workflow_status: Literal["pending", "in_progress", "completed", "failed"]
    """Current workflow status"""

    current_node: Optional[str]
    """Name of currently executing node"""

    error: Optional[str]
    """Error message if workflow failed"""

    execution_time_seconds: Optional[float]
    """Total execution time (excluding HITL wait time)"""
```

---

## 🤖 Agent Node Specifications

### Node 1: Data Analyst Agent

**Purpose**: Assess current resource situation and identify hospitals at risk

**File**: `agents/nodes/data_analyst.py`

**API Endpoints Used**:
- `GET /api/v1/shortages?resource_type={type}`
- `GET /api/v1/outbreaks/active`

**Inputs from State**:
- `resource_type`: Optional filter

**Outputs to State**:
- `shortage_count`: int
- `shortage_hospitals`: List[Dict]
- `active_outbreaks`: List[Dict]
- `analysis_summary`: str
- `messages`: Appends AI message with summary

**Implementation**:

```python
# agents/nodes/data_analyst.py
from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from agents.utils.formatters import format_shortage_summary
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger(__name__)

def data_analyst_node(state: MedFlowState) -> Dict:
    """
    Data Analyst Agent - Assess current resource situation.

    Responsibilities:
    1. Detect hospitals with shortage risks
    2. Identify active outbreak events
    3. Generate situational summary

    Args:
        state: Current workflow state

    Returns:
        State updates with shortage analysis
    """
    logger.info(f"[Data Analyst] Analyzing shortages for resource: {state.get('resource_type', 'all')}")

    # Initialize API client
    api_client = MedFlowAPIClient()

    # Call shortage detection API
    shortages = api_client.get_shortages(
        resource_type=state.get("resource_type")
    )

    # Get active outbreaks
    outbreaks = api_client.get_active_outbreaks()

    # Generate human-readable summary
    summary = format_shortage_summary(
        shortage_count=shortages["count"],
        shortages=shortages["shortages"],
        outbreaks=outbreaks["active_outbreaks"],
        resource_type=state.get("resource_type")
    )

    logger.info(f"[Data Analyst] Found {shortages['count']} hospitals with shortages")

    # Return state updates
    return {
        "shortage_count": shortages["count"],
        "shortage_hospitals": shortages["shortages"],
        "active_outbreaks": outbreaks["active_outbreaks"],
        "analysis_summary": summary,
        "messages": [AIMessage(content=summary)],
        "current_node": "data_analyst"
    }
```

**Conditional Routing**:

```python
# agents/utils/routing.py
def route_after_analysis(state: MedFlowState) -> str:
    """
    Route workflow based on shortage analysis.

    Logic:
    - If no shortages detected → END (no action needed)
    - If shortages detected → forecasting (predict future demand)
    """
    shortage_count = state.get("shortage_count", 0)

    if shortage_count == 0:
        logger.info("[Router] No shortages detected. Ending workflow.")
        return "END"
    else:
        logger.info(f"[Router] {shortage_count} shortages detected. Proceeding to forecasting.")
        return "forecasting"
```

---

### Node 2: Forecasting Agent

**Purpose**: Predict 14-day future demand for at-risk hospitals

**File**: `agents/nodes/forecasting.py`

**API Endpoints Used**:
- `POST /api/v1/predict/demand` ✅ **NOW WORKING** (17 features confirmed)

**Inputs from State**:
- `shortage_hospitals`: List of hospitals needing predictions
- `resource_type`: Resource to forecast

**Outputs to State**:
- `demand_forecasts`: Dict[hospital_id, predictions]
- `forecast_summary`: str
- `messages`: Appends forecast insights

**Implementation**:

```python
# agents/nodes/forecasting.py
from typing import Dict, List
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from agents.utils.formatters import format_forecast_summary
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger(__name__)

def forecasting_node(state: MedFlowState) -> Dict:
    """
    Forecasting Agent - Predict 14-day demand for at-risk hospitals.

    Uses LSTM models with 17 engineered features to generate:
    - Point predictions (expected daily consumption)
    - Probabilistic forecasts (P10-P90 confidence intervals)

    Args:
        state: Current workflow state

    Returns:
        State updates with demand forecasts
    """
    logger.info("[Forecasting] Generating 14-day demand predictions")

    api_client = MedFlowAPIClient()
    shortage_hospitals = state.get("shortage_hospitals", [])
    resource_type = state.get("resource_type")

    # Generate predictions for each at-risk hospital
    forecasts = {}

    for hospital in shortage_hospitals[:10]:  # Limit to top 10 for performance
        hospital_id = hospital["hospital_id"]

        try:
            prediction = api_client.predict_demand(
                hospital_id=hospital_id,
                resource_type=resource_type,
                forecast_days=14,
                probabilistic=True  # Get P10-P90 intervals
            )

            forecasts[hospital_id] = prediction
            logger.debug(f"[Forecasting] Predicted demand for {hospital_id}")

        except Exception as e:
            logger.warning(f"[Forecasting] Failed to predict for {hospital_id}: {e}")
            continue

    # Generate summary
    summary = format_forecast_summary(forecasts, resource_type)

    logger.info(f"[Forecasting] Generated predictions for {len(forecasts)} hospitals")

    return {
        "demand_forecasts": forecasts,
        "forecast_summary": summary,
        "messages": [AIMessage(content=summary)],
        "current_node": "forecasting"
    }
```

---

### Node 3: Optimization Agent

**Purpose**: Generate multiple allocation strategies with different objectives

**File**: `agents/nodes/optimization.py`

**API Endpoints Used**:
- `POST /api/v1/strategies`

**Inputs from State**:
- `shortage_hospitals`: Hospitals needing resources
- `demand_forecasts`: Predicted future demand
- `resource_type`: Resource to allocate

**Outputs to State**:
- `allocation_strategies`: List[Dict] (3 strategies)
- `strategy_count`: int
- `messages`: Strategy generation summary

**Implementation**:

```python
# agents/nodes/optimization.py
from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from agents.utils.formatters import format_strategy_summary
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger(__name__)

def optimization_node(state: MedFlowState) -> Dict:
    """
    Optimization Agent - Generate allocation strategies.

    Uses Linear Programming to create 3 strategies:
    1. Cost-Efficient: Minimize transfer costs
    2. Maximum Coverage: Help as many hospitals as possible
    3. Balanced: Trade-off between cost, coverage, urgency

    Args:
        state: Current workflow state

    Returns:
        State updates with allocation strategies
    """
    logger.info("[Optimization] Generating allocation strategies")

    api_client = MedFlowAPIClient()

    # Call strategy generation API
    result = api_client.generate_strategies(
        resource_type=state["resource_type"],
        n_strategies=3,  # Cost-Efficient, Maximum Coverage, Balanced
        limit=50  # Max hospitals to consider
    )

    strategies = result["strategies"]

    # Generate summary
    summary = format_strategy_summary(strategies)

    logger.info(f"[Optimization] Generated {len(strategies)} strategies")

    # Log key metrics
    for strategy in strategies:
        logger.info(
            f"  - {strategy['strategy_name']}: "
            f"${strategy['summary']['total_cost']:.0f}, "
            f"{strategy['summary']['hospitals_helped']} hospitals, "
            f"{strategy['summary']['shortage_reduction']:.1f}% reduction"
        )

    return {
        "allocation_strategies": strategies,
        "strategy_count": len(strategies),
        "messages": [AIMessage(content=summary)],
        "current_node": "optimization"
    }
```

---

### Node 4: Preference Learning Agent

**Purpose**: Rank strategies by learned user preferences using hybrid ML system

**File**: `agents/nodes/preference.py`

**API Endpoints Used**:
- `POST /api/v1/preferences/score`

**Inputs from State**:
- `allocation_strategies`: Strategies to rank
- `user_id`: User for personalization

**Outputs to State**:
- `ranked_strategies`: Sorted by preference_score
- `preference_profile`: User preference analysis
- `messages`: Ranking explanation

**Implementation**:

```python
# agents/nodes/preference.py
from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from agents.utils.formatters import format_ranking_summary
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger(__name__)

def preference_node(state: MedFlowState) -> Dict:
    """
    Preference Learning Agent - Rank strategies by user preferences.

    Uses hybrid ML system (40% RF + 30% LLM + 30% Vector):
    - Random Forest: Pattern recognition from past interactions
    - Groq/Llama 3.3 70B: Semantic analysis of preferences
    - Qdrant Vector DB: Similarity to past successful decisions

    Args:
        state: Current workflow state

    Returns:
        State updates with ranked strategies
    """
    logger.info("[Preference] Ranking strategies by user preferences")

    api_client = MedFlowAPIClient()

    # Call preference scoring API
    result = api_client.rank_strategies(
        user_id=state["user_id"],
        strategies=state["allocation_strategies"]
    )

    ranked = result["ranked_strategies"]
    profile = result.get("user_profile", {})

    # Generate summary
    summary = format_ranking_summary(ranked, profile)

    logger.info(
        f"[Preference] Top strategy: {ranked[0]['strategy_name']} "
        f"(score: {ranked[0]['preference_score']:.3f})"
    )
    logger.info(f"[Preference] User type: {profile.get('preference_type', 'unknown')}")

    return {
        "ranked_strategies": ranked,
        "preference_profile": profile,
        "messages": [AIMessage(content=summary)],
        "current_node": "preference"
    }
```

---

### Node 5: Reasoning Agent

**Purpose**: Generate natural language explanation using LLM

**File**: `agents/nodes/reasoning.py`

**LLM Used**: GPT-4 or Groq/Llama 3.3 70B

**Inputs from State**:
- `shortage_hospitals`: Current situation
- `ranked_strategies`: Top recommendations
- `preference_profile`: User preferences
- `active_outbreaks`: Context

**Outputs to State**:
- `final_recommendation`: Top strategy
- `explanation`: Natural language explanation
- `reasoning_chain`: Step-by-step reasoning

**Implementation**:

```python
# agents/nodes/reasoning.py
from typing import Dict
from agents.state import MedFlowState
from agents.prompts.reasoning_prompts import REASONING_SYSTEM_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import logging
import os

logger = logging.getLogger(__name__)

def reasoning_node(state: MedFlowState) -> Dict:
    """
    Reasoning Agent - Generate LLM explanation of recommendations.

    Uses GPT-4 to create:
    - Natural language explanation of top recommendation
    - Step-by-step reasoning chain
    - Contextualized insights based on outbreaks, user preferences

    Args:
        state: Current workflow state

    Returns:
        State updates with explanation
    """
    logger.info("[Reasoning] Generating LLM explanation")

    # Initialize LLM
    llm = ChatOpenAI(
        model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o"),
        temperature=float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.3"))
    )

    # Get top recommendation
    top_strategy = state["ranked_strategies"][0]

    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", REASONING_SYSTEM_PROMPT),
        ("human", build_reasoning_prompt(state))
    ])

    # Create chain
    chain = prompt | llm

    # Generate explanation
    response = chain.invoke({
        "shortage_count": state["shortage_count"],
        "resource_type": state["resource_type"],
        "top_strategy": top_strategy,
        "strategy_name": top_strategy["strategy_name"],
        "hospitals_helped": top_strategy["summary"]["hospitals_helped"],
        "total_cost": top_strategy["summary"]["total_cost"],
        "shortage_reduction": top_strategy["summary"]["shortage_reduction"],
        "preference_score": top_strategy["preference_score"],
        "preference_type": state["preference_profile"].get("preference_type", "unknown"),
        "active_outbreaks": len(state.get("active_outbreaks", [])),
        "llm_explanation": top_strategy.get("llm_explanation", "")
    })

    explanation = response.content

    logger.info(f"[Reasoning] Generated {len(explanation)} character explanation")

    return {
        "final_recommendation": top_strategy,
        "explanation": explanation,
        "reasoning_chain": explanation,  # Same for now, can add chain-of-thought later
        "messages": [AIMessage(content=explanation)],
        "current_node": "reasoning"
    }


def build_reasoning_prompt(state: MedFlowState) -> str:
    """Build the human prompt for reasoning"""
    top = state["ranked_strategies"][0]

    return f"""
You are an AI assistant helping a healthcare administrator make resource allocation decisions.

**Current Situation:**
- {state['shortage_count']} hospitals need {state['resource_type']}
- {len(state.get('active_outbreaks', []))} active outbreak(s)

**Top Recommended Strategy:**
- Name: {top['strategy_name']}
- Hospitals Helped: {top['summary']['hospitals_helped']}
- Total Cost: ${top['summary']['total_cost']:,.0f}
- Shortage Reduction: {top['summary']['shortage_reduction']:.1f}%
- Preference Score: {top['preference_score']:.3f}

**User Preference Profile:**
- Type: {state['preference_profile'].get('preference_type', 'unknown')}
- Confidence: {state['preference_profile'].get('confidence', 0):.0%}

**Preference Learning Explanation:**
{top.get('llm_explanation', 'No preference explanation available.')}

**Task:**
Generate a clear, actionable explanation of why this strategy is recommended.

**Format:**
1. Situation Summary (2-3 sentences)
2. Why This Strategy (3-4 bullet points)
3. Key Tradeoffs (2-3 sentences)
4. Next Steps (what user should review)

Be concise, specific, and data-driven. Use the preference learning explanation to personalize.
""".strip()
```

**Prompt Template**:

```python
# agents/prompts/reasoning_prompts.py
REASONING_SYSTEM_PROMPT = """
You are a senior healthcare operations analyst with expertise in resource allocation,
medical supply chain optimization, and crisis management.

Your role is to explain resource allocation recommendations in a clear, actionable way
that healthcare administrators can understand and act upon.

Guidelines:
- Be specific with numbers and metrics
- Explain tradeoffs clearly
- Consider both short-term urgency and long-term efficiency
- Acknowledge uncertainty where appropriate
- Tailor explanations to user preferences when available
- Keep explanations concise (200-300 words)

Always structure your response with:
1. Situation summary
2. Recommendation rationale
3. Key tradeoffs
4. Suggested next steps
""".strip()
```

---

### Node 6: Human Review (HITL)

**Purpose**: Allow user to review recommendations and provide feedback

**File**: `agents/nodes/human_review.py`

**Inputs from State**:
- `ranked_strategies`: Top 3 recommendations

**Outputs to State**:
- `user_decision`: Selected strategy index
- `user_feedback`: Optional text feedback

**Implementation**:

```python
# agents/nodes/human_review.py
from typing import Dict
from agents.state import MedFlowState
from agents.utils.formatters import display_recommendations_table
from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator, ValidationError
from rich.console import Console
from rich.panel import Panel
import logging

logger = logging.getLogger(__name__)
console = Console()

class StrategyIndexValidator(Validator):
    """Validate user input is a valid strategy index"""

    def __init__(self, max_index: int):
        self.max_index = max_index

    def validate(self, document):
        text = document.text

        if not text.isdigit():
            raise ValidationError(
                message="Please enter a number",
                cursor_position=len(text)
            )

        index = int(text)
        if index < 0 or index >= self.max_index:
            raise ValidationError(
                message=f"Please enter a number between 0 and {self.max_index - 1}",
                cursor_position=len(text)
            )


def human_review_node(state: MedFlowState) -> Dict:
    """
    Human-in-the-Loop Review Node.

    Displays top recommendations and prompts user to:
    1. Select a strategy to execute
    2. Optionally provide feedback

    Args:
        state: Current workflow state

    Returns:
        State updates with user decision
    """
    logger.info("[Human Review] Awaiting user decision")

    # Display recommendations
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🤖 AI Recommendations Ready for Review[/bold cyan]",
        border_style="cyan"
    ))

    # Show top 3 strategies in a table
    ranked = state["ranked_strategies"][:3]
    display_recommendations_table(ranked)

    # Show explanation
    console.print("\n")
    console.print(Panel(
        state["explanation"],
        title="[bold green]💡 Recommendation Explanation[/bold green]",
        border_style="green"
    ))

    # Prompt for selection
    console.print("\n")
    validator = StrategyIndexValidator(max_index=len(ranked))

    selected_index = int(prompt(
        "Select a strategy (enter index 0-2): ",
        validator=validator
    ))

    selected_strategy = ranked[selected_index]

    console.print(f"\n✅ Selected: {selected_strategy['strategy_name']}")

    # Optional feedback
    console.print("\n")
    feedback = prompt(
        "Optional feedback (press Enter to skip): ",
        default=""
    )

    logger.info(
        f"[Human Review] User selected strategy {selected_index}: "
        f"{selected_strategy['strategy_name']}"
    )

    if feedback:
        logger.info(f"[Human Review] User feedback: {feedback}")

    return {
        "user_decision": selected_index,
        "user_feedback": feedback if feedback else None,
        "current_node": "human_review"
    }
```

---

### Node 7: Feedback/Update Preferences

**Purpose**: Learn from user decision to improve future recommendations

**File**: `agents/nodes/feedback.py`

**API Endpoints Used**:
- `POST /api/v1/preferences/update`

**Inputs from State**:
- `user_decision`: Selected strategy index
- `ranked_strategies`: All strategies shown
- `user_feedback`: Optional text feedback
- `user_id`: User identifier

**Outputs to State**:
- `feedback_stored`: bool
- `messages`: Confirmation message

**Implementation**:

```python
# agents/nodes/feedback.py
from typing import Dict
from agents.state import MedFlowState
from agents.tools.api_client import MedFlowAPIClient
from langchain_core.messages import AIMessage
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def feedback_node(state: MedFlowState) -> Dict:
    """
    Feedback Node - Update preference learning from user decision.

    Sends interaction data to preference learning API to:
    - Update Random Forest model with new training sample
    - Store interaction in Qdrant vector database
    - Improve future recommendations for this user

    Args:
        state: Current workflow state

    Returns:
        State updates confirming feedback stored
    """
    logger.info("[Feedback] Updating user preferences")

    api_client = MedFlowAPIClient()

    # Build interaction object
    interaction = {
        "selected_recommendation_index": state["user_decision"],
        "recommendations": state["ranked_strategies"],
        "timestamp": datetime.now().isoformat(),
        "feedback_text": state.get("user_feedback"),
        "context": {
            "resource_type": state["resource_type"],
            "shortage_count": state["shortage_count"],
            "session_id": state["session_id"]
        }
    }

    # Call preference update API
    try:
        result = api_client.update_preferences(
            user_id=state["user_id"],
            interaction=interaction
        )

        feedback_stored = result.get("success", False)
        message = result.get("message", "Preferences updated")

        logger.info(f"[Feedback] Update result: {message}")

    except Exception as e:
        logger.error(f"[Feedback] Failed to update preferences: {e}")
        feedback_stored = False
        message = f"Failed to update preferences: {str(e)}"

    return {
        "feedback_stored": feedback_stored,
        "messages": [AIMessage(content=message)],
        "current_node": "feedback"
    }
```

---

## 🎯 LangGraph Workflow Implementation

### Main Graph Builder

**File**: `agents/graph.py`

```python
# agents/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.state import MedFlowState
from agents.nodes.data_analyst import data_analyst_node
from agents.nodes.forecasting import forecasting_node
from agents.nodes.optimization import optimization_node
from agents.nodes.preference import preference_node
from agents.nodes.reasoning import reasoning_node
from agents.nodes.human_review import human_review_node
from agents.nodes.feedback import feedback_node
from agents.utils.routing import (
    route_after_analysis,
    route_after_optimization,
    route_after_human_review
)
import logging

logger = logging.getLogger(__name__)

def build_medflow_graph() -> StateGraph:
    """
    Build the MedFlow LangGraph workflow.

    Returns:
        Compiled StateGraph with checkpointing enabled
    """
    logger.info("Building MedFlow agent graph")

    # Initialize StateGraph with our schema
    builder = StateGraph(MedFlowState)

    # ========================================================================
    # ADD NODES
    # ========================================================================
    builder.add_node("data_analyst", data_analyst_node)
    builder.add_node("forecasting", forecasting_node)
    builder.add_node("optimization", optimization_node)
    builder.add_node("preference", preference_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("feedback", feedback_node)

    # ========================================================================
    # ADD EDGES - Define workflow structure
    # ========================================================================

    # Entry point
    builder.add_edge(START, "data_analyst")

    # Conditional: After data analysis, check if shortages exist
    builder.add_conditional_edges(
        "data_analyst",
        route_after_analysis,
        {
            "forecasting": "forecasting",  # Has shortages → predict demand
            "END": END                      # No shortages → done
        }
    )

    # Linear: Forecasting → Optimization
    builder.add_edge("forecasting", "optimization")

    # Conditional: After optimization, check if we have feasible strategies
    builder.add_conditional_edges(
        "optimization",
        route_after_optimization,
        {
            "preference": "preference",  # Has strategies → rank them
            "END": END                    # No feasible strategies → done
        }
    )

    # Linear: Preference → Reasoning → Human Review
    builder.add_edge("preference", "reasoning")
    builder.add_edge("reasoning", "human_review")

    # Conditional: After human review, check if user wants to give feedback
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "feedback": "feedback",  # User gave feedback → update preferences
            "END": END               # No feedback → done
        }
    )

    # Linear: Feedback → END
    builder.add_edge("feedback", END)

    # ========================================================================
    # COMPILE WITH CHECKPOINTING
    # ========================================================================

    # Initialize SQLite checkpointer for state persistence
    checkpointer = SqliteSaver.from_conn_string("agents/checkpoints/workflows.db")

    # Compile graph
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("MedFlow agent graph compiled successfully")

    return graph


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_optimization(state: MedFlowState) -> str:
    """
    Route after optimization based on strategy availability.

    Logic:
    - If strategies generated and at least one is optimal/feasible → preference
    - If no strategies or all infeasible → END
    """
    strategies = state.get("allocation_strategies", [])

    if not strategies:
        logger.info("[Router] No strategies generated. Ending workflow.")
        return "END"

    # Check if any strategy is optimal or feasible
    feasible = [s for s in strategies if s["status"] in ["optimal", "feasible"]]

    if not feasible:
        logger.info("[Router] No feasible strategies. Ending workflow.")
        return "END"
    else:
        logger.info(f"[Router] {len(feasible)} feasible strategies. Proceeding to preference ranking.")
        return "preference"


def route_after_human_review(state: MedFlowState) -> str:
    """
    Route after human review based on feedback availability.

    Logic:
    - If user provided decision and wants to give feedback → feedback
    - Otherwise → END
    """
    user_decision = state.get("user_decision")
    user_feedback = state.get("user_feedback")

    # Always update preferences if user made a selection
    # (even without explicit feedback, the selection itself is valuable data)
    if user_decision is not None:
        logger.info("[Router] User made selection. Updating preferences.")
        return "feedback"
    else:
        logger.info("[Router] No user decision. Ending workflow.")
        return "END"


# ============================================================================
# EXPORT
# ============================================================================

# Create singleton graph instance
medflow_graph = build_medflow_graph()
```

---

## 🔧 API Client & Tools

### FastAPI Client with Retry Logic

**File**: `agents/tools/api_client.py`

```python
# agents/tools/api_client.py
import os
import httpx
from typing import Optional, Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

logger = logging.getLogger(__name__)

class MedFlowAPIClient:
    """
    Client for MedFlow FastAPI backend with retry logic.

    Features:
    - Automatic retries with exponential backoff
    - Timeout handling
    - Error logging
    - Type hints for IDE autocomplete
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        self.base_url = base_url or os.getenv("MEDFLOW_API_BASE", "http://localhost:8000")
        self.api_key = api_key or os.getenv("MEDFLOW_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

        # HTTP client with timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"X-API-Key": self.api_key} if self.api_key else {}
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
        risk_level: Optional[str] = None
    ) -> Dict:
        """
        Detect hospitals with shortage risks.

        Args:
            resource_type: Filter by resource (ppe, ventilators, etc.)
            risk_level: Filter by risk (critical, high, medium, low)

        Returns:
            {
                "shortages": [...],
                "count": int,
                "summary": {...}
            }
        """
        params = {}
        if resource_type:
            params["resource_type"] = resource_type
        if risk_level:
            params["risk_level"] = risk_level

        logger.debug(f"GET /api/v1/shortages with params: {params}")

        response = self.client.get("/api/v1/shortages", params=params)
        response.raise_for_status()

        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def get_active_outbreaks(self) -> Dict:
        """
        Get currently active outbreak events.

        Returns:
            {
                "active_outbreaks": [...],
                "count": int
            }
        """
        logger.debug("GET /api/v1/outbreaks/active")

        response = self.client.get("/api/v1/outbreaks/active")
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
        forecast_days: int = 14,
        probabilistic: bool = True
    ) -> Dict:
        """
        Predict future demand using LSTM forecasting.

        Args:
            hospital_id: Hospital UUID
            resource_type: Resource to predict
            forecast_days: Days to forecast (default 14)
            probabilistic: Include P10-P90 intervals

        Returns:
            {
                "predictions": {
                    "point_forecast": [...],
                    "lower_bound": [...],
                    "upper_bound": [...]
                },
                "metadata": {...}
            }
        """
        payload = {
            "hospital_id": hospital_id,
            "resource_type": resource_type,
            "forecast_days": forecast_days,
            "probabilistic": probabilistic
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
        limit: Optional[int] = None
    ) -> Dict:
        """
        Generate allocation strategies using Linear Programming.

        Args:
            resource_type: Resource to allocate
            n_strategies: Number of strategies (default 3)
            limit: Max hospitals to consider

        Returns:
            {
                "strategies": [...],
                "count": int,
                "computation_time": float
            }
        """
        payload = {
            "resource_type": resource_type,
            "n_strategies": n_strategies
        }
        if limit:
            payload["limit"] = limit

        logger.debug(f"POST /api/v1/strategies for {resource_type}")

        response = self.client.post("/api/v1/strategies", json=payload)
        response.raise_for_status()

        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    def rank_strategies(
        self,
        user_id: str,
        strategies: List[Dict]
    ) -> Dict:
        """
        Rank strategies by learned user preferences.

        Uses hybrid ML (40% RF + 30% LLM + 30% Vector).

        Args:
            user_id: User identifier
            strategies: List of strategies to rank

        Returns:
            {
                "ranked_strategies": [...],
                "user_profile": {...}
            }
        """
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
    def update_preferences(
        self,
        user_id: str,
        interaction: Dict
    ) -> Dict:
        """
        Update preference model from user interaction.

        Args:
            user_id: User identifier
            interaction: {
                "selected_recommendation_index": int,
                "recommendations": [...],
                "timestamp": str,
                "feedback_text": Optional[str]
            }

        Returns:
            {
                "success": bool,
                "message": str
            }
        """
        payload = {
            "user_id": user_id,
            "interaction": interaction
        }

        logger.debug(f"POST /api/v1/preferences/update for user {user_id}")

        response = self.client.post("/api/v1/preferences/update", json=payload)
        response.raise_for_status()

        return response.json()

    def close(self):
        """Close HTTP client"""
        self.client.close()
```

---

## 🖥️ CLI Interface

### Main CLI App

**File**: `cli/main.py`

```python
# cli/main.py
import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from agents.graph import medflow_graph
from agents.state import MedFlowState
from datetime import datetime
import uuid
import logging
import os

# Initialize Typer app
app = typer.Typer(
    name="medflow",
    help="🏥 MedFlow AI - Intelligent Resource Allocation Assistant",
    add_completion=False
)

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

@app.command()
def allocate(
    resource_type: str = typer.Option(
        "ventilators",
        "--resource",
        "-r",
        help="Resource type to allocate (ppe, ventilators, o2, beds, medications)"
    ),
    user_id: str = typer.Option(
        "default_user",
        "--user",
        "-u",
        help="User ID for preference learning"
    ),
    session_id: str = typer.Option(
        None,
        "--session",
        "-s",
        help="Session ID (auto-generated if not provided)"
    )
):
    """
    🚀 Run resource allocation workflow

    This command executes the full 5-agent workflow:
    1. Data Analyst - Assess current shortages
    2. Forecasting - Predict 14-day demand
    3. Optimization - Generate 3 allocation strategies
    4. Preference - Rank by learned preferences
    5. Reasoning - Generate LLM explanation
    6. Human Review - You select a strategy
    7. Feedback - System learns from your decision
    """
    # Display header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🏥 MedFlow AI - Resource Allocation Assistant[/bold cyan]\n"
        "[dim]Powered by LangGraph • Phase 5[/dim]",
        border_style="cyan"
    ))

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Build initial state
    initial_state: MedFlowState = {
        "resource_type": resource_type,
        "user_id": user_id,
        "session_id": session_id,
        "messages": [],
        "workflow_status": "pending",
        "timestamp": datetime.now().isoformat(),
        "current_node": None,
        "shortage_count": 0,
        "shortage_hospitals": [],
        "active_outbreaks": [],
        "analysis_summary": "",
        "demand_forecasts": {},
        "forecast_summary": "",
        "allocation_strategies": [],
        "strategy_count": 0,
        "ranked_strategies": [],
        "preference_profile": {},
        "final_recommendation": {},
        "explanation": "",
        "reasoning_chain": "",
        "user_decision": None,
        "user_feedback": None,
        "feedback_stored": False,
        "error": None,
        "execution_time_seconds": None
    }

    # Display config
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Resource Type: [cyan]{resource_type}[/cyan]")
    console.print(f"  User ID: [cyan]{user_id}[/cyan]")
    console.print(f"  Session ID: [dim]{session_id}[/dim]")
    console.print()

    # Run workflow with checkpointing
    try:
        with console.status("[bold green]Running workflow...", spinner="dots"):
            start_time = datetime.now()

            result = medflow_graph.invoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": user_id  # Use user_id for checkpointing
                    }
                }
            )

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

        # Display success
        console.print(f"\n✅ [bold green]Workflow completed![/bold green]")
        console.print(f"   Execution time: {execution_time:.1f}s")

        # Display final recommendation
        if result.get("final_recommendation"):
            rec = result["final_recommendation"]
            console.print(f"\n[bold]Selected Strategy:[/bold] {rec['strategy_name']}")
            console.print(f"   Preference Score: {rec['preference_score']:.3f}")
            console.print(f"   Hospitals Helped: {rec['summary']['hospitals_helped']}")
            console.print(f"   Total Cost: ${rec['summary']['total_cost']:,.0f}")
            console.print(f"   Shortage Reduction: {rec['summary']['shortage_reduction']:.1f}%")

        # Display feedback confirmation
        if result.get("feedback_stored"):
            console.print(f"\n💾 [green]Preferences updated successfully![/green]")

    except Exception as e:
        console.print(f"\n❌ [bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def history(
    user_id: str = typer.Argument(..., help="User ID to view history for"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent sessions to show")
):
    """
    📜 View decision history for a user
    """
    console.print(f"\n[bold]Decision History for {user_id}[/bold]\n")
    console.print("[dim]Feature coming in Phase 5.2[/dim]\n")


@app.command()
def preferences(
    user_id: str = typer.Argument(..., help="User ID to view preferences for")
):
    """
    🧠 View learned preferences for a user
    """
    console.print(f"\n[bold]Learned Preferences for {user_id}[/bold]\n")
    console.print("[dim]Feature coming in Phase 5.2[/dim]\n")


@app.command()
def version():
    """
    📌 Show version information
    """
    console.print("\n[bold cyan]MedFlow AI[/bold cyan]")
    console.print("  Phase: [green]5.0[/green]")
    console.print("  LangGraph: >=0.2.28")
    console.print("  Backend: http://localhost:8000")
    console.print()


if __name__ == "__main__":
    app()
```

---

## ✅ Implementation Checklist

This comprehensive plan is ready for implementation. Follow the steps sequentially:

### Phase 5.1: Core Setup (Week 1)

**Day 1-2: Project Setup**
- [ ] Create directory structure (`agents/`, `cli/`, `dashboard/`)
- [ ] Update `requirements.txt` with new dependencies
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Set up `.env` with `OPENAI_API_KEY`, `GROQ_API_KEY`
- [ ] Create `agents/checkpoints/` directory

**Day 3-4: State & API Client**
- [ ] Implement `agents/state.py` with `MedFlowState` TypedDict
- [ ] Implement `agents/tools/api_client.py` with retry logic
- [ ] Write tests for API client (`agents/tests/test_tools.py`)
- [ ] Verify API connectivity

**Day 5-7: Agent Nodes**
- [ ] Implement `data_analyst_node` in `agents/nodes/data_analyst.py`
- [ ] Implement `forecasting_node` in `agents/nodes/forecasting.py`
- [ ] Implement `optimization_node` in `agents/nodes/optimization.py`
- [ ] Implement `preference_node` in `agents/nodes/preference.py`
- [ ] Implement `reasoning_node` in `agents/nodes/reasoning.py`
- [ ] Implement `human_review_node` in `agents/nodes/human_review.py`
- [ ] Implement `feedback_node` in `agents/nodes/feedback.py`
- [ ] Write unit tests for each node

### Phase 5.2: Workflow & CLI (Week 2)

**Day 8-10: LangGraph Workflow**
- [ ] Implement `agents/graph.py` with StateGraph
- [ ] Add all 7 nodes to graph
- [ ] Configure conditional routing logic
- [ ] Set up SQLite checkpointer
- [ ] Test end-to-end workflow
- [ ] Write integration tests (`agents/tests/test_graph.py`)

**Day 11-13: CLI Interface**
- [ ] Implement `cli/main.py` with Typer
- [ ] Create Rich UI components in `cli/ui/`
- [ ] Implement `allocate` command
- [ ] Implement `history` command stub
- [ ] Implement `preferences` command stub
- [ ] Test CLI flows

**Day 14: Refinement**
- [ ] Add error handling and graceful degradation
- [ ] Improve logging
- [ ] Add progress indicators
- [ ] Polish CLI output

### Phase 5.3: Testing & Documentation (Week 3)

**Day 15-17: Testing**
- [ ] Write comprehensive unit tests (target 80% coverage)
- [ ] Write integration tests
- [ ] Write end-to-end tests
- [ ] Test error scenarios
- [ ] Test with real API

**Day 18-20: Documentation**
- [ ] Create `docs/PHASE_5_DOCUMENTATION.md`
- [ ] Create `docs/AGENT_ARCHITECTURE.md`
- [ ] Create `docs/CLI_GUIDE.md`
- [ ] Update main `README.md`
- [ ] Add inline code documentation
- [ ] Create usage examples

**Day 21: Final Review**
- [ ] Code review
- [ ] Performance testing
- [ ] Bug fixes
- [ ] Deployment preparation

### Phase 5.4: Optional Enhancements (Week 4+)

- [ ] Implement Streamlit dashboard
- [ ] Add LangSmith monitoring
- [ ] Implement decision history viewer
- [ ] Add preference analytics dashboard
- [ ] Multi-user support
- [ ] Docker containerization

---

## 🎯 Success Metrics

### Functional Requirements ✅
- All 7 agent nodes implemented and functional
- State flows correctly through entire graph
- Conditional routing works as expected
- HITL interaction is smooth and intuitive
- Preference learning feedback loop operational
- State persistence working across sessions

### Performance Requirements ⚡
- Workflow completion < 60s (excluding human review wait time)
- API calls have retry logic with exponential backoff
- Graceful degradation if any single node fails
- No memory leaks during long-running sessions

### Quality Requirements 📊
- Unit test coverage > 80%
- All integration tests passing
- At least 5 end-to-end test scenarios passing
- Documentation complete and accurate
- Code follows PEP 8 style guide

---

## 📚 Additional Resources

### Learning Materials
- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangChain Documentation**: https://python.langchain.com/
- **Typer Documentation**: https://typer.tiangolo.com/
- **Rich Documentation**: https://rich.readthedocs.io/

### Phase 4 References
- `docs/PHASE_4_DOCUMENTATION.md` - Backend API overview
- `docs/API_OVERVIEW.md` - Endpoint reference
- `docs/AGENT_WORKFLOW.md` - Agent design patterns
- `docs/INTEGRATION_EXAMPLES.md` - Code examples

---

## 🚀 Getting Started

```bash
# 1. Ensure Phase 4 backend is running
cd backend
uvicorn app.main:app --reload --port 8000

# 2. In a new terminal, install Phase 5 dependencies
cd /home/user/MedFlow
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add OPENAI_API_KEY, GROQ_API_KEY

# 4. Run your first workflow
python -m cli.main allocate --resource ventilators --user john_doe

# 5. View help
python -m cli.main --help
```

---

## 📝 Notes

1. **Demand Forecasting Fixed**: The 17-feature engineering issue is resolved. All 5 agent nodes will be functional.

2. **LLM Choice**: Default is GPT-4 for reasoning agent, but can easily switch to Groq/Llama 3.3 70B by setting `DEFAULT_LLM_MODEL=llama-3.3-70b-versatile` in `.env`.

3. **State Persistence**: SQLite checkpointer stores state in `agents/checkpoints/workflows.db`. Can resume interrupted workflows using the same `thread_id`.

4. **Monitoring**: Optional LangSmith integration for debugging and performance monitoring.

5. **Extensibility**: Architecture supports adding new nodes (e.g., approval routing, notification system) without major refactoring.

---

**Phase 5 Status**: 📋 **PLANNED - Ready for Implementation**

**Estimated Timeline**: 3-4 weeks (part-time) or 1-2 weeks (full-time)

**Next Step**: Begin implementation with Phase 5.1 setup tasks.
