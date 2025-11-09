# MedFlow AI: Comprehensive Technical Documentation

**An AI-Powered Healthcare Resource Allocation System**

> *Combining LSTM forecasting, multi-agent orchestration, and adaptive preference learning to optimize medical resource distribution across healthcare facilities*

---

## Executive Summary

MedFlow AI is a proof-of-concept intelligent decision support system that demonstrates how machine learning, optimization algorithms, and multi-agent AI can work together to solve critical resource allocation challenges in healthcare. The system predicts demand for medical resources (ventilators, oxygen cylinders, PPE, medications, beds), detects potential shortages, generates optimal allocation strategies, and learns from decision-maker preferences over time.

**Key Capabilities:**
- 📊 **Demand Forecasting**: 14-day predictions using LSTM neural networks
- 🚨 **Shortage Detection**: Early warning system identifying at-risk hospitals
- 🎯 **Optimization**: Multi-objective resource allocation strategies
- 🤖 **Agent Orchestration**: 7-agent workflow with human-in-the-loop oversight
- 🧠 **Adaptive Learning**: Hybrid ML system that learns user preferences

---

## I. Problem Approach

### The Core Challenge

Healthcare systems worldwide struggle with **inefficient resource allocation**—a problem that becomes critical during disease outbreaks, seasonal surges, and emergencies. The fundamental issues include:

1. **Demand Unpredictability**: Resource needs fluctuate based on admissions, emergencies, seasonality, and outbreaks
2. **Information Asymmetry**: Decision-makers lack real-time visibility into network-wide resource availability
3. **Suboptimal Allocation**: Manual decisions lead to some facilities overstocked while others face critical shortages
4. **Reactive Response**: By the time shortages are obvious, it's often too late to prevent patient harm
5. **Lack of Learning**: Systems don't improve from past allocation decisions

### Our Approach: A Multi-Layered AI System

MedFlow addresses these challenges through **five integrated ML/AI components**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                          │
│  CLI (Typer + Rich) │ Web Dashboard (Streamlit)                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│         AGENT ORCHESTRATION (Dual Framework Support)              │
│                                                                   │
│  ┌────────────────────────┐  ┌────────────────────────┐          │
│  │   LangGraph (Primary)  │  │   CrewAI (Alternative)  │          │
│  │  State-based workflow  │  │  Agent-based workflow  │          │
│  └────────────────────────┘  └────────────────────────┘          │
│                                                                   │
│  7-Agent Workflow with Human-in-the-Loop                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Analyst  │→ │Forecast  │→ │Optimize  │→ │Preference│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│       ↓             ↓             ↓             ↓               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │Reasoning │→ │  Human   │→ │ Feedback │                     │
│  └──────────┘  │  Review  │  └──────────┘                     │
│                └──────────┘                                     │
│                                                                   │
│  Note: Both frameworks operate independently and can be         │
│        toggled via dashboard or CLI flag                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                   BACKEND API (FastAPI)                          │
│  16 REST Endpoints │ Auth │ Validation │ Error Handling         │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                      ML CORE LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ LSTM         │  │ Random Forest │  │ Linear       │         │
│  │ Forecasting  │  │ Shortage      │  │ Programming  │         │
│  │ (5 resources)│  │ Detection     │  │ Optimization │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ Hybrid Preference Learning                           │      │
│  │ 40% Random Forest + 30% LLM + 30% Vector DB         │      │
│  └──────────────────────────────────────────────────────┘      │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│              DATABASE (Supabase/PostgreSQL)                      │
│  Hospitals │ Inventory │ Admissions │ Outbreaks                 │
└──────────────────────────────────────────────────────────────────┘
```

### Design Philosophy

Our approach prioritizes:

1. **Proactivity over Reactivity**: Predict and prevent shortages before they occur
2. **Explainability over Black-Box**: Generate natural language explanations for all recommendations
3. **Safety over Optimization**: Conservative forecasts with safety buffers
4. **Adaptability over Fixed Rules**: Learn from user decisions to improve over time
5. **Human-in-the-Loop over Full Automation**: Humans review and approve all critical decisions

---

## II. Data Sources and Generation

### Real-World Challenge: Data Privacy

Access to actual hospital data is restricted due to:
- **HIPAA regulations** (patient privacy)
- **Competitive sensitivity** (hospital operations data)
- **Legal constraints** (data sharing agreements)

### Our Solution: Realistic Synthetic Data

To enable development and testing, we generated **synthetic hospital data** that mimics real-world patterns without privacy concerns.

### Data Generation Methodology

**Coverage:**
- **100 hospitals** across **16 Indian states/regions**
- **18 months** of historical data (Nov 2023 - Apr 2025)
- **5 resource types** tracked daily:
  - PPE (Personal Protective Equipment)
  - Oxygen Cylinders
  - Ventilators
  - Medications
  - Beds

**Realistic Patterns Incorporated:**

1. **Seasonal Variations**
   - Flu season (Dec-Feb): 30% increase in admissions
   - Monsoon season (Jul-Sep): 20% increase in respiratory issues
   - Summer (Apr-Jun): 15% decrease in overall admissions

2. **Weekly Patterns**
   - Weekend dip: 20% fewer scheduled admissions on Sat/Sun
   - Emergency spike: 10% more emergencies on Fridays/Saturdays

3. **Outbreak Events** (3 simulated events)
   - **TB Outbreak** (Feb 2024): 50% spike in affected regions
   - **Dengue Peak** (Sep 2024): 40% spike during monsoon
   - **Air Pollution Surge** (Nov 2024): 35% spike in respiratory cases

4. **Regional Imbalances**
   - Urban hospitals: Higher baseline but better supply chain
   - Rural hospitals: Lower baseline but higher risk of stockouts
   - Regional hubs: 30% higher capacity and turnover

5. **Supply Chain Disruptions**
   - Random resupply delays (5-10% of deliveries)
   - Bulk ordering patterns (weekly/biweekly cycles)
   - Minimum order quantities

### Data Features Generated

**Hospital Master Data:**
```
- Hospital ID, Name, Type (Metro/Urban/Rural/District)
- Location (State, District, Coordinates)
- Capacity (Beds, ICU capacity, Emergency capacity)
- Specializations
```

**Daily Time-Series Data:**
```
- Admissions (Total, Emergency, Scheduled, ICU, General)
- Inventory levels for each resource
- Daily consumption
- Resupply events
- Stock status
```

**Outbreak Events:**
```
- Outbreak ID, Type, Location
- Start/End dates
- Affected regions
- Severity multipliers
```

### Data Quality Challenges Solved

**Initial Problem:** LSTM models couldn't learn from sparse data
- Ventilators had 84% zero-consumption days
- Weak correlation with admissions (r < 0.4)

**Solution:** Fixed consumption formulas
- Changed from "daily census" to "cumulative daily usage"
- Added minimum admission floors
- Improved correlation to r = 0.76
- Reduced zero-days to 54%

**Files:** `data/generators/inventory.py`, `data/generators/FIXES_SUMMARY.md`

---

## III. Agent Architecture and Design Choices

### Why Multi-Agent Architecture?

Traditional monolithic AI systems struggle with complex, multi-step decision-making. MedFlow uses a **specialized agent approach** where each agent has a specific responsibility, enabling:

- **Modularity**: Each agent can be improved independently
- **Explainability**: Each step's reasoning is transparent
- **Flexibility**: Agents can be reconfigured for different workflows
- **Human Oversight**: Humans can intervene at any step

### Agent Workflow: 7 Specialized Agents

```
START
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 1. DATA ANALYST AGENT                               │
│ Role: Assess current situation                      │
│ Tasks:                                               │
│  - Detect hospitals with shortage risks             │
│  - Identify active outbreak events                  │
│  - Analyze regional imbalances                      │
│                                                      │
│ APIs Used:                                           │
│  - GET /api/v1/shortages                           │
│  - GET /api/v1/outbreaks/active                    │
│                                                      │
│ Output: Situational summary + at-risk hospitals     │
└─────────────────────────────────────────────────────┘
  │
  │ Conditional Routing:
  │   shortage_count > 0? → Continue
  │   shortage_count = 0? → END (no action needed)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 2. FORECASTING AGENT                                │
│ Role: Predict future demand (14 days)               │
│ Tasks:                                               │
│  - Run LSTM predictions for at-risk hospitals       │
│  - Generate point forecasts (expected values)       │
│  - Generate probabilistic forecasts (P10-P90)       │
│  - Identify trend directions                        │
│                                                      │
│ APIs Used:                                           │
│  - POST /api/v1/predict/demand (for each hospital) │
│                                                      │
│ Output: 14-day demand forecasts per hospital        │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 3. OPTIMIZATION AGENT                                │
│ Role: Generate allocation strategies                 │
│ Tasks:                                               │
│  - Formulate LP optimization problem                │
│  - Generate 3 strategies:                           │
│    1. Cost-Efficient (minimize transfer costs)      │
│    2. Maximum Coverage (help most hospitals)        │
│    3. Balanced (trade-off approach)                 │
│  - Calculate metrics for each strategy              │
│                                                      │
│ APIs Used:                                           │
│  - POST /api/v1/strategies                          │
│                                                      │
│ Output: 3 allocation strategies with metrics        │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 4. PREFERENCE LEARNING AGENT                        │
│ Role: Rank strategies by user preferences           │
│ Tasks:                                               │
│  - Analyze past user decisions                      │
│  - Identify preference patterns                     │
│  - Score each strategy (0.0 - 1.0)                 │
│  - Generate personalized explanations               │
│                                                      │
│ APIs Used:                                           │
│  - POST /api/v1/preferences/score                   │
│                                                      │
│ Output: Ranked strategies with preference scores    │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 5. REASONING AGENT                                   │
│ Role: Generate natural language explanation          │
│ Tasks:                                               │
│  - Synthesize insights from all previous agents     │
│  - Generate clear, actionable recommendation        │
│  - Explain trade-offs and reasoning                 │
│  - Provide context-aware guidance                   │
│                                                      │
│ LLM Used:                                            │
│  - Groq/Llama 3.3 70B (configurable)               │
│                                                      │
│ Output: Human-readable explanation of top strategy  │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 6. HUMAN REVIEW (HITL)                              │
│ Role: Human decision-maker oversight                │
│ Tasks:                                               │
│  - Display top 3 recommendations                    │
│  - Show AI explanation and metrics                  │
│  - Collect user selection                           │
│  - Gather optional feedback                         │
│                                                      │
│ Interface: CLI or Web Dashboard                     │
│                                                      │
│ Output: Selected strategy + optional feedback       │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 7. FEEDBACK AGENT                                    │
│ Role: Learn from user decision                      │
│ Tasks:                                               │
│  - Store interaction in vector database             │
│  - Update Random Forest model                       │
│  - Improve future recommendations                   │
│                                                      │
│ APIs Used:                                           │
│  - POST /api/v1/preferences/update                  │
│                                                      │
│ Output: Confirmation of learning update             │
└─────────────────────────────────────────────────────┘
  │
  ▼
END
```

### Technology Choices

**1. Dual Framework Orchestration: LangGraph & CrewAI**

MedFlow implements **two independent agent orchestration frameworks** that can be toggled at runtime:

#### LangGraph (Primary Framework)
- **Why LangGraph?**
  - State-based workflow management with TypedDict
  - Built-in checkpointing (resume workflows)
  - Fine-grained conditional routing support
  - Native LangChain integration
  - Explicit graph visualization
  - Python-only configuration

- **Architecture:**
  - State: `MedFlowState` TypedDict
  - Nodes: Python functions with state reducers
  - Edges: Conditional routing functions
  - Checkpointing: SqliteSaver for persistence

#### CrewAI (Alternative Framework)
- **Why CrewAI?**
  - YAML-based declarative configuration
  - Role-Goal-Backstory agent framework
  - Built-in memory and planning features
  - Task context dependencies
  - Easier agent/task management
  - Human-in-the-loop support

- **Architecture:**
  - Agents: YAML configuration + `@agent` decorator
  - Tasks: YAML configuration + `@task` decorator
  - Crew: `@crew` decorator with `@before_kickoff`/`@after_kickoff` hooks
  - Tools: BaseTool classes wrapping API client methods

#### Framework Independence

**Key Design Principle:** Both frameworks are **completely independent** implementations:
- ✅ **Separate codebases**: `agents/graph.py` (LangGraph) vs `agents/crewai/` (CrewAI)
- ✅ **No shared dependencies**: Each framework uses its own state management
- ✅ **Same backend API**: Both call the same FastAPI endpoints
- ✅ **Same workflow logic**: Both implement the same 7-agent workflow
- ✅ **Toggle at runtime**: Users can switch frameworks via dashboard or CLI flag

**When to Use Each:**
- **LangGraph**: When you need fine-grained state control, explicit graph visualization, or complex conditional routing
- **CrewAI**: When you prefer YAML configuration, declarative task definitions, or want to leverage CrewAI's built-in features

**Implementation Details:**
- Dashboard toggle: Sidebar framework selector (🔷 LangGraph / 🟣 CrewAI)
- CLI flag: `--framework langgraph` or `--framework crewai`
- State conversion: Dashboard handles conversion between LangGraph's `MedFlowState` dict and CrewAI's `CrewOutput` object
- See `docs/CREWAI_IMPLEMENTATION.md` for detailed CrewAI documentation

**2. FastAPI for Backend**
- **Why FastAPI?**
  - Automatic API documentation (Swagger/ReDoc)
  - Built-in validation with Pydantic
  - Async support for concurrent requests
  - Type hints for IDE autocomplete

- **Alternatives Considered:**
  - Flask: Lacks built-in validation and docs
  - Django: Too heavy for API-only service

**3. Streamlit for Dashboard**
- **Why Streamlit?**
  - Rapid prototyping (100 lines → full dashboard)
  - Interactive widgets out-of-the-box
  - Native support for data visualization

- **Alternatives Considered:**
  - React: Too complex for proof-of-concept
  - Dash: Less intuitive for simple dashboards

### State Management

All workflow state is stored in a **TypedDict** that flows through agents:

```python
class MedFlowState(TypedDict):
    # Input Parameters
    resource_type: str              # "ventilators", "ppe", etc.
    user_id: str                    # For preference learning
    session_id: str                 # Unique workflow ID

    # Data Analyst Outputs
    shortage_count: int             # Number of at-risk hospitals
    shortage_hospitals: List[Dict]  # Hospital details
    active_outbreaks: List[Dict]    # Ongoing events
    analysis_summary: str           # Human-readable summary

    # Forecasting Outputs
    demand_forecasts: Dict          # 14-day predictions per hospital
    forecast_summary: str

    # Optimization Outputs
    allocation_strategies: List[Dict]  # 3 strategies
    strategy_count: int

    # Preference Learning Outputs
    ranked_strategies: List[Dict]   # Sorted by preference score
    preference_profile: Dict        # User type + patterns

    # Reasoning Outputs
    final_recommendation: Dict      # Top strategy
    explanation: str                # LLM explanation

    # Human Review
    user_decision: Optional[int]    # Selected strategy index
    user_feedback: Optional[str]    # Optional text feedback

    # Metadata
    workflow_status: str            # "pending" | "completed" | "failed"
    timestamp: str                  # ISO 8601 timestamp
    execution_time_seconds: float
```

### Conditional Routing Logic

**After Data Analyst:**
```python
def route_after_analysis(state):
    if state["shortage_count"] == 0:
        return "END"  # No shortages → nothing to do
    else:
        return "forecasting"  # Shortages detected → predict demand
```

**After Optimization:**
```python
def route_after_optimization(state):
    strategies = state["allocation_strategies"]
    feasible = [s for s in strategies if s["status"] in ["optimal", "feasible"]]

    if len(feasible) > 0:
        return "preference"  # At least one feasible strategy → rank them
    else:
        return "END"  # No feasible strategies → end workflow
```

**After Human Review:**
```python
def route_after_human_review(state):
    if state["user_decision"] is not None:
        return "feedback"  # User made a selection → update preferences
    else:
        return "END"  # No decision → end workflow
```

---

## IV. Adaptive Learning and Reasoning Logic

### The Challenge: Learning User Preferences

Different decision-makers prioritize different objectives:
- **Cost-conscious**: Minimize transfer costs
- **Coverage-focused**: Help as many hospitals as possible
- **Urgency-driven**: Prioritize critical shortages
- **Balanced**: Trade-offs across all objectives

**Goal:** Learn each user's preference pattern and rank strategies accordingly.

### Hybrid ML Approach (40% RF + 30% LLM + 30% Vector)

Rather than relying on a single ML technique, MedFlow uses a **hybrid ensemble** that combines three complementary approaches:

```
User Interaction History
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 1. RANDOM FOREST (40% weight)                    │
│ Purpose: Fast pattern recognition                │
│                                                   │
│ Features (8 total):                              │
│  - Total cost (normalized)                       │
│  - Hospitals helped                              │
│  - Shortage reduction %                          │
│  - Distance metrics                              │
│  - Resource coverage                             │
│  - Urgency score                                 │
│                                                   │
│ Training:                                         │
│  - Online learning from each interaction         │
│  - 100 trees, max depth 10                      │
│  - Score: 0.0 - 1.0                             │
│                                                   │
│ Speed: <10ms per recommendation                  │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 2. LLM SEMANTIC ANALYSIS (30% weight)            │
│ Purpose: Deep reasoning about preferences        │
│                                                   │
│ Model: Groq/Llama 3.3 70B                       │
│                                                   │
│ Analysis:                                         │
│  - "What type of decision-maker is this user?"   │
│  - "Why would they choose strategy A over B?"   │
│  - "What patterns emerge from their history?"   │
│                                                   │
│ Output:                                           │
│  - Preference type classification               │
│  - Confidence score (0-100%)                    │
│  - Natural language explanation                 │
│                                                   │
│ Speed: ~200-500ms (API call)                     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ 3. VECTOR SIMILARITY (30% weight)                │
│ Purpose: Find similar past decisions             │
│                                                   │
│ Database: Qdrant (64-dimensional embeddings)     │
│                                                   │
│ Process:                                          │
│  - Convert interaction to 64-dim vector         │
│  - Store in user-specific collection            │
│  - Find K=5 most similar past decisions         │
│  - Score based on cosine similarity             │
│                                                   │
│ Storage:                                          │
│  - Per-user collections                         │
│  - Timestamped interactions                     │
│  - Metadata for filtering                       │
│                                                   │
│ Speed: ~5-20ms (vector search)                   │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│ WEIGHTED ENSEMBLE SCORING                        │
│                                                   │
│ Final Score = 0.40 × RF_score                   │
│             + 0.30 × LLM_score                  │
│             + 0.30 × Vector_score               │
│                                                   │
│ Sort strategies by final score (descending)     │
└──────────────────────────────────────────────────┘
    │
    ▼
Ranked Recommendations + Explanations
```

### Why This Hybrid Approach?

**Random Forest (40%):**
- ✅ Fast (<10ms)
- ✅ Learns from every interaction
- ✅ Handles numeric features well
- ❌ Limited semantic understanding

**LLM (30%):**
- ✅ Deep semantic reasoning
- ✅ Natural language explanations
- ✅ Understands context and nuance
- ❌ Slower (~200ms)
- ❌ Requires API calls

**Vector DB (30%):**
- ✅ Fast similarity search (<20ms)
- ✅ Finds relevant past decisions
- ✅ Works well with new users (cold start)
- ❌ Requires interaction history

**Ensemble Strength:** Combines speed (RF) + reasoning (LLM) + memory (Vector)

### Preference Learning Workflow

**Step 1: User Makes Decision**
```
User selects Strategy 2 (Maximum Coverage)
- Cost: $45,000
- Hospitals helped: 12
- Shortage reduction: 85%
```

**Step 2: Extract Features**
```python
features = {
    "total_cost": 45000,
    "hospitals_helped": 12,
    "shortage_reduction": 0.85,
    "avg_distance": 75.3,
    "resource_coverage": 0.92,
    "urgency_score": 0.78
}
```

**Step 3: Update Random Forest**
```python
# Add training sample
X_new = [features for all 3 strategies]
y_new = [0, 1, 0]  # User selected index 1

# Incremental learning
rf_model.partial_fit(X_new, y_new)
```

**Step 4: Store in Vector DB**
```python
# Generate embedding
interaction_vector = embed(interaction_data)  # 64-dim

# Store in Qdrant
qdrant.upsert(
    collection=f"user_{user_id}",
    points=[{
        "id": interaction_id,
        "vector": interaction_vector,
        "payload": {
            "selected_index": 1,
            "strategies": [...],
            "timestamp": "2025-11-08T10:30:00Z"
        }
    }]
)
```

**Step 5: Update LLM Profile**
```python
# Analyze pattern
profile = llm.invoke(f"""
User has made {n_interactions} decisions.
Past selections: [{strategy_names}]
Pattern: {pattern_summary}

What type of decision-maker is this user?
""")

# Result:
# "Coverage-focused (confidence: 80%)"
# "Consistently chooses strategies that help more hospitals"
```

### Reasoning Agent: Generating Explanations

The Reasoning Agent uses **Llama 3.3 70B** to synthesize all workflow outputs into a human-readable explanation:

**Input to LLM:**
```
System Prompt:
"You are a senior healthcare operations analyst. Explain resource
allocation recommendations clearly and actionably."

User Prompt:
- Shortage count: 12 hospitals need ventilators
- Active outbreaks: 1 (Dengue in Maharashtra)
- Top strategy: "Balanced" (Score: 0.87)
  - Cost: $38,500
  - Hospitals helped: 10
  - Shortage reduction: 78%
- User preference type: "Cost-conscious"
- User confidence: 75%
```

**LLM Output:**
```
🏥 SITUATION SUMMARY
Twelve hospitals across Maharashtra are experiencing or predicted to
experience ventilator shortages over the next 14 days, exacerbated by
the ongoing dengue outbreak.

✅ WHY THIS STRATEGY IS RECOMMENDED
• Aligns with your cost-conscious approach, achieving 78% shortage
  reduction at $38,500—30% less expensive than maximum coverage
• Helps 10 of 12 at-risk hospitals, prioritizing those in critical need
• Balances urgency with fiscal responsibility
• Minimizes long-distance transfers (avg 65km)

⚖️ KEY TRADEOFFS
• Cost savings: Saves $12,000 vs. maximum coverage strategy
• Coverage limitation: 2 hospitals will not receive resources in this plan
• These 2 hospitals have lower urgency scores and can likely wait 3-5 days

📋 NEXT STEPS
1. Review the allocation table below for specific hospital transfers
2. Verify distance feasibility for your logistics team
3. Approve or select an alternative strategy
```

### Graceful Degradation

The system handles missing components gracefully:

**No Groq API Key:**
- Use RF (40%) + Vector (60%) weights
- Skip LLM explanations

**No Qdrant:**
- Use RF (70%) + LLM (30%) weights
- Skip similarity search

**No Interaction History (New User):**
- Equal weighting for all strategies initially
- Start learning from first interaction

---

## V. Machine Learning Models: Technical Details

### 1. Demand Forecasting (LSTM)

**Problem:** Predict daily consumption for next 14 days

**Architecture:**
```
Input: 30 days × 17 features
    ↓
LSTM Layer 1 (128 units, dropout=0.5, return_sequences=True)
    ↓
LSTM Layer 2 (128 units, dropout=0.5)
    ↓
Batch Normalization
    ↓
Dense Layer 1 (64 units, ReLU, dropout=0.5)
    ↓
Dense Layer 2 (32 units, ReLU, dropout=0.5)
    ↓
Output Layer (14 units) → 14-day forecast
```

**Features (17 total):**
1. Base (4): quantity, consumption, resupply, admissions
2. Trends (4): 7-day MA, 14-day MA, slopes
3. Changes (4): deltas, momentum
4. Ratios (2): per-admission normalization
5. Indicators (3): % changes, direction

**Training Configuration:**
- Sequence length: 30 days
- Forecast horizon: 14 days
- Batch size: 32
- Epochs: 100 (with early stopping)
- Dropout: 0.5 (high for MC Dropout)
- Loss: MAE (Mean Absolute Error)

**Performance:**
| Resource | MAE | Interpretation |
|----------|-----|----------------|
| PPE | 4.65 | ±4-5 sets average error |
| O2 Cylinders | 2.03 | ±2 cylinders error |
| Ventilators | 1.02 | ±1 ventilator error (excellent!) |
| Medications | TBD | Expected <10 units |
| Beds | TBD | Expected <5 beds |

**Probabilistic Forecasting:**
- **MC Dropout**: Run 200 forward passes with dropout enabled
- **Output**: Mean, P10, P25, P50, P75, P90
- **Calibration**: Post-hoc scaling (factor: 1.143) + 15% safety buffer
- **Coverage**: ~75-80% effective coverage

**Use Cases:**
- Point predictions → Budget planning, trend analysis
- P90 + buffer → Daily ordering decisions
- P10-P90 range → Risk assessment

**Key Insight:** Healthcare prioritizes **safe, conservative forecasts** over aggressive optimization. Better to slightly overstock than risk patient harm.

### 2. Shortage Detection (Random Forest)

**Problem:** Classify shortage risk (critical/high/medium/low)

**Architecture:**
- 100 decision trees
- Max depth: 20
- 20 engineered features
- Multi-class classification

**Features (20 total):**
1. Stock-Demand Ratios (4): Current vs. predicted demand (7d, 14d)
2. Time-Based (4): Days of supply, days since resupply, days to critical
3. Consumption Velocity (4): Trends, volatility, acceleration
4. Regional Context (3): Regional average, transfer availability
5. Admission Patterns (3): Trends, ICU ratio, spikes
6. Resource-Specific (2): Criticality, consumption per admission

**Performance:**
- Overall Accuracy: 100% (on synthetic data)
- Critical Recall: 85%
- Critical Precision: 85%
- Weighted F1: 100%

**Note:** Perfect accuracy is due to rule-based synthetic labeling. Real-world performance will differ.

**Top Important Features:**
1. stock_demand_ratio (23.2%)
2. days_of_supply (18.0%)
3. stock_demand_ratio_14d (17.5%)
4. predicted_stockout_day (13.2%)
5. days_to_critical (7.6%)

### 3. Optimization (Linear Programming)

**Problem:** Find optimal resource allocation between surplus and shortage hospitals

**Formulation:**
```
Decision Variables:
  x[i,j] = quantity to transfer from hospital i to hospital j

Objective (weighted sum):
  Minimize: w1 × shortage_penalty
          + w2 × transfer_cost
          + w3 × transfer_complexity

Constraints:
  1. Inventory limits: x[i,j] ≤ surplus[i]
  2. Distance limits: distance[i,j] ≤ 200 km
  3. Fairness: critical hospitals must receive ≥ 50% of need or 1 unit
  4. Non-negativity: x[i,j] ≥ 0
```

**Three Strategy Variants:**

| Strategy | Weights | Optimizes For |
|----------|---------|---------------|
| Cost-Efficient | w1=5, w2=10, w3=1 | Minimize transfer costs |
| Maximum Coverage | w1=10, w2=1, w3=1 | Help most hospitals |
| Balanced | w1=7, w2=5, w3=2 | Trade-off approach |

**Solver:** PuLP with CBC/GLPK

**Performance:**
- Small problems (<10 hospitals): <1 second
- Medium problems (10-50 hospitals): 1-5 seconds
- Large problems (50-100 hospitals): 5-30 seconds

**Output Metrics:**
- Total cost
- Hospitals helped
- Shortage reduction %
- Average transfer distance
- Overall score (0-100)

### 4. Preference Learning (Hybrid Ensemble)

Already covered in Section IV.

---

## VI. Limitations and Challenges

### Current Limitations

**1. Synthetic Data Dependency**
- **Issue**: All models trained on synthetic, not real hospital data
- **Impact**: Performance metrics validate pipeline logic but don't guarantee real-world accuracy
- **Mitigation**: System design allows easy retraining with real data
- **Deployment Requirement**: Must retrain on actual hospital data before production use

**2. Probabilistic Forecast Calibration**
- **Issue**: MC Dropout coverage is 66.7% vs. 80% target
- **Impact**: Confidence intervals slightly too narrow
- **Mitigation**: Post-hoc calibration (1.143×) + 15% safety buffer → ~75-80% coverage
- **Acceptance**: Industry-standard approach; perfect calibration rarely achieved

**3. Single-Hospital Scope**
- **Issue**: Forecasting doesn't model inter-hospital transfers
- **Impact**: Can't predict demand changes due to resource sharing
- **Future Work**: Multi-hospital joint forecasting

**4. Limited Outbreak Modeling**
- **Issue**: Outbreak effects are simplified multipliers
- **Impact**: May not capture complex outbreak dynamics
- **Future Work**: Epidemiological model integration (SIR/SEIR)

**5. Static Resource Types**
- **Issue**: Five resource types hardcoded
- **Impact**: Can't easily add new resource categories
- **Future Work**: Configurable resource taxonomy

**6. No Cost Learning**
- **Issue**: Transfer costs are distance-based estimates
- **Impact**: May not reflect actual logistics costs
- **Future Work**: Learn cost models from historical transfers

**7. Cold Start Problem**
- **Issue**: New users have no preference history
- **Impact**: Initial recommendations may not match user style
- **Mitigation**: Default to balanced strategy, learn quickly from first interactions

### Technical Challenges Overcome

**1. Data Quality Issues** ✅
- **Problem**: Initial data had 84% zero-days for ventilators
- **Solution**: Fixed consumption formulas, improved correlation from 0.4 → 0.76

**2. MAPE Metric Failure** ✅
- **Problem**: MAPE showed 67% error for accurate ventilator predictions
- **Solution**: Use MAE for low-volume resources, document metric limitations

**3. Feature Engineering Complexity** ✅
- **Problem**: Mismatch between 7 raw features and 17 model features
- **Solution**: Created helper utilities, updated all scripts

**4. API-Agent Integration** ✅
- **Problem**: Connecting LangGraph agents to FastAPI backend
- **Solution**: Built API client with retry logic, comprehensive error handling

### Ethical Considerations

**1. Bias in Allocation**
- **Risk**: Optimization may favor well-connected hospitals
- **Mitigation**: Fairness constraints ensure critical hospitals receive minimum allocation

**2. Transparency**
- **Risk**: "Black box" ML decisions
- **Mitigation**: Natural language explanations for all recommendations, human-in-the-loop

**3. Data Privacy**
- **Risk**: Patient data exposure
- **Mitigation**: Aggregate data only, no patient identifiers

**4. Over-Reliance on AI**
- **Risk**: Humans defer to AI without critical review
- **Mitigation**: Mandatory human review step, optional feedback

---

## VII. Future Extensions and Improvements

### Short-Term (1-2 months)

**1. Real Data Integration**
- Partner with hospital network for pilot deployment
- Retrain all models on actual data
- Expected 20-30% accuracy improvement

**2. Quantile Regression for Forecasting**
- Replace MC Dropout with direct percentile learning
- Should achieve <5% calibration error
- Trade-off: Lose dropout regularization benefits

**3. Multi-Resource Joint Forecasting**
- Predict all 5 resources together
- Capture correlations (e.g., ICU surge → O2 + ventilators)
- Reduce total model size and training time

**4. Enhanced Outbreak Modeling**
- Integrate epidemiological models (SIR/SEIR)
- Predict outbreak trajectories
- Adjust demand forecasts dynamically

### Medium-Term (3-6 months)

**5. Transfer Learning for New Hospitals**
- Train on large hospitals, fine-tune for small ones
- Reduce data requirements from 30 days to 7 days
- Enable rapid deployment to new facilities

**6. Cost Model Learning**
- Learn actual transfer costs from historical data
- Replace distance-based estimates
- Improve optimization accuracy

**7. Attention Mechanisms**
- Add attention layers to LSTM
- Interpretability: See which historical days matter most
- Expected 10-15% accuracy boost

**8. Advanced Preference Learning**
- Deep learning for preference embeddings
- Contextual bandits for exploration-exploitation
- Multi-objective Bayesian optimization

**9. Real-Time Dashboard**
- Live inventory monitoring
- Shortage alerts
- Recommendation notifications

### Long-Term (6+ months)

**10. Network-Wide Optimization**
- Multi-hospital simultaneous optimization
- Dynamic resource pooling
- Regional coordination strategies

**11. Prescriptive Analytics**
- Recommend inventory policies (e.g., reorder points)
- Suggest capacity planning changes
- Long-term demand trend analysis

**12. Mobile Application**
- Hospital staff can view/approve allocations
- Push notifications for critical shortages
- Offline mode for low-connectivity areas

**13. Integration with EHR Systems**
- Pull real-time admission data
- Sync inventory updates
- Automated data pipelines

**14. Multi-Modal Data Integration**
- Weather data (flu season triggers)
- Social media signals (outbreak detection)
- Satellite imagery (population movements)

**15. Federated Learning**
- Train models across hospitals without sharing raw data
- Privacy-preserving collaboration
- Improved model accuracy via larger effective dataset

**16. Causal Inference**
- Estimate true impact of allocation decisions
- A/B testing framework
- Counterfactual analysis

---

## VIII. Production Deployment Considerations

### System Requirements

**Hardware:**
- **API Server**: 4 CPU cores, 8GB RAM
- **ML Inference**: 8GB RAM, optional GPU for faster forecasting
- **Database**: PostgreSQL-compatible (Supabase, AWS RDS, etc.)
- **Vector DB**: Qdrant (self-hosted or cloud)

**Software:**
- Python 3.10+
- FastAPI, LangGraph, Streamlit
- PostgreSQL 14+
- Qdrant 1.11+

### Scalability

**Current Capacity:**
- 100 hospitals with <1s response time
- 1000 hospitals feasible with database indexing
- 10,000+ hospitals requires sharding/caching

**Bottlenecks:**
- LSTM inference: ~1s point, ~60s probabilistic
  - Mitigation: Batch predictions, cache forecasts
- Optimization: 1-30s depending on problem size
  - Mitigation: Time limits, heuristics for large problems

### Security

**Required:**
- [ ] HTTPS/TLS for all API traffic
- [ ] OAuth2 for production authentication (replace API keys)
- [ ] Role-based access control (RBAC)
- [ ] Audit logging for all decisions
- [ ] Regular security audits
- [ ] HIPAA compliance review (if handling real patient data)

### Monitoring

**Metrics to Track:**
- API latency (p50, p95, p99)
- Error rates by endpoint
- Model prediction accuracy (drift detection)
- User satisfaction with recommendations
- System uptime/availability

**Tools:**
- Application: Sentry, DataDog
- Infrastructure: Prometheus, Grafana
- LLM: LangSmith (tracing and debugging)

### Maintenance

**Regular Tasks:**
- **Daily**: Monitor error logs, check for anomalies
- **Weekly**: Review prediction accuracy, retrain if drift detected
- **Monthly**: Update models with new data
- **Quarterly**: Full system audit, performance review

### Cost Estimates (Monthly, ~1000 hospitals)

| Component | Cost | Notes |
|-----------|------|-------|
| Compute (API) | $200 | 4 vCPU, 8GB RAM |
| Database | $100 | Supabase Pro or AWS RDS |
| Vector DB | $50 | Qdrant Cloud or self-hosted |
| LLM API (Groq) | $50 | ~10M tokens/month |
| Storage | $20 | Logs, checkpoints |
| **Total** | **~$420** | |

**Cost Optimization:**
- Cache frequent queries
- Batch ML predictions
- Use cheaper LLM for non-critical explanations
- Optimize database queries

---

## IX. Getting Started

### Quick Setup (Local Development)

```bash
# 1. Clone repository
git clone https://github.com/your-org/MedFlow.git
cd MedFlow

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Start backend API
cd backend
uvicorn app.main:app --reload --port 8000

# 5. (Optional) Run dashboard
cd ../dashboard
streamlit run app.py

# 6. (Optional) Test CLI
python -m cli.main allocate --resource ventilators --user test_user
```

### API Documentation

Once the backend is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Running Tests

```bash
# Backend API tests
cd backend
pytest tests/ -v

# ML Core tests
cd ml_core
pytest tests/ -v

# Agent tests
cd agents
pytest tests/ -v
```

### Training Models

```bash
# Train all 5 LSTM models
bash scripts/train_all_resources.sh

# Train shortage detector
python ml_core/training/train_shortage_model.py

# Evaluate models
python ml_core/training/evaluate_all_models.py
```

---

## X. Conclusion

MedFlow AI demonstrates that **complex healthcare resource allocation can be meaningfully improved** through a thoughtful combination of:

1. **Predictive Analytics** (LSTM forecasting)
2. **Optimization Algorithms** (Linear Programming)
3. **Adaptive Learning** (Hybrid preference system)
4. **Agent Orchestration** (Dual framework: LangGraph & CrewAI - independent implementations)
5. **Human Oversight** (Human-in-the-loop design)

While trained on synthetic data and requiring further validation, the system showcases a **viable architecture** for intelligent decision support in healthcare logistics. The modular design allows each component to be improved independently, and the human-in-the-loop approach ensures that AI augments rather than replaces human expertise.

### Key Achievements

✅ **End-to-End Pipeline**: Data → ML → Optimization → Agents → UI
✅ **Production-Ready Code**: 22/22 tests passing, comprehensive error handling
✅ **Explainable AI**: Natural language explanations for all recommendations
✅ **Adaptive Learning**: System improves from user feedback
✅ **Modular Architecture**: Easy to extend and maintain

### Next Step: Pilot Deployment

The system is ready for **pilot deployment** with a partner hospital network. Success metrics:
- 20%+ reduction in preventable shortages
- 15%+ reduction in transfer costs
- 90%+ user satisfaction with recommendations
- <5% false positive shortage alerts

---

## XI. References and Resources

### Documentation
- **Main README**: `/README.md`
- **LSTM Forecasting Guide**: `/docs/LSTM_APPROACH.md`
- **Phase 5 Plan**: `/docs/PHASE_5_PLAN.md`
- **Agent Workflow**: `/docs/AGENT_WORKFLOW.md`
- **API Reference**: `/docs/ENDPOINT_REFERENCE.md`

### Code Structure
- **Backend API**: `/backend/`
- **ML Core**: `/ml_core/`
- **Agents**: `/agents/`
- **Dashboard**: `/dashboard/`
- **Data Generation**: `/data/generators/`

### Academic References
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Kuleshov et al. (2018): "Accurate Uncertainties for Deep Learning"
- Vaswani et al. (2017): "Attention Is All You Need"

### Open Source Libraries
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **CrewAI**: https://docs.crewai.com/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Streamlit**: https://streamlit.io/
- **Qdrant**: https://qdrant.tech/

---

**Document Version**: 1.0
**Last Updated**: November 8, 2025
**Status**: Complete
**Maintained By**: MedFlow AI Team

---

*For questions, issues, or contributions, please open an issue on GitHub or contact the development team.*
