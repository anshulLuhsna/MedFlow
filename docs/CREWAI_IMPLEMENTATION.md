# CrewAI Implementation Guide

## Overview

This document describes the **CrewAI implementation** of the MedFlow AI resource allocation system. 

### Framework Independence

**Important:** The CrewAI implementation is **completely independent** from the LangGraph implementation:

- ✅ **Separate codebase**: Located in `agents/crewai/` (vs `agents/graph.py` for LangGraph)
- ✅ **Independent state management**: Uses CrewAI's `CrewOutput` objects (vs LangGraph's `MedFlowState` TypedDict)
- ✅ **No shared dependencies**: Each framework manages its own workflow execution
- ✅ **Same backend API**: Both frameworks call the same FastAPI endpoints
- ✅ **Same workflow logic**: Both implement identical 7-agent workflows
- ✅ **Toggle at runtime**: Users can switch between frameworks via dashboard or CLI flag

The CrewAI version provides an alternative orchestration approach using CrewAI's agent/task model with YAML configuration and decorator-based architecture, while maintaining full feature parity with the LangGraph implementation.

## Architecture

### Core Components

1. **YAML Configuration Files** (`agents/crewai/config/`)
   - `agents.yaml`: Defines 7 specialized agents with Role-Goal-Backstory framework
   - `tasks.yaml`: Defines 7 tasks with detailed descriptions following 80/20 rule

2. **CrewAI Tools** (`agents/crewai/tools/`)
   - 8 BaseTool classes wrapping MedFlowAPIClient methods
   - Each tool has clear descriptions for agent understanding

3. **Structured Output Models** (`agents/crewai/models/`)
   - Pydantic models for machine-readable task outputs

4. **Main Crew Definition** (`agents/crewai/crew/medflow_crew.py`)
   - `@CrewBase` decorator pattern
   - `@agent`, `@task`, `@crew` decorators
   - `@before_kickoff` and `@after_kickoff` for conditional routing

## Design Principles

### 80/20 Rule

**80% effort on tasks, 20% on agents** - Well-designed tasks elevate even simple agents.

- **Task Descriptions**: Detailed, step-by-step instructions (the "how")
- **Expected Outputs**: Clear format specifications (the "what")
- **Context Dependencies**: Explicit task chaining via `context` field

### Role-Goal-Backstory Framework

Each agent follows the Role-Goal-Backstory pattern:

- **Role**: Specific and specialized (e.g., "Healthcare Data Analyst specializing in resource shortage detection")
- **Goal**: Clear, outcome-focused with quality standards
- **Backstory**: Establishes expertise, working style, and values (2-3 sentences minimum)

### Task Design Best Practices

- **Single Purpose, Single Output**: One clear objective per task
- **Explicit Inputs/Outputs**: Clearly specify what inputs are used and output format
- **Purpose and Context**: Explain why task matters and how it fits workflow
- **Structured Outputs**: JSON structure specified in expected_output

## Agent Definitions

### 1. Data Analyst
- **Role**: Healthcare Data Analyst specializing in resource shortage detection and outbreak identification
- **Tools**: DetectShortagesTool, GetActiveOutbreaksTool
- **Responsibility**: Foundation analysis that drives all subsequent decisions

### 2. Forecasting Specialist
- **Role**: Demand Forecasting Specialist in Healthcare Resource Management
- **Tools**: ForecastDemandTool
- **Responsibility**: Predict future resource demand using time-series analysis

### 3. Optimization Specialist
- **Role**: Resource Allocation Optimizer specializing in healthcare logistics
- **Tools**: GenerateStrategiesTool
- **Responsibility**: Generate optimal allocation strategies (cost-efficient, maximum coverage, balanced)

### 4. Preference Analyst
- **Role**: User Preference Analyst specializing in adaptive learning systems
- **Tools**: ScorePreferencesTool
- **Responsibility**: Rank strategies based on user preferences using hybrid ML

### 5. Reasoning Specialist
- **Role**: AI Reasoning Specialist specializing in translating complex decisions into clear explanations
- **Tools**: None (uses LLM reasoning)
- **Responsibility**: Generate natural language explanations for recommendations

### 6. Human Review Coordinator
- **Role**: Human-in-the-Loop Coordinator specializing in AI-human collaboration interfaces
- **Tools**: None
- **Responsibility**: Present strategies to human decision-makers and collect feedback

### 7. Feedback Specialist
- **Role**: Preference Learning Specialist specializing in adaptive systems
- **Tools**: UpdatePreferencesTool, CreateAllocationTool, CreateUserInteractionTool
- **Responsibility**: Update preference models and store allocation records

## Task Workflow

1. **analyze_shortages_task** → Data Analyst
   - Detects hospitals with shortages
   - Identifies active outbreaks
   - Generates analysis summary

2. **forecast_demand_task** → Forecasting Specialist
   - Predicts demand for shortage hospitals
   - Uses context from analyze_shortages_task

3. **generate_strategies_task** → Optimization Specialist
   - Creates 3 allocation strategies
   - Uses context from forecast_demand_task

4. **rank_strategies_task** → Preference Analyst
   - Ranks strategies by user preferences
   - Uses context from generate_strategies_task

5. **explain_recommendation_task** → Reasoning Specialist
   - Generates LLM explanation
   - Uses context from rank_strategies_task

6. **review_strategies_task** → Human Review Coordinator
   - Presents strategies to user (human_input=True)
   - Collects user decision and feedback
   - Uses context from explain_recommendation_task

7. **update_preferences_task** → Feedback Specialist
   - Updates preference model
   - Stores allocation and interaction records
   - Uses context from review_strategies_task

## Conditional Routing

The `@after_kickoff` decorator implements conditional routing:

- Checks `analyze_shortages_task` output for `shortage_count == 0`
- If zero, marks subsequent tasks as skipped
- Prevents unnecessary processing when no shortages exist

## Human-in-the-Loop

The `review_strategies_task` has `human_input=True`:

- Pauses execution and waits for user input
- User provides strategy selection (0, 1, or 2) and optional feedback
- Task resumes with user decision in output

## Usage

### CLI

```bash
# Use CrewAI framework
python -m cli.main allocate --resource ventilators --user user123 --framework crewai

# Use LangGraph framework (default)
python -m cli.main allocate --resource ventilators --user user123 --framework langgraph
```

### Streamlit Dashboard

1. Open the demo dashboard: `./dashboard/run.sh`
2. In the Setup tab, select "CrewAI" from the Framework dropdown
3. Configure scenario (Normal/Outbreak, dates, resources)
4. Run workflow in Daily Simulation tab

### Python API

```python
from agents.crewai import MedFlowCrew

# Initialize crew
crew = MedFlowCrew().crew()

# Prepare inputs
inputs = {
    "resource_type": "ventilators",
    "user_id": "user123",
    "simulation_date": "2024-07-01",
    "hospital_ids": None,
    "outbreak_id": None,
    "regions": None,
    "hospital_limit": 5
}

# Run workflow
result = crew.kickoff(inputs=inputs)

# Access results
for task in result.tasks:
    print(f"Task: {task.description}")
    print(f"Output: {task.output}")
```

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Required for LLM (Groq/Llama 3.3 70B)
- `MEDFLOW_API_BASE`: Backend API URL (default: http://localhost:8000)
- `MEDFLOW_API_KEY`: API key (default: dev-key-123)
- `CREWAI_MEMORY`: Enable memory (default: true)
- `CREWAI_PLANNING`: Enable planning (default: true)
- `CREWAI_VERBOSE`: Enable verbose logging (default: true)

### YAML Variable Interpolation

Tasks support variable interpolation in descriptions:

```yaml
analyze_shortages_task:
  description: >
    Analyze the current resource situation for {resource_type}...
```

Variables are passed via `kickoff(inputs={...})`:
- `{resource_type}`
- `{user_id}`
- `{simulation_date}`
- `{hospital_ids}`
- `{outbreak_id}`
- `{regions}`
- `{hospital_limit}`

## Comparison with LangGraph

**Note:** This comparison is for informational purposes only. Both frameworks are **independent implementations** and can be used interchangeably.

### Similarities

- ✅ Same 7-agent workflow structure
- ✅ Same backend API integration (both call the same FastAPI endpoints)
- ✅ Same human-in-the-loop interaction
- ✅ Same preference learning system
- ✅ Same output format and functionality
- ✅ Both can be toggled at runtime via dashboard or CLI

### Differences

| Feature | LangGraph | CrewAI |
|---------|-----------|--------|
| **Codebase Location** | `agents/graph.py` | `agents/crewai/` |
| **Configuration** | Python-only | YAML + Python |
| **State Management** | TypedDict (`MedFlowState`) with reducers | Task context dependencies (`CrewOutput`) |
| **Conditional Routing** | Graph edges with conditions | `@after_kickoff` decorator |
| **State Persistence** | SqliteSaver checkpoints | `memory=True` |
| **Task Definition** | Python functions | YAML + `@task` decorator |
| **Agent Definition** | Python functions | YAML + `@agent` decorator |
| **Workflow Execution** | `graph.invoke(state)` | `crew.kickoff(inputs={...})` |
| **Output Format** | Dictionary (`MedFlowState`) | `CrewOutput` object with tasks |

### Independence Guarantees

- 🔒 **No code sharing**: Each framework has its own implementation
- 🔒 **No state sharing**: Each uses its own state management system
- 🔒 **No dependencies**: Installing/removing one framework doesn't affect the other
- 🔒 **Same functionality**: Both produce identical results and call the same APIs

### When to Use Each

**Use LangGraph when:**
- You need fine-grained control over state transitions
- You want explicit graph visualization
- You need complex conditional routing logic
- You prefer Python-only configuration

**Use CrewAI when:**
- You want YAML-based configuration
- You prefer declarative task definitions
- You want to leverage CrewAI's built-in features (memory, planning)
- You want easier agent/task management

## Migration Guide

### From LangGraph to CrewAI

1. **Install CrewAI**: `pip install crewai>=0.80.0 crewai-tools>=0.1.0`

2. **Update CLI/Dashboard**: Use `--framework crewai` flag or select CrewAI in dashboard

3. **Input Format**: CrewAI uses a dictionary of inputs instead of TypedDict state:
   ```python
   # LangGraph
   state: MedFlowState = {...}
   result = medflow_graph.invoke(state)
   
   # CrewAI
   inputs = {"resource_type": "...", "user_id": "..."}
   result = crew.kickoff(inputs=inputs)
   ```

4. **Output Format**: CrewAI returns a result object with tasks:
   ```python
   # LangGraph
   result["shortage_count"]
   
   # CrewAI
   result.tasks[0].output  # JSON string or dict
   ```

5. **Error Handling**: CrewAI errors are raised as exceptions, not stored in state

## Testing

### Unit Tests

```bash
# Run CrewAI tests
pytest agents/crewai/tests/test_crewai_workflow.py -v
```

Tests cover:
- Tool definitions and schemas
- YAML configuration loading
- Crew initialization
- Agent and task definitions

### Integration Tests

Integration tests require a running backend. They test:
- Agent-task pairs
- Tool execution
- End-to-end workflow

### E2E Tests

E2E tests match LangGraph functionality:
- Full workflow execution
- Output structure validation
- Preference learning verification

## Best Practices

1. **Follow 80/20 Rule**: Spend 80% effort on detailed task descriptions, 20% on agent definitions

2. **Use Role-Goal-Backstory**: Every agent should have a clear role, goal, and backstory

3. **Single Purpose Tasks**: Each task should have one clear objective

4. **Explicit Context**: Use `context` field to chain tasks explicitly

5. **Structured Outputs**: Always specify JSON structure in expected_output

6. **Error Handling**: Tools should return JSON with error information, not raise exceptions

7. **Variable Interpolation**: Use YAML variable interpolation for dynamic content

## Common Pitfalls

1. **YAML Key Mismatch**: Method names in Python MUST match YAML keys exactly
2. **Missing Context**: Tasks without `context` won't receive previous task outputs
3. **Tool Descriptions**: Vague tool descriptions lead to poor agent performance
4. **Output Format**: Agents may not follow expected_output if it's unclear
5. **Human Input**: Tasks with `human_input=True` will pause execution - ensure UI is ready

## Troubleshooting

### CrewAI Not Available

If you see "CrewAI not available" in the dashboard:
- Install: `pip install crewai>=0.80.0 crewai-tools>=0.1.0`
- Check imports: `from agents.crewai import MedFlowCrew`

### YAML Loading Errors

If YAML files fail to load:
- Check file paths: `agents/crewai/config/agents.yaml`
- Verify YAML syntax (use a YAML validator)
- Ensure method names match YAML keys

### Tool Execution Errors

If tools fail:
- Check backend is running: `http://localhost:8000`
- Verify API keys in environment variables
- Check tool descriptions are clear

### Workflow Hanging

If workflow hangs:
- Check for `human_input=True` tasks - they wait for user input
- Verify backend is responding
- Check logs for errors

## Future Enhancements

- [ ] Add hierarchical process support for complex workflows
- [ ] Implement task output caching
- [ ] Add more structured output validation
- [ ] Enhance error recovery mechanisms
- [ ] Add workflow visualization
- [ ] Implement task parallelization where possible

## References

- [CrewAI Documentation](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [MedFlow LangGraph Implementation](../agents/graph.py)
- [MedFlow Backend API](../backend/app)

