# Agent Tests

Comprehensive test suite for the MedFlow Phase 5 agentic layer.

## Test Structure

```
agents/tests/
├── __init__.py
├── conftest.py              # Shared fixtures and mocks
├── test_tools.py            # API client tests
├── test_nodes.py            # Unit tests for all 7 nodes
├── test_graph.py            # Integration tests for workflow
└── test_e2e.py              # End-to-end workflow tests
```

## Running Tests

### Run All Tests

```bash
# From project root
cd agents
pytest tests/ -v

# Or with coverage
pytest tests/ --cov=agents --cov-report=html -v
```

### Run Specific Test Files

```bash
# Test API client
pytest tests/test_tools.py -v

# Test agent nodes
pytest tests/test_nodes.py -v

# Test workflow integration
pytest tests/test_graph.py -v

# Test end-to-end
pytest tests/test_e2e.py -v
```

### Run Specific Test Classes

```bash
# Test specific node
pytest tests/test_nodes.py::TestDataAnalystNode -v

# Test routing
pytest tests/test_graph.py::TestRouting -v
```

### Run with Markers

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run only e2e tests
pytest -m e2e -v
```

## Test Coverage

### Test Files Created

1. **conftest.py** - Shared fixtures
   - `sample_state` - Sample MedFlowState
   - `mock_shortages_response` - Mock API response
   - `mock_outbreaks_response` - Mock API response
   - `mock_demand_forecast_response` - Mock API response
   - `mock_strategies_response` - Mock API response
   - `mock_preferences_response` - Mock API response
   - `mock_preferences_update_response` - Mock API response

2. **test_tools.py** - API Client Tests
   - Initialization tests
   - API method tests (get_shortages, predict_demand, etc.)
   - Error handling tests
   - Retry logic tests
   - **~15 test cases**

3. **test_nodes.py** - Node Unit Tests
   - `TestDataAnalystNode` - 3 test cases
   - `TestForecastingNode` - 4 test cases
   - `TestOptimizationNode` - 2 test cases
   - `TestPreferenceNode` - 2 test cases
   - `TestReasoningNode` - 2 test cases
   - `TestHumanReviewNode` - 3 test cases
   - `TestFeedbackNode` - 3 test cases
   - **~21 test cases**

4. **test_graph.py** - Integration Tests
   - `TestGraphWorkflow` - 4 test cases
   - `TestRouting` - 6 test cases
   - **~10 test cases**

5. **test_e2e.py** - End-to-End Tests
   - `TestEndToEndWorkflow` - 4 test cases
   - **~4 test cases**

**Total: ~50 test cases**

## Test Coverage Areas

### ✅ Covered

- **API Client** - All methods tested with mocks
- **All 7 Agent Nodes** - Each node has unit tests
- **Routing Logic** - All conditional routing tested
- **Workflow Integration** - Complete workflow tested
- **Error Handling** - Error scenarios tested
- **State Management** - State flow through nodes

### ⚠️ Not Yet Covered (Future)

- Real backend API integration (requires running backend)
- Performance testing
- Load testing
- State persistence verification (SQLite checkpointer)
- LLM response validation (requires API key)

## Writing New Tests

### Example: Test a New Node Method

```python
from unittest.mock import patch, MagicMock
from agents.nodes.your_node import your_node_function
from agents.tests.conftest import sample_state

class TestYourNode:
    @patch('agents.nodes.your_node.MedFlowAPIClient')
    def test_your_node_success(self, mock_client_class, sample_state):
        """Test successful execution"""
        mock_client = MagicMock()
        mock_client.your_method.return_value = {"result": "success"}
        mock_client_class.return_value = mock_client
        
        result = your_node_function(sample_state)
        
        assert result["key"] == "expected_value"
        mock_client.your_method.assert_called_once()
```

### Best Practices

1. **Use fixtures** from `conftest.py` for common test data
2. **Mock external dependencies** (API calls, LLM calls)
3. **Test both success and error cases**
4. **Verify state updates** after node execution
5. **Use descriptive test names** that explain what's being tested

## Dependencies

Tests require:
- `pytest>=7.4.3`
- `pytest-cov>=4.1.0` (optional, for coverage)
- `pytest-asyncio>=0.21.0` (if async tests added)

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-asyncio
```

## Continuous Integration

To run tests in CI/CD:

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
pytest agents/tests/ -v --cov=agents --cov-report=xml

# Check coverage threshold
pytest agents/tests/ --cov=agents --cov-report=term --cov-fail-under=80
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:
```bash
cd /home/anshul/Desktop/MedFlow
pytest agents/tests/ -v
```

### Mock Issues

If mocks aren't working, check that you're patching the correct import path:
```python
# Correct: patch where it's used, not where it's defined
@patch('agents.nodes.data_analyst.MedFlowAPIClient')
```

### LLM Tests

LLM tests are mocked to avoid API costs. To test with real LLM:
1. Set `OPENAI_API_KEY` environment variable
2. Remove `@patch('agents.nodes.reasoning.ChatOpenAI')` decorator
3. Use `@pytest.mark.slow` marker for slow tests

## Next Steps

1. ✅ **Test suite created** - All test files written
2. ⏳ **Run tests** - Verify all tests pass
3. ⏳ **Coverage report** - Generate and review coverage
4. ⏳ **CI/CD integration** - Add to CI pipeline
5. ⏳ **Real API tests** - Add tests with running backend

---

**Test Status:** ✅ **Test Suite Complete**  
**Coverage Target:** 80%+  
**Total Tests:** ~50 test cases

