# Namel3ss Testing Framework Tests

This directory contains comprehensive tests for the namel3ss testing framework itself. These tests ensure the reliability and correctness of the testing infrastructure before it's used to test namel3ss applications.

## Test Structure

### Unit Tests
- **`test_dsl_level_tests.py`** - Tests for test specification DSL and YAML parsing
- **`test_llm_mocks.py`** - Tests for LLM mocking system (deterministic testing)
- **`test_tool_mocks.py`** - Tests for tool mocking (HTTP, database, vector search)
- **`test_runner.py`** - Tests for core test execution engine
- **`test_cli_test_command.py`** - Tests for CLI integration

### Integration Tests
- **`test_integration.py`** - End-to-end tests using fixture applications

### Test Fixtures
- **`fixtures/apps/`** - Sample namel3ss applications for testing
  - `simple_test_app.ai` - Basic app with prompts, agents, and chains
  - `complex_test_app.ai` - Advanced app with tools and complex workflows
- **`fixtures/tests/`** - Test suite YAML files for fixture applications
  - `simple_test_app.test.yaml` - Test cases for simple app
  - `complex_test_app.test.yaml` - Test cases for complex app

### Configuration
- **`conftest.py`** - Pytest fixtures and configuration
- **`__init__.py`** - Test package initialization

## Running Tests

### Run All Tests
```bash
# From namel3ss root directory
pytest tests/testing/ -v
```

### Run by Category
```bash
# Unit tests only
pytest tests/testing/ -m unit -v

# Integration tests only  
pytest tests/testing/ -m integration -v

# Exclude slow tests
pytest tests/testing/ -m "not slow" -v
```

### Run Specific Test Files
```bash
# Test DSL parsing
pytest tests/testing/test_dsl_level_tests.py -v

# Test LLM mocking
pytest tests/testing/test_llm_mocks.py -v

# Test CLI integration
pytest tests/testing/test_cli_test_command.py -v
```

### Run with Coverage
```bash
# Generate coverage report
pytest tests/testing/ --cov=namel3ss.testing --cov-report=html
```

## Test Coverage Goals

The testing framework tests aim for comprehensive coverage:

### DSL Tests (`test_dsl_level_tests.py`)
- ✅ YAML parsing and validation
- ✅ Test suite loading
- ✅ Test case model creation
- ✅ Assertion type validation
- ✅ Mock specification parsing
- ✅ Error handling for malformed files

### LLM Mock Tests (`test_llm_mocks.py`)
- ✅ Pattern matching (exact and regex)
- ✅ Response mapping and fallbacks
- ✅ Mock provider creation
- ✅ Metadata handling
- ✅ Response delay simulation
- ✅ Error scenarios

### Tool Mock Tests (`test_tool_mocks.py`)
- ✅ HTTP tool mocking
- ✅ Database tool mocking
- ✅ Vector search tool mocking
- ✅ Input pattern matching
- ✅ Registry management
- ✅ Fallback responses

### Runner Tests (`test_runner.py`)
- ✅ Application loading pipeline
- ✅ Target execution (prompts, agents, chains, apps)
- ✅ Assertion evaluation (all types)
- ✅ Test case execution
- ✅ Error handling and timeouts
- ✅ Result aggregation

### CLI Tests (`test_cli_test_command.py`)
- ✅ Test file discovery
- ✅ Command argument handling
- ✅ Test filtering
- ✅ Output formatting
- ✅ Verbose mode
- ✅ Fail-fast behavior

### Integration Tests (`test_integration.py`)
- ✅ End-to-end test execution
- ✅ Real fixture applications
- ✅ Complex assertion scenarios
- ✅ Error resilience
- ✅ Mock configuration validation

## Test Data and Fixtures

### Shared Fixtures (conftest.py)
- **`sample_test_case`** - Basic test case for unit tests
- **`sample_test_suite`** - Complete test suite with mocks
- **`mock_llm_provider`** - Pre-configured LLM mock
- **`mock_tool_registry`** - Pre-configured tool registry
- **`mock_application`** - Mock namel3ss application
- **`sample_assertion_test_cases`** - Data for assertion testing

### Fixture Applications
The fixture applications in `fixtures/apps/` provide realistic namel3ss code for integration testing:

**Simple Test App** (`simple_test_app.ai`):
- Basic prompts with input variables
- Simple agent with single capability
- Linear chain workflow
- Minimal dependencies

**Complex Test App** (`complex_test_app.ai`):
- Multi-modal prompts and agents
- Tool integration (HTTP, vector search)
- Complex multi-step workflows
- Advanced features (reasoning, memory)

## Test Patterns

### Mock Usage
Tests extensively use mocking to isolate components:
```python
@patch('namel3ss.testing.runner.TestRunner._load_application')
def test_execution_with_mock(self, mock_load):
    # Mock the application loading
    mock_app = Mock()
    mock_load.return_value = (mock_app, Mock())
    # ... test execution logic
```

### Assertion Testing
Each assertion type is tested with both passing and failing cases:
```python
def test_assertion_type(self):
    assertion = TestAssertion(type=AssertionType.CONTAINS, value="hello")
    
    # Should pass
    result = runner._evaluate_assertion(assertion, "hello world")
    assert result.passed is True
    
    # Should fail
    result = runner._evaluate_assertion(assertion, "goodbye world") 
    assert result.passed is False
```

### Error Scenarios
Tests cover error conditions comprehensively:
- File not found
- Invalid YAML syntax
- Malformed test specifications
- Application loading failures
- Runtime exceptions
- Timeout scenarios

## Maintenance

### Adding New Tests
When adding functionality to the testing framework:

1. **Unit Tests**: Create tests in the appropriate `test_*.py` file
2. **Fixtures**: Update `conftest.py` if shared test data is needed
3. **Integration**: Add end-to-end scenarios to `test_integration.py`
4. **Documentation**: Update this README with new test coverage

### Test Naming Convention
- Test classes: `TestComponentName`
- Test methods: `test_specific_behavior`
- Fixtures: `descriptive_fixture_name`

### Debugging Failed Tests
```bash
# Run single test with detailed output
pytest tests/testing/test_runner.py::TestTestRunner::test_specific_method -v -s

# Drop into debugger on failure
pytest tests/testing/ --pdb

# Show local variables on failure
pytest tests/testing/ --tb=long --showlocals
```

This comprehensive test suite ensures the namel3ss testing framework is production-ready and reliable for testing real namel3ss applications.