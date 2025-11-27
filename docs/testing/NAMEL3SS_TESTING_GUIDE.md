# Namel3ss Testing Framework - API Reference

## Table of Contents

1. [Core Classes](#core-classes)
2. [Test Specification](#test-specification)
3. [Mock System](#mock-system)
4. [Test Runner](#test-runner)
5. [CLI Interface](#cli-interface)
6. [Python Integration](#python-integration)
7. [Utilities](#utilities)

## Core Classes

### TestSuite

Main test suite container.

```python
class TestSuite:
    name: str                    # Human-readable suite name
    app_module: str              # Path to namel3ss application
    cases: List[TestCase]        # Test cases in this suite  
    global_mocks: Dict[str, Any] # Global mock configurations
    description: Optional[str]   # Suite description
    timeout_ms: Optional[int]    # Default timeout for all tests
```

#### Methods

```python
def __init__(
    self,
    name: str,
    app_module: str, 
    cases: List[TestCase],
    global_mocks: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    timeout_ms: Optional[int] = None
) -> None
```

### TestCase

Individual test case definition.

```python
class TestCase:
    name: str                         # Unique test case name
    target: Dict[str, str]           # Target component to test
    assertions: List[TestAssertion]   # Expected outcomes
    inputs: Optional[Dict[str, Any]]  # Input data for test
    mocks: Optional[Dict[str, Any]]   # Test-specific mocks
    timeout_ms: Optional[int]         # Test-specific timeout
    description: Optional[str]        # Test description
```

#### Methods

```python
def __init__(
    self,
    name: str,
    target: Dict[str, str],
    assertions: List[TestAssertion],
    inputs: Optional[Dict[str, Any]] = None,
    mocks: Optional[Dict[str, Any]] = None,
    timeout_ms: Optional[int] = None,
    description: Optional[str] = None
) -> None
```

### TestAssertion

Assertion specification for test verification.

```python
class TestAssertion:
    type: AssertionType          # Type of assertion
    value: Any                   # Expected value
    path: Optional[str]          # JSON path for path assertions
    description: Optional[str]   # Assertion description
```

#### Methods

```python
def __init__(
    self,
    type: AssertionType,
    value: Any,
    path: Optional[str] = None,
    description: Optional[str] = None
) -> None
```

### AssertionType

Enumeration of supported assertion types.

```python
class AssertionType(Enum):
    EQUALS = "equals"                # Exact equality
    NOT_EQUALS = "not_equals"        # Inequality
    CONTAINS = "contains"            # Substring/element containment
    NOT_CONTAINS = "not_contains"    # Absence of substring/element
    MATCHES = "matches"              # Regex pattern match
    NOT_MATCHES = "not_matches"      # Regex pattern non-match
    HAS_KEYS = "has_keys"           # Dict contains keys
    MISSING_KEYS = "missing_keys"    # Dict missing keys
    HAS_LENGTH = "has_length"        # Collection length
    TYPE_IS = "type_is"             # Type checking
    JSON_PATH = "json_path"          # JSONPath query
    FIELD_EXISTS = "field_exists"    # Field existence
    FIELD_MISSING = "field_missing"  # Field absence
```

### TargetType

Enumeration of test target types.

```python
class TargetType(Enum):
    PROMPT = "prompt"    # Test a prompt component
    AGENT = "agent"      # Test an agent component  
    CHAIN = "chain"      # Test a chain component
    APP = "app"          # Test full application
```

## Test Specification

### load_test_suite

Load test suite from YAML file.

```python
def load_test_suite(file_path: Union[str, Path]) -> TestSuite:
    """
    Load and parse a test suite from YAML file.
    
    Args:
        file_path: Path to .test.yaml file
        
    Returns:
        Parsed TestSuite object
        
    Raises:
        FileNotFoundError: If test file doesn't exist
        ValueError: If YAML format is invalid
        yaml.YAMLError: If YAML syntax is malformed
    """
```

**Example:**
```python
from namel3ss.testing import load_test_suite

suite = load_test_suite("my_app.test.yaml")
print(f"Loaded suite: {suite.name}")
print(f"Test cases: {len(suite.cases)}")
```

## Mock System

### MockLLMSpec

Specification for LLM mock responses.

```python
class MockLLMSpec:
    model_name: str              # LLM model name
    prompt_pattern: str          # Pattern to match (exact or regex)
    response: MockLLMResponse    # Response configuration
```

### MockLLMResponse

LLM mock response definition.

```python
class MockLLMResponse:
    output_text: str                    # Response text
    metadata: Dict[str, Any]           # Response metadata
    delay_ms: int                      # Simulated response delay
```

### MockLLMProvider

Mock LLM provider for testing.

```python
class MockLLMProvider:
    model_name: str              # Provider model name
    
    def __init__(
        self,
        model_name: str,
        responses: Dict[str, Union[str, MockLLMResponse]],
        fallback: Optional[Union[str, MockLLMResponse]] = None
    ) -> None
    
    def generate(self, prompt: str) -> MockLLMResponse:
        """Generate mock response for prompt."""
```

**Example:**
```python
from namel3ss.testing.mocks import MockLLMProvider

provider = MockLLMProvider(
    "gpt-4",
    responses={
        "Hello": "Hi there!",
        r"My name is (\w+)": "Nice to meet you, {1}!"
    },
    fallback="I don't understand."
)

response = provider.generate("Hello")
print(response.output_text)  # "Hi there!"
```

### MockToolSpec

Specification for tool mock responses.

```python
class MockToolSpec:
    tool_name: str               # Tool identifier
    tool_type: str              # Tool type (http, database, etc.)
    input_pattern: Any          # Pattern to match inputs
    response: MockToolResponse   # Response configuration
```

### MockToolResponse

Tool mock response definition.

```python
class MockToolResponse:
    output: Any                 # Tool output data
    success: bool              # Success status
    error: Optional[str]       # Error message if failed
    metadata: Dict[str, Any]   # Response metadata
```

### MockToolRegistry

Registry for managing mock tools.

```python
class MockToolRegistry:
    def register_tool(self, tool: ToolMock) -> None:
        """Register a mock tool."""
    
    def get_tool(self, name: str) -> ToolMock:
        """Retrieve a mock tool by name."""
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        
    def __enter__(self) -> 'MockToolRegistry':
        """Context manager entry."""
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
```

**Example:**
```python
from namel3ss.testing.tools import MockToolRegistry, MockHttpTool

registry = MockToolRegistry()

# Create and register HTTP mock
http_tool = MockHttpTool("api_client")
http_tool.add_response(
    input_pattern={"method": "GET", "url": "https://api.example.com/data"},
    output={"status": 200, "data": ["item1", "item2"]},
    success=True
)

registry.register_tool(http_tool)

# Use in context
with registry:
    # Mock tools are active
    tool = registry.get_tool("api_client")
    result = tool.execute({"method": "GET", "url": "https://api.example.com/data"})
```

## Test Runner

### TestRunner

Core test execution engine.

```python
class TestRunner:
    verbose: bool               # Enable verbose output
    timeout_ms: int            # Default test timeout
    
    def __init__(
        self, 
        verbose: bool = False, 
        timeout_ms: int = 30000
    ) -> None
    
    def run_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run all test cases in a test suite."""
        
    def run_test_case(
        self, 
        test_case: TestCase, 
        app_module: str
    ) -> TestResult:
        """Run a single test case."""
```

**Example:**
```python
from namel3ss.testing.runner import TestRunner
from namel3ss.testing import load_test_suite

# Load test suite
suite = load_test_suite("my_app.test.yaml")

# Run tests
runner = TestRunner(verbose=True, timeout_ms=60000)
results = runner.run_test_suite(suite)

# Process results
for result in results:
    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: {result.test_name} ({result.execution_time_ms}ms)")
    if not result.passed:
        print(f"  Error: {result.error}")
```

### TestResult

Test execution result.

```python
class TestResult:
    test_name: str              # Test case name
    passed: bool               # Test success status
    execution_time_ms: int     # Execution duration
    output: Optional[Any]      # Test output (if successful)
    assertions_passed: int     # Number of passed assertions
    assertions_total: int      # Total number of assertions  
    error: Optional[str]       # Error message (if failed)
```

### TestExecutionError

Exception for test execution failures.

```python
class TestExecutionError(Exception):
    test_name: Optional[str]   # Name of failed test
    
    def __init__(
        self, 
        message: str, 
        test_name: Optional[str] = None
    ) -> None
```

## CLI Interface

### Command Line Usage

```bash
namel3ss test [OPTIONS] PATH
```

#### Options

- `--verbose, -v`: Enable verbose output
- `--timeout MILLISECONDS`: Set test timeout (default: 30000)
- `--filter PATTERN`: Filter tests by name pattern
- `--fail-fast`: Stop on first failure
- `--output-format {text,json}`: Output format (default: text)
- `--debug`: Enable debug mode
- `--help`: Show help message

#### Examples

```bash
# Basic usage
namel3ss test my_app.test.yaml

# Run with options
namel3ss test tests/ --verbose --timeout 60000

# Filter and fail fast
namel3ss test tests/ --filter "auth" --fail-fast

# JSON output for CI/CD
namel3ss test tests/ --output-format json > results.json
```

### cmd_test Function

CLI test command implementation.

```python
def cmd_test(args: argparse.Namespace) -> int:
    """
    Execute test command from CLI.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
```

## Python Integration

### pytest_integration Module

Integration with pytest framework.

```python
def parametrize_namel3ss_tests(test_file: str) -> Callable:
    """
    Decorator to parametrize pytest with namel3ss test cases.
    
    Args:
        test_file: Path to .test.yaml file
        
    Returns:
        Pytest parametrize decorator
    """
```

**Example:**
```python
import pytest
from namel3ss.testing.pytest_integration import parametrize_namel3ss_tests

@parametrize_namel3ss_tests("my_app.test.yaml")
def test_namel3ss_cases(namel3ss_test_case, namel3ss_runner):
    result = namel3ss_runner.run_test_case(namel3ss_test_case, "my_app.ai")
    assert result.passed, f"Test failed: {result.error}"
```

### Fixtures

Pytest fixtures for namel3ss testing.

```python
@pytest.fixture
def namel3ss_runner():
    """Provide TestRunner instance."""
    return TestRunner(verbose=True)

@pytest.fixture  
def mock_setup():
    """Provide MockSetup helper."""
    return MockSetup()
```

## Utilities

### Pattern Matching

```python
def _match_input_pattern(pattern: Any, input_data: Any) -> bool:
    """
    Match input data against a pattern.
    
    Args:
        pattern: Expected pattern (dict, str, list, etc.)
        input_data: Actual input data
        
    Returns:
        True if input matches pattern
    """
```

### File Discovery

```python
def _discover_test_files(path: str) -> List[str]:
    """
    Discover test files in directory or return single file.
    
    Args:
        path: File path or directory path
        
    Returns:
        List of test file paths
        
    Raises:
        FileNotFoundError: If path doesn't exist
    """
```

### Response Mapping

```python
class ResponseMapping:
    """Maps input patterns to responses for mocks."""
    
    def __init__(
        self,
        responses: Dict[str, Any],
        fallback: Optional[Any] = None
    ) -> None
    
    def get_response(self, input_text: str) -> Optional[Any]:
        """Get response for input text."""
```

## Error Handling

### Exception Hierarchy

```
Exception
├── TestExecutionError          # Test execution failures
├── TestSpecificationError      # Invalid test specification
├── MockConfigurationError      # Mock setup failures
└── AssertionError             # Assertion evaluation failures
```

### Error Messages

The framework provides detailed error messages:

```python
# Test file parsing error
"Invalid test file format: missing required field 'app_module'"

# Application loading error  
"Failed to load application 'app.ai': No such file or directory"

# Mock response error
"No mock response found for prompt: 'Unmocked prompt text'"

# Assertion failure
"Assertion failed: Expected 'success' but got 'error'"

# Timeout error
"Test 'test_name' exceeded timeout of 30000ms"
```

## Configuration

### Environment Variables

- `NAMEL3SS_TEST_TIMEOUT`: Default test timeout in milliseconds
- `NAMEL3SS_TEST_VERBOSE`: Enable verbose mode (true/false)
- `NAMEL3SS_TEST_DEBUG`: Enable debug logging (true/false)

### Configuration Files

```yaml
# namel3ss.toml
[testing]
default_timeout_ms = 30000
verbose = false
mock_delay_simulation = true
parallel_execution = false
max_workers = 4
```

## Type Hints

The framework provides comprehensive type hints:

```python
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from enum import Enum

# All public APIs include full type annotations
def load_test_suite(file_path: Union[str, Path]) -> TestSuite: ...
def run_tests(suite: TestSuite, verbose: bool = False) -> List[TestResult]: ...
```

## Version Compatibility

- **Python**: 3.8+
- **Namel3ss Core**: Compatible with current version
- **Dependencies**: PyYAML, jsonpath-ng, pytest (optional)

---

This API reference covers all public interfaces in the namel3ss testing framework. For implementation details and examples, see the [Developer Guide](NAMEL3SS_TESTING_DEVELOPER_GUIDE.md).# Namel3ss Testing Framework - Developer Guide

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Test Specification DSL](#test-specification-dsl)
4. [Mock System](#mock-system)
5. [Test Runner](#test-runner)
6. [CLI Usage](#cli-usage)
7. [Python Integration](#python-integration)
8. [Best Practices](#best-practices)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## Overview

The namel3ss testing framework provides **first-class, production-ready testing** for namel3ss applications. It features:

- **DSL-level testing**: YAML-based test specifications
- **Deterministic mocking**: No live API calls during testing
- **Native integration**: Uses real namel3ss parser/resolver/typechecker
- **CLI and Python support**: Run tests from command line or CI/CD
- **Comprehensive assertions**: Test prompts, agents, chains, and full applications

### Key Principles

1. **Production-Ready**: No shortcuts or toy implementations
2. **Deterministic**: Tests run the same way every time
3. **Integrated**: Leverages existing namel3ss infrastructure
4. **Comprehensive**: Test any namel3ss application component

## Quick Start

### 1. Create Your First Test

Create a test file `my_app.test.yaml`:

```yaml
app_module: "my_app.ai"
name: "My App Test Suite"
description: "Tests for my namel3ss application"

global_mocks:
  llms:
    - model_name: "gpt-4"
      prompt_pattern: "Hello .*"
      response:
        output_text: "Hi there! I'm happy to help."
        delay_ms: 100

cases:
  - name: "test_greeting_prompt"
    target:
      type: "prompt"
      name: "greeting"
    inputs:
      name: "Alice"
    assertions:
      - type: "contains"
        value: "Hello"
      - type: "contains"
        value: "Alice"
```

### 2. Run the Test

```bash
# Run single test file
namel3ss test my_app.test.yaml

# Run all tests in directory
namel3ss test tests/

# Verbose output
namel3ss test my_app.test.yaml --verbose
```

### 3. Review Results

```
Running test suite: My App Test Suite
✓ test_greeting_prompt (0.125s)

Results: 1 passed, 0 failed (0.125s total)
```

## Test Specification DSL

### Test Suite Structure

```yaml
# Required fields
app_module: "path/to/your/app.ai"  # Namel3ss application to test
name: "Test Suite Name"            # Human-readable suite name

# Optional fields  
description: "Suite description"   # Additional context
timeout_ms: 30000                 # Default timeout for all tests

# Mock configurations
global_mocks:
  llms: []     # LLM provider mocks
  tools: []    # Tool mocks

# Test cases
cases:
  - name: "test_name"
    # ... test case definition
```

### Test Case Structure

```yaml
- name: "test_case_name"           # Required: unique test name
  description: "What this tests"   # Optional: test description
  
  # What to test
  target:
    type: "prompt"                 # prompt, agent, chain, app
    name: "prompt_name"           # Component name (not needed for app)
  
  # Input data
  inputs:                         # Optional: inputs to component
    key1: "value1"
    key2: "value2"
  
  # Verification
  assertions:                     # List of assertions
    - type: "contains"
      value: "expected text"
      description: "Should contain greeting"
  
  # Test configuration
  timeout_ms: 5000               # Optional: test-specific timeout
  mocks:                         # Optional: test-specific mocks
    llms: []
    tools: []
```

### Target Types

#### Prompt Testing
```yaml
target:
  type: "prompt"
  name: "greeting_prompt"
inputs:
  name: "Alice"
  language: "English"
```

#### Agent Testing
```yaml
target:
  type: "agent"
  name: "research_agent"
inputs:
  query: "What is machine learning?"
  max_sources: 5
```

#### Chain Testing
```yaml
target:
  type: "chain"  
  name: "analysis_workflow"
inputs:
  document: "Text to analyze..."
  options:
    include_sentiment: true
```

#### Full Application Testing
```yaml
target:
  type: "app"
inputs:
  config:
    mode: "production"
    verbose: false
```

### Assertion Types

#### Text Assertions
```yaml
# Exact match
- type: "equals"
  value: "Expected exact text"

# Contains substring
- type: "contains"
  value: "substring"

# Does not contain
- type: "not_contains"
  value: "unwanted text"

# Regex pattern
- type: "matches"
  value: "\\d{3}-\\d{3}-\\d{4}"  # Phone number pattern

# Does not match pattern
- type: "not_matches"
  value: "bad_pattern"
```

#### Structure Assertions
```yaml
# Has specific keys (for dict output)
- type: "has_keys"
  value: ["name", "age", "email"]

# Missing specific keys
- type: "missing_keys"  
  value: ["password", "secret"]

# Check length
- type: "has_length"
  value: 5

# Check type
- type: "type_is"
  value: "dict"  # dict, list, str, int, float, bool
```

#### Advanced Assertions
```yaml
# JSON path query
- type: "json_path"
  path: "$.user.profile.email"
  value: "alice@example.com"

# Field existence
- type: "field_exists"
  value: "metadata"

# Field absence
- type: "field_missing" 
  value: "deprecated_field"
```

## Mock System

### LLM Mocking

Mock LLM responses for deterministic testing:

```yaml
global_mocks:
  llms:
    # Exact string matching
    - model_name: "gpt-4"
      prompt_pattern: "Hello world"
      response:
        output_text: "Hi there!"
        delay_ms: 100
        metadata:
          tokens_used: 15
    
    # Regex pattern matching with substitution
    - model_name: "claude"
      prompt_pattern: "My name is (\\w+)"
      response:
        output_text: "Nice to meet you, {1}!"
        delay_ms: 150
    
    # Complex analysis pattern
    - model_name: "gpt-4"
      prompt_pattern: "Analyze: (.*)"
      response:
        output_text: "Analysis of '{1}': The text shows positive sentiment with high confidence."
        metadata:
          confidence: 0.95
          sentiment: "positive"
```

#### Pattern Matching Rules

1. **Exact Match**: String patterns match exactly
2. **Regex Match**: Patterns starting with `\` or containing regex special chars
3. **Substitution**: Use `{1}`, `{2}`, etc. for regex group substitutions
4. **Fallback**: Configure fallback responses for unmatched patterns

### Tool Mocking

Mock external tools (HTTP, database, vector search):

```yaml
global_mocks:
  tools:
    # HTTP API mocking
    - tool_name: "api_client"
      tool_type: "http"
      input_pattern:
        method: "GET"
        url: "https://api.example.com/users"
      response:
        output:
          status_code: 200
          data:
            users: ["alice", "bob", "charlie"]
        success: true
        
    # Database mocking  
    - tool_name: "db_client"
      tool_type: "database"
      input_pattern:
        query: "SELECT * FROM users WHERE active = true"
      response:
        output:
          rows:
            - {id: 1, name: "alice", active: true}
            - {id: 2, name: "bob", active: true}
          count: 2
        success: true
    
    # Vector search mocking
    - tool_name: "vector_search"
      tool_type: "vector_search"
      input_pattern:
        query: "machine learning"
        top_k: 5
      response:
        output:
          results:
            - {id: "doc1", score: 0.95, content: "ML is a subset of AI"}
            - {id: "doc2", score: 0.87, content: "Neural networks"}
        success: true
```

#### Tool Types
- **`http`**: HTTP API calls
- **`database`**: Database queries
- **`vector_search`**: Vector similarity search
- **Custom**: Extend with your own tool types

## Test Runner

### Core Engine

The test runner integrates with the namel3ss pipeline:

```
Application File (.ai)
       ↓
   Parser → AST
       ↓  
   Resolver → Resolved AST
       ↓
   TypeChecker → Typed AST
       ↓
   Test Execution → Results
```

### Execution Flow

1. **Load Application**: Parse, resolve, and type-check namel3ss app
2. **Setup Mocks**: Configure LLM and tool mocks
3. **Execute Target**: Run prompt/agent/chain/app with test inputs
4. **Evaluate Assertions**: Check output against expected results
5. **Collect Results**: Aggregate test outcomes and timing

### Error Handling

The test runner handles errors gracefully:

- **Syntax Errors**: Application parsing failures
- **Type Errors**: Type checking failures  
- **Runtime Errors**: Execution exceptions
- **Timeout Errors**: Tests taking too long
- **Assertion Errors**: Failed expectations

## CLI Usage

### Basic Commands

```bash
# Run single test file
namel3ss test app.test.yaml

# Run all tests in directory
namel3ss test tests/

# Run with filter
namel3ss test tests/ --filter "test_auth"

# Verbose output  
namel3ss test app.test.yaml --verbose

# Fail fast (stop on first failure)
namel3ss test tests/ --fail-fast

# Custom timeout
namel3ss test app.test.yaml --timeout 60000

# JSON output format
namel3ss test app.test.yaml --output-format json
```

### Test Discovery

The CLI automatically discovers test files:

- **Pattern**: `*.test.yaml` or `*.test.yml`
- **Recursive**: Searches subdirectories
- **Exclusions**: Ignores non-test YAML files

### Output Formats

#### Text Format (Default)
```
Running test suite: My App Test Suite
✓ test_greeting_prompt (0.125s)
✓ test_analysis_chain (0.340s)
✗ test_error_handling (0.089s)
  └─ Assertion failed: Expected 'error' but got 'success'

Results: 2 passed, 1 failed (0.554s total)
```

#### JSON Format
```json
{
  "suite_name": "My App Test Suite",
  "total_tests": 3,
  "passed": 2,
  "failed": 1,
  "execution_time_ms": 554,
  "results": [
    {
      "test_name": "test_greeting_prompt", 
      "passed": true,
      "execution_time_ms": 125,
      "assertions_passed": 2,
      "assertions_total": 2
    }
  ]
}
```

## Python Integration

### Pytest Plugin

Use namel3ss tests in your pytest workflow:

```python
# test_my_app.py
import pytest
from namel3ss.testing.pytest_integration import parametrize_namel3ss_tests

# Load and parametrize namel3ss tests
@parametrize_namel3ss_tests("my_app.test.yaml")
def test_namel3ss_cases(namel3ss_test_case, namel3ss_runner):
    """Run namel3ss test cases as pytest tests."""
    result = namel3ss_runner.run_test_case(namel3ss_test_case, "my_app.ai")
    assert result.passed, f"Test failed: {result.error}"
```

### Direct Integration

Use the testing framework directly in Python:

```python
from namel3ss.testing import load_test_suite
from namel3ss.testing.runner import TestRunner

# Load test suite
suite = load_test_suite("my_app.test.yaml")

# Run tests
runner = TestRunner(verbose=True)
results = runner.run_test_suite(suite)

# Process results
for result in results:
    if not result.passed:
        print(f"FAILED: {result.test_name}")
        print(f"Error: {result.error}")
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: Test Namel3ss App
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install namel3ss
        run: pip install -e .
      - name: Run namel3ss tests
        run: namel3ss test tests/ --output-format json > test-results.json
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test-results.json
```

## Best Practices

### Test Organization

```
project/
├── src/
│   └── my_app.ai
├── tests/
│   ├── unit/
│   │   ├── test_prompts.test.yaml
│   │   ├── test_agents.test.yaml
│   │   └── test_chains.test.yaml
│   ├── integration/
│   │   └── test_full_app.test.yaml
│   └── fixtures/
│       ├── sample_data.json
│       └── mock_responses.yaml
└── namel3ss.toml
```

### Test Naming

- **Descriptive**: `test_user_authentication_with_valid_credentials`
- **Consistent**: Use prefixes like `test_`, `should_`, `when_`
- **Hierarchical**: Group related tests with common prefixes

### Mock Design

```yaml
# ✅ Good: Specific, realistic mocks
- model_name: "gpt-4"
  prompt_pattern: "Summarize the document: (.*)"
  response:
    output_text: "Summary: {1} contains 3 key points about AI applications."
    metadata:
      summary_length: 156
      confidence: 0.92

# ❌ Bad: Generic, unrealistic mocks  
- model_name: "gpt-4"
  prompt_pattern: ".*"
  response:
    output_text: "Mock response"
```

### Assertion Strategy

1. **Test Behavior, Not Implementation**: Focus on what the component should do
2. **Multiple Assertions**: Verify different aspects of the output
3. **Meaningful Messages**: Include descriptions for failed assertions
4. **Avoid Brittleness**: Don't test exact formatting unless it matters

### Error Testing

```yaml
# Test error conditions
- name: "test_invalid_input_handling"
  target:
    type: "agent" 
    name: "validator"
  inputs:
    data: null  # Invalid input
  assertions:
    - type: "contains"
      value: "error"
      description: "Should report validation error"
```

## Advanced Features

### Custom Assertions

Extend the framework with custom assertion types:

```python
# namel3ss/testing/custom_assertions.py
from namel3ss.testing import TestAssertion, AssertionResult

def evaluate_custom_assertion(assertion: TestAssertion, output: Any) -> AssertionResult:
    if assertion.type.value == "custom_metric":
        # Custom logic here
        passed = calculate_metric(output) > assertion.value
        message = f"Metric {assertion.value} {'passed' if passed else 'failed'}"
        return AssertionResult(passed, message)
    
    raise ValueError(f"Unknown custom assertion: {assertion.type}")
```

### Dynamic Test Generation

Generate tests programmatically:

```python
from namel3ss.testing import TestSuite, TestCase, TestAssertion, AssertionType

def generate_prompt_tests(prompt_names: List[str]) -> TestSuite:
    cases = []
    for name in prompt_names:
        case = TestCase(
            name=f"test_{name}_prompt",
            target={"type": "prompt", "name": name},
            inputs={"test_input": "sample"},
            assertions=[
                TestAssertion(type=AssertionType.TYPE_IS, value="str")
            ]
        )
        cases.append(case)
    
    return TestSuite(
        name="Generated Prompt Tests",
        app_module="app.ai", 
        cases=cases
    )
```

### Parallel Execution

Run tests in parallel for faster execution:

```python
from namel3ss.testing.runner import ParallelTestRunner

runner = ParallelTestRunner(max_workers=4)
results = runner.run_test_suite(suite)
```

## Troubleshooting

### Common Issues

#### Test File Not Found
```
Error: Test file 'app.test.yaml' not found
```
**Solution**: Check file path and ensure `.test.yaml` extension

#### Invalid YAML Syntax
```
Error: Invalid YAML in test file: mapping values are not allowed here
```
**Solution**: Validate YAML syntax, check indentation

#### Application Loading Failed
```
Error: Failed to load application 'app.ai': No such file or directory
```
**Solution**: Verify `app_module` path in test suite

#### Mock Not Matching
```
Error: No mock response found for prompt: "Hello world"
```
**Solution**: Check mock patterns match exactly or use regex

#### Assertion Failed
```
Assertion failed: Expected 'success' but got 'error'
```
**Solution**: Review application logic or adjust assertion expectations

### Debug Mode

Enable verbose debugging:

```bash
# Maximum verbosity
namel3ss test app.test.yaml --verbose --debug

# Show mock matching details
namel3ss test app.test.yaml --debug-mocks

# Show assertion details
namel3ss test app.test.yaml --debug-assertions
```

### Logging Configuration

Configure detailed logging:

```python
import logging

# Enable debug logging
logging.getLogger('namel3ss.testing').setLevel(logging.DEBUG)

# Log mock responses
logging.getLogger('namel3ss.testing.mocks').setLevel(logging.DEBUG)

# Log test execution
logging.getLogger('namel3ss.testing.runner').setLevel(logging.DEBUG)
```

### Performance Tips

1. **Mock Efficiently**: Use specific patterns to avoid regex overhead
2. **Minimize I/O**: Keep test files and applications small
3. **Parallel Testing**: Use parallel runner for large test suites
4. **Timeout Appropriately**: Set realistic timeouts based on application complexity

---

This developer guide provides comprehensive coverage of the namel3ss testing framework. For additional examples and advanced usage patterns, see the [examples directory](examples/) and [API reference](API_REFERENCE.md).