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

This API reference covers all public interfaces in the namel3ss testing framework. For implementation details and examples, see the [Developer Guide](NAMEL3SS_TESTING_DEVELOPER_GUIDE.md).