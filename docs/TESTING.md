# Namel3ss Testing Guide

This document describes the testing infrastructure for Namel3ss, covering all aspects from language-level compilation tests to frontend-backend integration validation.

---

## Table of Contents

1. [Overview](#overview)
2. [Running Tests](#running-tests)
3. [Test Categories](#test-categories)
4. [Writing Tests](#writing-tests)
5. [Test Utilities](#test-utilities)
6. [CI Integration](#ci-integration)
7. [Coverage](#coverage)

---

## Overview

The Namel3ss test suite ensures:

- **Language correctness**: All `.n3` syntax compiles and resolves correctly
- **Runtime behavior**: Generated backends respond with correct schemas and status codes
- **AI features**: Structured prompts, chains, memory, and agents work as documented
- **Error quality**: Developers receive clear, actionable error messages
- **Integration stability**: Frontend and backend maintain compatible contracts

### Test Philosophy

- **No external dependencies**: Tests use mocks and test doubles, not real LLM APIs or databases
- **Deterministic**: Tests produce consistent results; no random behavior
- **No demo data in production**: Synthetic data belongs only in test fixtures
- **Documentation coverage**: Every documented example has a corresponding test

---

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_structured_prompts_validator.py
```

### Run Tests by Category

```bash
# Language-level tests
pytest tests/test_parser.py tests/test_loader.py tests/test_resolver.py

# Backend integration tests
pytest tests/test_backend_integration.py tests/test_frames_api.py

# AI feature tests
pytest tests/test_structured_prompts_*.py tests/test_memory_system.py tests/test_agent_runtime.py

# Error handling tests
pytest tests/test_errors.py tests/test_error_messages.py
```

### Run with Coverage

```bash
pytest --cov=namel3ss --cov-report=html
open htmlcov/index.html
```

### Run with Verbose Output

```bash
pytest -v
```

### Run Specific Test Function

```bash
pytest tests/test_memory_system.py::test_memory_registry_basic -v
```

---

## Test Categories

### 1. Language Tests

**Purpose**: Validate that `.n3` source files compile through the full pipeline without errors.

**Location**: `tests/test_language_integration.py`

**What They Test**:
- Parsing: Syntax is correctly recognized and AST is built
- Resolution: Symbols are resolved, types are checked, references are valid
- Code generation: Backend and frontend are generated without errors
- Examples: All `.n3` files in `examples/` compile successfully

**Example**:

```python
def test_demo_app_compiles():
    """Test that demo_app.n3 compiles through full pipeline."""
    from namel3ss.loader import load_program, extract_single_app
    
    program = load_program("demo_app.n3")
    app = extract_single_app(program)
    
    assert app is not None
    assert app.name == "Demo API"
```

### 2. Backend Integration Tests

**Purpose**: Validate that generated FastAPI endpoints behave correctly.

**Location**: `tests/test_backend_integration.py`, `tests/test_frames_api.py`

**What They Test**:
- Page endpoints return correct status codes and schemas
- Component endpoints (tables, charts, forms) validate requests
- Error responses follow consistent format
- CORS and security headers are present
- State registries contain expected entries

**Example**:

```python
def test_page_endpoint_returns_schema(backend_client):
    """Test that page endpoint returns correct schema."""
    response = backend_client.get("/api/pages/home")
    
    assert response.status_code == 200
    data = response.json()
    assert data["slug"] == "home"
    assert "components" in data
```

**Pattern**: Tests use `FastAPI.TestClient` with generated backends loaded dynamically.

### 3. Structured Prompt Tests

**Purpose**: Validate that structured prompts enforce their schemas correctly.

**Location**: `tests/test_structured_prompts_validator.py`, `tests/test_structured_prompts_runtime.py`

**What They Test**:
- Valid outputs pass validation
- Invalid outputs produce specific error messages
- Enum constraints are enforced
- Nullable fields behave correctly
- Nested objects and lists validate properly
- Type coercion follows documented rules

**Example**:

```python
def test_enum_validation_rejects_invalid_values():
    """Test that invalid enum values produce clear errors."""
    schema = OutputSchema(fields=[
        OutputField(
            name="category",
            field_type=OutputFieldType(base_type="enum", enum_values=["a", "b", "c"])
        )
    ])
    validator = OutputValidator(schema)
    
    result = validator.validate({"category": "invalid"})
    
    assert not result.is_valid
    assert any("category" in err and "a, b, c" in err for err in result.errors)
```

### 4. Chain and Agent Tests

**Purpose**: Validate chain orchestration and agent behavior.

**Location**: `tests/test_chain_runtime.py`, `tests/test_agent_runtime.py`, `tests/test_graph_integration.py`

**What They Test**:
- Chain steps execute in correct order
- Data flows between steps correctly
- Errors propagate with clear messages
- Mock LLM providers simulate responses
- Agent tools are invoked correctly
- Graph routing follows conditional logic

**Example**:

```python
def test_chain_data_flow():
    """Test that data flows correctly between chain steps."""
    chain = Chain(
        name="test_chain",
        steps=[
            ChainStep(kind="prompt", target="step1"),
            ChainStep(kind="transform", target="step2"),
        ]
    )
    
    result = await execute_chain(chain, input_data)
    
    assert result.steps[1].input == result.steps[0].output
```

### 5. Memory System Tests

**Purpose**: Validate memory operations, scoping, and persistence.

**Location**: `tests/test_memory_system.py`, `tests/test_memory_runtime.py`

**What They Test**:
- Memory registration and retrieval
- Scope isolation (session, user, conversation, global)
- Type validation for different memory kinds
- Capacity enforcement and eviction
- Prompt template memory substitution
- Built-in memory functions

**Example**:

```python
async def test_memory_scope_isolation():
    """Test that different users/sessions have isolated memory."""
    registry = MemoryRegistry()
    registry.register(MemorySpec(name="data", scope="user", kind="list"))
    
    handle1 = registry.get("data", scope_context={"user_id": "user1"})
    handle2 = registry.get("data", scope_context={"user_id": "user2"})
    
    await handle1.write(["item1"])
    await handle2.write(["item2"])
    
    data1 = await handle1.read()
    data2 = await handle2.read()
    
    assert data1 == ["item1"]
    assert data2 == ["item2"]
```

### 6. Error Handling Tests

**Purpose**: Validate that errors are clear, consistent, and actionable.

**Location**: `tests/test_errors.py`, `tests/test_error_messages.py`

**What They Test**:
- Parser errors include line/column numbers
- Resolver errors identify undefined symbols
- Runtime errors return proper HTTP status codes
- Error messages follow consistent format
- Field-level validation errors are specific

**Example**:

```python
def test_undefined_symbol_error_is_clear():
    """Test that undefined symbol errors identify the problem."""
    source = 'page "Test" at "/": show table "Data" from dataset undefined_dataset'
    
    with pytest.raises(ModuleResolutionError) as exc_info:
        parser = Parser(source)
        module = parser.parse()
        resolve_program(Program(modules=[module]))
    
    error = str(exc_info.value)
    assert "undefined_dataset" in error
    assert "not found" in error or "undefined" in error
```

### 7. Frontend-Backend Integration Tests

**Purpose**: Validate that generated frontend correctly communicates with backend.

**Location**: `tests/test_frontend_integration.py`

**What They Test**:
- `usePageData` hook fetches data correctly
- Loading states display appropriately
- Error states show user-friendly messages
- Client library (`n3Client.ts`) handles responses
- Components render with real backend data

**Example** (requires React Testing Library):

```python
def test_page_loads_data_from_backend(backend_client, frontend_app):
    """Test that React page fetches and displays backend data."""
    # Start backend with TestClient
    with TestClient(backend_app) as client:
        # Mock fetch to point to TestClient
        response = client.get("/api/pages/home")
        
        # Render React component with mocked fetch
        render_result = render_page("Home", mock_fetch=lambda: response.json())
        
        # Assert data is displayed
        assert "Welcome" in render_result.text
```

---

## Writing Tests

### Test File Structure

```python
"""Brief description of what this test module covers."""

import pytest
from namel3ss.loader import load_program
from namel3ss.parser import Parser
# ... other imports


@pytest.fixture
def example_app():
    """Fixture providing a test app."""
    source = 'app "TestApp".\npage "Home" at "/": show text "Hello"'
    return Parser(source).parse_app()


def test_specific_behavior():
    """Test that [specific behavior] works correctly.
    
    This test validates that when [condition], the system [expected behavior].
    """
    # Arrange
    input_data = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.expected_property == expected_value
    assert result.is_valid
```

### Guidelines

1. **One concept per test**: Each test validates one specific behavior
2. **Clear naming**: Test names describe what is tested and expected outcome
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and validation
4. **Deterministic**: No randomness, external APIs, or uncontrolled state
5. **Fast**: Tests should complete in milliseconds, not seconds
6. **Independent**: Tests don't depend on execution order
7. **Docstrings**: Explain what the test validates and why it matters

### Async Tests

Tests with async code use the custom pytest hook in `conftest.py`:

```python
async def test_async_operation():
    """Test async functionality."""
    result = await async_function()
    assert result.success
```

No decorator required—async functions are automatically detected and run in an event loop.

### Mock LLM Providers

Never call real LLM APIs in tests. Use test doubles:

```python
class MockLLMProvider:
    """Test double for LLM providers."""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Return deterministic response based on prompt."""
        if "classify" in prompt.lower():
            return '{"category": "technical", "urgency": "high"}'
        return '{"response": "mock response"}'


@pytest.fixture
def mock_llm():
    """Provide mock LLM for tests."""
    return MockLLMProvider()
```

### Backend Test Utilities

The `backend_test_utils.py` module provides helpers:

```python
from tests.backend_test_utils import (
    generate_test_backend,
    load_backend_module,
    cleanup_backend_modules,
)

def test_with_generated_backend(tmp_path):
    """Test using a dynamically generated backend."""
    app_source = 'app "Test".\npage "Home" at "/": show text "Test"'
    backend_dir = generate_test_backend(app_source, tmp_path)
    
    try:
        main_module = load_backend_module("test_backend", backend_dir)
        with TestClient(main_module.app) as client:
            response = client.get("/api/pages/home")
            assert response.status_code == 200
    finally:
        cleanup_backend_modules("test_backend")
```

---

## Test Utilities

### `conftest.py`

Shared fixtures and pytest configuration:

- Custom async test runner (no external dependencies)
- `asyncio` marker registration
- Shared fixtures for memory registry, session context, etc.

### `backend_test_utils.py`

Utilities for backend integration tests:

- `generate_test_backend(source, output_dir)`: Generate backend from N3 source
- `load_backend_module(package_name, backend_dir)`: Dynamically import backend
- `cleanup_backend_modules(package_name)`: Clean up sys.modules after test

### Mock Providers

Located in test fixtures, not production code:

- `MockLLMProvider`: Deterministic LLM responses
- `MockVectorStore`: In-memory vector storage for RAG tests
- `MockDatabase`: Test database without external dependencies

---

## CI Integration

### GitHub Actions

Tests run automatically on:
- Every push to `main`
- All pull requests
- Nightly builds for extended test suite

### CI Configuration

`.github/workflows/test.yml`:

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest --cov=namel3ss --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Failure Criteria

CI fails if:
- Any test fails
- Coverage drops below threshold (80%)
- Any `.n3` example fails to compile
- Linting errors are present (`ruff`, `mypy`)

---

## Coverage

### Target Coverage

- **Overall**: ≥ 80%
- **Parser**: ≥ 90% (critical for language correctness)
- **Validator**: ≥ 95% (critical for error quality)
- **Codegen**: ≥ 75% (some branches are environment-specific)

### Viewing Coverage

```bash
# Generate HTML report
pytest --cov=namel3ss --cov-report=html

# Open in browser
open htmlcov/index.html
```

### Coverage Gaps

Areas with intentionally lower coverage:
- CLI interactive prompts (tested manually)
- Dev server hot reload (integration tested separately)
- Provider-specific error handling (varies by external API)

---

## Test Organization

```
tests/
├── conftest.py                          # Shared fixtures and pytest config
├── backend_test_utils.py                # Backend generation helpers
│
├── test_parser.py                       # Parser and grammar tests
├── test_loader.py                       # Module loading and discovery
├── test_resolver.py                     # Symbol resolution and type checking
│
├── test_language_integration.py         # End-to-end .n3 compilation tests
├── test_backend_integration.py          # Generated backend endpoint tests
├── test_frontend_integration.py         # React frontend integration tests
│
├── test_structured_prompts_parser.py    # Prompt syntax parsing
├── test_structured_prompts_validator.py # Output validation logic
├── test_structured_prompts_runtime.py   # Prompt execution with LLMs
│
├── test_memory_system.py                # Memory registry and operations
├── test_memory_runtime.py               # Memory in prompts and chains
│
├── test_agent_runtime.py                # Agent execution and tools
├── test_graph_integration.py            # Graph routing and state
│
├── test_logic_integration.py            # Logic rules and inference
├── test_symbolic_integration.py         # Symbolic expression evaluation
├── test_rag_runtime.py                  # RAG queries and retrieval
│
├── test_errors.py                       # Error message quality
├── test_error_messages.py               # Specific error format validation
│
├── test_frames_api.py                   # Frame endpoints
├── test_training_integration.py         # Training/tuning pipelines
└── test_eval_suites.py                  # Experiment evaluation
```

---

## Debugging Failed Tests

### View Full Output

```bash
pytest tests/test_file.py::test_function -vv
```

### Drop into Debugger on Failure

```bash
pytest tests/test_file.py::test_function --pdb
```

### Show Print Statements

```bash
pytest tests/test_file.py::test_function -s
```

### Run Only Failed Tests

```bash
pytest --lf  # last failed
pytest --ff  # failed first, then rest
```

---

## Best Practices

### ✅ DO

- Test one concept per test function
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Mock external dependencies (LLMs, databases, APIs)
- Use fixtures for common setup
- Assert on specific error messages, not just exception types
- Clean up resources (temp files, sys.modules) in `finally` blocks
- Document why a test exists (in docstring)

### ❌ DON'T

- Call real LLM APIs or external services
- Use random data without seeding
- Depend on test execution order
- Test multiple unrelated things in one function
- Use production data or credentials
- Skip cleanup (causes inter-test pollution)
- Write tests that pass/fail randomly

---

## Adding Tests for New Features

When implementing a new feature:

1. **Write tests first** (TDD approach recommended)
2. **Test parser** if adding new syntax
3. **Test resolver** if adding semantic rules
4. **Test runtime** if adding execution behavior
5. **Test errors** for all failure modes
6. **Update docs** to match test examples
7. **Add integration test** for end-to-end flow

Example workflow:

```bash
# 1. Create test file
touch tests/test_new_feature.py

# 2. Write failing test
# Edit test_new_feature.py with expected behavior

# 3. Run test (should fail)
pytest tests/test_new_feature.py -v

# 4. Implement feature
# Edit namel3ss/ files

# 5. Run test (should pass)
pytest tests/test_new_feature.py -v

# 6. Add documentation
# Edit docs/NEW_FEATURE.md

# 7. Update integration tests
# Edit tests/test_language_integration.py
```

---

## Questions?

- Check existing tests for patterns
- Review test utilities in `backend_test_utils.py`
- See `conftest.py` for shared fixtures
- Open an issue if test infrastructure is inadequate

**Last Updated**: November 2025
