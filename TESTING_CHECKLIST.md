# Namel3ss (N3) Production-Grade Testing Checklist

This checklist provides a complete, production-ready workflow for installing, testing, and validating the Namel3ss AI programming language on a fresh machine. All commands are designed to work without external API keys, using mocks and fixtures for AI provider interactions.

---

## Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Operating System**: Windows, Linux, or macOS
- **Network**: Internet required only for initial dependency installation

---

## Quick Start: Full Test Suite

For developers who want to immediately validate the entire repository:

```powershell
# Run the automated test script (handles virtualenv, dependencies, and full test suite)
.\scripts\run_tests.sh
```

**What this verifies:**
- Creates and activates virtual environment
- Installs all development dependencies
- Runs complete pytest suite with coverage reporting
- Tests pass on first run without configuration

---

## Manual Setup (Step-by-Step)

For developers who need full control or want to understand each step:

### 1. Create and Activate Virtual Environment

```powershell
# Create a Python virtual environment
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Alternative for Windows Command Prompt:
# .\.venv\Scripts\activate.bat

# Verify Python is from the virtual environment
python -c "import sys; print(sys.prefix)"
# Should show path ending in ".venv"
```

**What this verifies:** Isolated Python environment created successfully.

---

### 2. Upgrade pip (Recommended)

```powershell
# Ensure latest pip version to avoid dependency resolution issues
python -m pip install --upgrade pip
```

---

### 3. Install Namel3ss in Development Mode

```powershell
# Install namel3ss with all development and testing tools
pip install -e ".[dev]"
```

**What this installs:**
- **Core dependencies**: `fastapi`, `httpx`, `pydantic`, `uvicorn`, `sqlalchemy`, `pygls`
- **Test tools**: `pytest>=8.0`, `pytest-cov>=4.1`, `coverage>=7.6`
- **Linting/formatting**: `black>=24.0`, `ruff>=0.5`, `mypy>=1.10`
- **Build tools**: `build>=1.2`

**What this verifies:**
- Package installs without errors
- Entry point `namel3ss` CLI is available
- All dependencies resolve correctly

---

### 4. Verify CLI Installation

```powershell
# Check CLI is available and shows version
namel3ss --version
```

**Expected output:** Shows language version (e.g., `0.4.1`).

**What this verifies:**
- Package installed correctly
- CLI entry point registered
- Python can find namel3ss modules

---

## Running Tests

### Option A: Full Test Suite with Coverage (Recommended)

```powershell
# Run all tests with coverage reporting and fail on first error
pytest --maxfail=1 --disable-warnings --cov=namel3ss --cov-report=term-missing --cov-report=xml
```

**What this verifies:**
- **Parser**: `.n3` file parsing (grammar, AST generation, error handling)
- **Type Checker**: Static type validation for apps, datasets, prompts, agents
- **Code Generation**: FastAPI backend generation and React frontend scaffolding
- **Runtime Systems**:
  - Memory (session/user/global scopes with TTL)
  - Agents (multi-agent orchestration, handoffs, tool calling)
  - Structured Prompts (typed I/O validation, schema enforcement)
  - RAG (vector search, reranking, retrieval)
  - Chains (multi-step workflows, conditional execution)
  - Symbolic Expressions (eval suite, numeric operations)
- **Integrations**:
  - Backend API contracts (FastAPI endpoints, schema validation)
  - Dataset adapters (SQL, CSV, JSON, REST)
  - Connector drivers (OpenAI, Anthropic, Azure - all mocked)
  - Training pipelines (hyperparameter tuning, model registry)
  - Observability (tracing, metrics, health checks)
- **CLI Commands**: `build`, `run`, `test`, `lint`, `typecheck`, `train`, `deploy`
- **Security**: Input sanitization, SQL injection prevention, policy enforcement

**Output format:**
- Test results with PASS/FAIL status
- Coverage report showing % coverage per module
- `coverage.xml` for CI integration
- Missing coverage lines highlighted

---

### Option B: Quick Sanity Subset (Fast)

```powershell
# Run core parser, codegen, and runtime tests (~30 seconds)
pytest tests/test_parser.py tests/test_codegen.py tests/test_backend_integration.py tests/test_cli.py -v
```

**What this verifies:**
- Core parsing works (grammar, AST)
- Code generation produces valid backends
- Backend integration contracts are correct
- CLI commands execute without errors

**Use case:** Quick validation during active development.

---

### Option C: Run Tests by Category

```powershell
# Parser and type checking only
pytest tests/parser/ tests/test_type_checker.py -v

# AI/LLM features (prompts, agents, RAG, memory)
pytest tests/test_structured_prompts_*.py tests/test_agent_*.py tests/test_rag_*.py tests/test_memory_*.py -v

# Backend generation and integration
pytest tests/test_backend_integration.py tests/test_codegen.py tests/test_dataset_adapters.py -v

# Runtime systems (chains, workflows, experiments)
pytest tests/test_runtime_*.py tests/test_workflow_runtime.py tests/test_experiment_*.py -v

# Connector drivers (mocked provider interactions)
pytest tests/test_connector_drivers.py tests/test_llm_connectors.py tests/test_providers.py -v

# CLI and end-to-end
pytest tests/test_cli.py tests/test_language_integration.py tests/test_agent_e2e.py -v
```

**What this verifies:** Specific subsystems in isolation.

---

### Option D: Async Tests Only

```powershell
# Run all async/streaming tests (agent runtime, connectors, memory)
pytest -k "asyncio or streaming" -v
```

**What this verifies:** Async/await patterns, streaming responses, concurrent operations.

---

### Option E: Skip Known-Failing Tests

```powershell
# Skip tests marked with @pytest.mark.skip (e.g., legacy parser updates pending)
pytest --ignore-glob="*test_end_to_end_symbolic*" -v
```

**What this verifies:** All currently-supported features work (excludes WIP features).

---

## Static Analysis and Linting

### 1. Run Ruff (Fast Linter)

```powershell
# Check code quality and style issues
ruff check namel3ss/ tests/
```

**What this verifies:**
- Code style consistency
- Common bug patterns (unused imports, undefined names)
- Security issues (SQL injection patterns, unsafe evals)

**Note:** Ruff configuration is implicit (uses defaults). Project has `ruff>=0.5` in dev dependencies.

---

### 2. Run Black (Code Formatter - Check Only)

```powershell
# Check formatting without modifying files
black --check namel3ss/ tests/
```

**What this verifies:** Code follows Black formatting standards.

**To auto-format:**
```powershell
black namel3ss/ tests/
```

---

### 3. Run MyPy (Type Checker)

```powershell
# Static type checking for type hints
mypy namel3ss/ --ignore-missing-imports
```

**What this verifies:**
- Type hint correctness
- Function signature compatibility
- Pydantic model type safety

**Note:** Uses `--ignore-missing-imports` because some third-party stubs may be missing.

---

## Advanced Testing Scenarios

### 1. Test Backend Generation End-to-End

```powershell
# Generate backend from example, verify it runs
namel3ss generate-backend examples/provider_demo.n3 ./test_backend_output
cd test_backend_output
python -c "from main import app; print('Backend imports successfully')"
cd ..
```

**What this verifies:**
- Code generation produces valid Python
- Generated FastAPI app can be imported
- No circular imports or missing dependencies

---

### 2. Test Frontend Generation

```powershell
# Generate static frontend from example
namel3ss generate-frontend examples/provider_demo.n3 ./test_frontend_output
ls ./test_frontend_output/index.html
```

**What this verifies:**
- Frontend generator produces valid HTML/JS
- Static files are created with correct structure

---

### 3. Test CLI Type Checking

```powershell
# Run type checker on example N3 program
namel3ss typecheck examples/provider_demo.n3
```

**Expected output:** No type errors or list of detected issues.

**What this verifies:**
- Static type checker works on real programs
- Catches type mismatches in datasets, prompts, agents

---

### 4. Test CLI Linting

```powershell
# Run linter on example N3 program
namel3ss lint examples/provider_demo.n3
```

**What this verifies:**
- Style checker works on N3 code
- Detects unused variables, missing types

---

## CI/CD Integration

The repository includes GitHub Actions workflows. The commands above map directly to CI steps:

### Existing CI Configuration (`.github/workflows/tests.yml`)

```yaml
name: Tests
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  tests:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Run test suite
        run: ./scripts/run_tests.sh
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage-${{ matrix.python-version }}
          path: coverage.xml
```

**CI verifies:**
- Tests pass on Python 3.10, 3.11, and 3.12
- Cross-platform compatibility (Linux runner)
- Coverage reports are generated

---

### Recommended CI Workflow Enhancement (Optional)

For a more comprehensive CI pipeline, add these jobs:

```yaml
# Add to .github/workflows/tests.yml

lint:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - run: pip install -e ".[dev]"
    - run: ruff check namel3ss/ tests/
    - run: black --check namel3ss/ tests/
    - run: mypy namel3ss/ --ignore-missing-imports

type-check-examples:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - run: pip install -e ".[dev]"
    - run: |
        for f in examples/*.n3; do
          namel3ss typecheck "$f" || exit 1
        done
```

---

## Test Coverage Targets

Based on repository analysis, these coverage targets are recommended:

| Module | Target Coverage | Notes |
|--------|----------------|-------|
| `namel3ss.parser` | 90%+ | Core parsing well-tested |
| `namel3ss.codegen` | 85%+ | Backend/frontend generation |
| `namel3ss.types` | 90%+ | Type checker critical |
| `namel3ss.runtime` | 80%+ | Agent execution, memory |
| `namel3ss.providers` | 85%+ | Mock-based testing |
| `namel3ss.agents` | 85%+ | Runtime behavior |
| `namel3ss.cli` | 75%+ | CLI commands |

**View current coverage:**
```powershell
pytest --cov=namel3ss --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Known Test Markers and Skip Patterns

The test suite uses `@pytest.mark.skip()` for features under active development:

```python
# Examples from tests/test_end_to_end_symbolic.py
@pytest.mark.skip("Top-level rule syntax requires legacy parser update")
@pytest.mark.skip("Undefined function detection not yet implemented")
```

**To run only non-skipped tests:**
```powershell
pytest --ignore-glob="*test_end_to_end_symbolic*"
```

**To run including skipped tests (will fail):**
```powershell
pytest --run-skipped
```

---

## Troubleshooting

### Issue: `ImportError: No module named 'namel3ss'`

**Solution:**
```powershell
# Ensure installed in editable mode
pip install -e .

# Verify installation
python -c "import namel3ss; print(namel3ss.__file__)"
```

---

### Issue: `pytest: command not found`

**Solution:**
```powershell
# Install dev dependencies
pip install -e ".[dev]"

# Run via python module
python -m pytest
```

---

### Issue: Tests fail with `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```powershell
# Reinstall dependencies
pip install -e ".[dev]"
```

---

### Issue: Async tests hang or fail

**Solution:**
- Repository includes custom async test runner in `tests/conftest.py`
- Ensure no external pytest-asyncio plugin conflicts:
  ```powershell
  pip uninstall pytest-asyncio -y
  pytest tests/test_async_streaming.py -v
  ```

---

### Issue: Coverage report not generated

**Solution:**
```powershell
# Ensure pytest-cov is installed
pip install pytest-cov

# Generate coverage explicitly
pytest --cov=namel3ss --cov-report=term --cov-report=xml
```

---

## Development Workflow Recommendations

### 1. Pre-Commit Checks (Before Every Commit)

```powershell
# Format code
black namel3ss/ tests/

# Lint
ruff check namel3ss/ tests/ --fix

# Quick tests
pytest tests/test_parser.py tests/test_codegen.py -v

# Type check
mypy namel3ss/ --ignore-missing-imports
```

---

### 2. Before Opening PR

```powershell
# Full test suite
pytest --maxfail=1 --cov=namel3ss --cov-report=term-missing

# Ensure no regressions
black --check namel3ss/ tests/
ruff check namel3ss/ tests/

# Verify examples still work
namel3ss typecheck examples/provider_demo.n3
namel3ss generate-backend examples/provider_demo.n3 ./test_output
```

---

### 3. After Major Changes

```powershell
# Run full suite with verbose output
pytest -v --cov=namel3ss --cov-report=html

# Check coverage report
# Open htmlcov/index.html

# Test all Python versions (if available)
tox  # If tox.ini is configured, otherwise use virtualenvs
```

---

## Summary: One-Command Validation

To validate the entire repository is working correctly:

```powershell
# Single command for complete validation
pip install -e ".[dev]" ; pytest --maxfail=1 --cov=namel3ss --cov-report=term-missing ; ruff check namel3ss/ tests/ ; black --check namel3ss/ tests/
```

**Expected result:**
- All dependencies install without errors
- All tests pass (or show expected skips)
- Coverage > 80% overall
- No linting issues
- No formatting issues

---

## Test Execution Time Estimates

| Test Suite | Approximate Time | Use Case |
|------------|-----------------|----------|
| Quick sanity (4 core files) | ~30 seconds | Active development |
| Parser + type checker | ~1 minute | Grammar changes |
| Backend integration | ~2 minutes | Codegen changes |
| Full suite (all tests) | ~5-10 minutes | Pre-commit validation |
| Full suite + coverage + lint | ~10-15 minutes | CI/CD pipeline |

---

## Notes for CI/CD Engineers

1. **No API Keys Required**: All AI provider tests use mocks (see `tests/test_llm_connectors.py`, `tests/test_providers.py`).
2. **No External Services**: Tests do not connect to real databases, Redis, or MongoDB.
3. **Deterministic**: Tests do not depend on random data or timestamps.
4. **Parallel-Safe**: Tests use temporary directories (`tmp_path` fixture) and do not share state.
5. **Coverage Artifacts**: CI uploads `coverage.xml` for integration with CodeCov, Coveralls, etc.

---

## Additional Resources

- **Parser Tests**: `tests/parser/` directory contains modular parser test suites
- **Integration Tests**: `tests/test_backend_integration.py`, `tests/test_agent_e2e.py`
- **Example Programs**: `examples/` directory has real N3 programs for manual testing
- **Smoke Test**: `scripts/smoke_test.sh` generates backend+frontend for manual validation

---

**Last Updated**: November 2025  
**Namel3ss Version**: 0.4.1  
**Supported Python**: 3.10, 3.11, 3.12
