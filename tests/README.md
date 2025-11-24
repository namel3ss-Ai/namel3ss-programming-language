# Test Organization Structure

This document describes the organization of tests in the Namel3ss programming language project.

## Directory Structure

### Core Test Categories

#### `/agents/`
- **Purpose**: Agent system tests
- **Files**: Agent parsing, runtime, typechecking, e2e tests
- **Coverage**: Multi-agent orchestration, agent graphs, handoffs

#### `/ai/`
- **Purpose**: AI and LLM integration tests
- **Files**: Grammar integration, provider tests, model registry
- **Coverage**: AI connectors, RAG runtime, rerankers

#### `/backend/`
- **Purpose**: Backend system tests
- **Files**: Integration tests, security validation
- **Coverage**: FastAPI backend, API endpoints, middleware

#### `/cli/`
- **Purpose**: Command-line interface tests
- **Files**: CLI commands, validation, integration
- **Coverage**: All CLI functionality, argument parsing, execution

#### `/core/`
- **Purpose**: Core language system tests
- **Files**: Adapters, effects, plugins, symbolic processing
- **Coverage**: Core language features, runtime systems

#### `/data/`
- **Purpose**: Data binding and dataset tests
- **Files**: Data binding implementations, SQL compilation
- **Coverage**: Database connections, data processing

#### `/frontend/`
- **Purpose**: Frontend generation tests
- **Files**: Frontend delegation, React/Vite generation
- **Coverage**: UI generation, component systems

#### `/integration/`
- **Purpose**: Integration and end-to-end tests
- **Files**: System integration, provider integration
- **Coverage**: Complete workflows, system interactions

#### `/language/`
- **Purpose**: Language feature tests
- **Files**: Parser, type checker, resolver, language integration
- **Coverage**: Core language features, syntax, semantics

#### `/logic/`
- **Purpose**: Logic engine tests
- **Files**: Backtracking, unification, pattern matching
- **Coverage**: Constraint resolution, logical reasoning

#### `/parser/`
- **Purpose**: Parsing system tests
- **Files**: Inline blocks, parser units
- **Coverage**: AST generation, syntax parsing

#### `/providers/`
- **Purpose**: Provider system tests
- **Files**: Local providers (vLLM, Ollama, LocalAI), cloud providers
- **Coverage**: Model deployment, provider integration

#### `/runtime/`
- **Purpose**: Runtime execution tests
- **Files**: Memory systems, frame analysis, streaming
- **Coverage**: Execution engine, runtime evaluation

#### `/security/`
- **Purpose**: Security feature tests
- **Files**: Security integration, validation, IR security
- **Coverage**: Authentication, authorization, security policies

#### `/structured_prompts/`
- **Purpose**: Structured prompts system tests
- **Files**: Parser, validation, integration, runtime
- **Coverage**: Prompt templates, schema validation

#### `/system/`
- **Purpose**: System-wide tests
- **Files**: Observability, tooling integration, comprehensive suites
- **Coverage**: System monitoring, performance, reliability

### Support Directories

#### `/fixtures/`
- Test fixtures and sample data

#### `/testing/`
- Testing infrastructure and utilities

#### `/tools/`
- Testing tools and helpers

#### `/examples/`
- Example-based tests

#### `/conformance/`
- Language conformance tests

### Root Files

- `conftest.py` - PyTest configuration and fixtures
- `run_all_tests.py` - Test runner script

## Running Tests

### Run All Tests
```bash
python run_all_tests.py
```

### Run Specific Categories
```bash
# Core language tests
pytest tests/language/ -v

# AI integration tests
pytest tests/ai/ -v

# Local model provider tests
pytest tests/providers/local/ -v

# Integration tests
pytest tests/integration/ -v
```

### Run Tests by Feature
```bash
# Agent system
pytest tests/agents/ -v

# Backend functionality
pytest tests/backend/ tests/api/ -v

# Frontend generation
pytest tests/frontend/ -v

# Security features
pytest tests/security/ -v
```

## Test Coverage

The test suite provides comprehensive coverage across:

- **Language Features**: Parsing, type checking, compilation
- **AI Integration**: Providers, prompts, chains, agents
- **Runtime Systems**: Execution, memory, streaming
- **Backend Generation**: FastAPI, endpoints, middleware  
- **Frontend Generation**: React, components, routing
- **Security**: Authentication, validation, policies
- **Performance**: Benchmarks, load testing, optimization
- **Integration**: End-to-end workflows, system validation

## Contributing

When adding new tests:

1. **Place in appropriate directory** based on functionality
2. **Follow naming convention**: `test_<feature>_<aspect>.py`
3. **Add to relevant test suites** in documentation
4. **Update this README** if adding new test categories

## Test Infrastructure

- **PyTest**: Primary testing framework
- **Fixtures**: Shared test data in `/fixtures/`
- **Utilities**: Test helpers in `/testing/`
- **Mocking**: Comprehensive mocks for external dependencies
- **Coverage**: Automated coverage reporting
- **CI/CD**: Automated test execution in pipelines