# Test Fixtures

This directory contains test fixtures organized by functionality for the Namel3ss test suite.

## Structure

```
tests/
├── unit/fixtures/
│   ├── agents/          # Agent definition fixtures
│   ├── prompts/         # Prompt template fixtures  
│   ├── llms/           # LLM configuration fixtures
│   └── syntax/         # Syntax testing fixtures
└── integration/fixtures/
    └── templates/      # Complete app templates for integration tests
```

## Unit Test Fixtures

### Agents (`unit/fixtures/agents/`)
- `simple_agent.ai` - Basic agent with conversation memory
- `content_analyzer.ai` - Content analysis agent fixture

### Prompts (`unit/fixtures/prompts/`)
- `greeting.ai` - Simple greeting template with variables
- `analysis.ai` - Content analysis prompt template

### LLMs (`unit/fixtures/llms/`)
- `openai.ai` - OpenAI GPT configuration
- `ollama.ai` - Local Ollama model configuration

### Syntax (`unit/fixtures/syntax/`)
- `dashboard.ai` - Dashboard syntax examples
- `metrics.ai` - Metrics collection syntax
- `syntax_error.ai` - Intentional syntax errors for error testing
- `type_error.ai` - Type checking error examples

## Integration Test Fixtures

### Templates (`integration/fixtures/templates/`)
- `minimal_app.ai` - Minimal complete application
- `agent_app.ai` - Agent-based application template

## Usage

### In Unit Tests
```python
import pytest
from pathlib import Path

@pytest.fixture
def agent_fixture():
    return Path(__file__).parent / "unit/fixtures/agents/simple_agent.ai"

def test_agent_parsing(agent_fixture):
    # Test code here
    pass
```

### In Integration Tests
```python
from pathlib import Path

def test_template_builds():
    template = Path(__file__).parent / "integration/fixtures/templates/minimal_app.ai"
    # Build and test template
```

## Maintenance

- Keep fixtures simple and focused on specific functionality
- Update fixtures when core syntax changes
- Add new fixtures for new language features
- Ensure all fixtures build successfully with `namel3ss build` or `namel3ss check`