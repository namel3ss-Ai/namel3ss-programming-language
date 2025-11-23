# N3Provider System

Production-grade LLM provider abstraction for Namel3ss.

## Features

- ✅ **Unified Interface**: Single async API for all LLM backends
- ✅ **6 Provider Types**: OpenAI, Anthropic, Google, Azure, Local, HTTP
- ✅ **Async-First**: Native async/await with excellent performance
- ✅ **Streaming Support**: Consistent streaming across all providers
- ✅ **Batch Processing**: Efficient concurrent request handling
- ✅ **Environment Config**: Secure configuration via environment variables
- ✅ **Production Ready**: Comprehensive error handling, logging, metrics
- ✅ **Type Safe**: Full type annotations with mypy support
- ✅ **Test Ready**: Built-in test providers and mocking support

## Quick Start

```python
from namel3ss.providers import create_provider_from_spec, ProviderMessage
import asyncio

# Set environment variable first
# export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."

async def main():
    # Create provider
    provider = create_provider_from_spec("openai", "gpt-4")
    
    # Generate response
    messages = [
        ProviderMessage(role="system", content="You are helpful."),
        ProviderMessage(role="user", content="What is 2+2?"),
    ]
    
    response = await provider.generate(messages, temperature=0.7)
    print(response.output_text)  # "4"
    print(response.total_tokens)  # 45

asyncio.run(main())
```

## Installation

```bash
# Base providers (no optional dependencies)
pip install namel3ss

# With HTTP client support
pip install httpx

# With Google Cloud support
pip install httpx google-auth

# All providers
pip install httpx google-auth
```

## Supported Providers

| Provider | Type | Features |
|----------|------|----------|
| OpenAI | `openai` | GPT-4, GPT-3.5, streaming, function calling |
| Anthropic | `anthropic` | Claude 3 (Opus, Sonnet, Haiku), streaming |
| Google | `google`, `vertex` | Gemini Pro, Vertex AI, streaming |
| Azure OpenAI | `azure_openai` | All Azure OpenAI models, streaming |
| Local | `local`, `ollama`, `vllm` | Ollama, vLLM, local engines |
| HTTP | `http` | Any HTTP LLM API endpoint |

## Configuration

### Environment Variables

```bash
# OpenAI
export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."
export NAMEL3SS_PROVIDER_OPENAI_BASE_URL="https://api.openai.com/v1"
export NAMEL3SS_PROVIDER_OPENAI_ORG_ID="org-..."

# Anthropic
export NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY="sk-ant-..."
export NAMEL3SS_PROVIDER_ANTHROPIC_BASE_URL="https://api.anthropic.com"

# Google Vertex AI
export NAMEL3SS_PROVIDER_GOOGLE_PROJECT_ID="my-project"
export NAMEL3SS_PROVIDER_GOOGLE_LOCATION="us-central1"
export NAMEL3SS_PROVIDER_GOOGLE_USE_VERTEX="true"

# Google Gemini API
export NAMEL3SS_PROVIDER_GOOGLE_API_KEY="..."
export NAMEL3SS_PROVIDER_GOOGLE_USE_VERTEX="false"

# Azure OpenAI
export NAMEL3SS_PROVIDER_AZURE_OPENAI_API_KEY="..."
export NAMEL3SS_PROVIDER_AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com/"
export NAMEL3SS_PROVIDER_AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# Local
export NAMEL3SS_PROVIDER_LOCAL_BASE_URL="http://localhost:11434"

# HTTP
export NAMEL3SS_PROVIDER_HTTP_ENDPOINT="https://my-api.com/generate"
export NAMEL3SS_PROVIDER_HTTP_AUTH_HEADER="Authorization"
export NAMEL3SS_PROVIDER_HTTP_AUTH_VALUE="Bearer token"
```

## Usage Examples

### Basic Generation

```python
from namel3ss.providers import create_provider_from_spec, ProviderMessage

provider = create_provider_from_spec("openai", "gpt-4")

messages = [ProviderMessage(role="user", content="Hello!")]
response = await provider.generate(messages)

print(response.output_text)
print(f"Tokens: {response.total_tokens}")
```

### Streaming

```python
async for chunk in provider.stream(messages):
    print(chunk, end="", flush=True)
```

### Batch Processing

```python
batch = [
    [ProviderMessage(role="user", content="Question 1")],
    [ProviderMessage(role="user", content="Question 2")],
    [ProviderMessage(role="user", content="Question 3")],
]

responses = await provider.generate_batch(batch, max_concurrent=5)
for response in responses:
    print(response.output_text)
```

### Agent Integration

```python
from namel3ss.providers.integration import ProviderLLMBridge
from namel3ss.agents.runtime import AgentRuntime

provider = create_provider_from_spec("anthropic", "claude-3-sonnet")
llm = ProviderLLMBridge(provider)

agent = AgentRuntime(agent_def, llm, tools)
result = agent.act("Analyze sales data")
```

### Chain Integration

```python
from namel3ss.providers.integration import run_chain_with_provider

result = await run_chain_with_provider(
    chain_steps=chain.steps,
    provider=provider,
    initial_input={"question": "What is AI?"},
)
```

## Provider Registry

```python
from namel3ss.providers import ProviderRegistry

# Create registry
async with ProviderRegistry() as registry:
    # Create and register providers
    gpt4 = registry.create_and_register("gpt4", "openai", "gpt-4")
    claude = registry.create_and_register("claude", "anthropic", "claude-3-sonnet")
    
    # Use providers
    response1 = await gpt4.generate(messages)
    response2 = await claude.generate(messages)
    
    # Retrieve providers
    gpt4_again = registry.get("gpt4")
    
    # List all
    all_providers = registry.list_providers()

# Registry automatically cleaned up
```

## Error Handling

```python
from namel3ss.providers import ProviderError

try:
    response = await provider.generate(messages)
except ProviderError as e:
    print(f"Error: {e}")
    print(f"Status: {e.status_code}")
    print(f"Provider: {e.provider_name}")
    
    if e.status_code == 429:
        # Rate limit - implement backoff
        await asyncio.sleep(10)
    elif e.status_code == 401:
        # Auth error - check credentials
        raise
```

## Testing

```python
import pytest
from unittest.mock import AsyncMock
from namel3ss.providers import ProviderResponse

@pytest.mark.asyncio
async def test_generation():
    # Mock provider
    provider = AsyncMock()
    provider.generate.return_value = ProviderResponse(
        model="test",
        output_text="Test response",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        raw={},
    )
    
    # Test
    response = await provider.generate(messages)
    assert response.output_text == "Test response"
```

## Documentation

- **[Full Documentation](../../PROVIDER_SYSTEM.md)**: Complete system documentation
- **[Migration Guide](../../PROVIDER_MIGRATION.md)**: Migrate from BaseLLM
- **[Examples](../../examples/provider_demo.ai)**: Example .ai files
- **[Advanced Examples](../../examples/advanced_providers.ai)**: Advanced patterns

## Architecture

```
namel3ss/providers/
├── base.py              # N3Provider interface and base types
├── config.py            # Environment-based configuration
├── factory.py           # Provider creation and registry
├── integration.py       # Integration with Namel3ss runtime
├── openai_provider.py   # OpenAI implementation
├── anthropic_provider.py # Anthropic implementation
├── google_provider.py   # Google/Vertex implementation
├── azure_openai_provider.py # Azure OpenAI implementation
├── local_provider.py    # Local engines (Ollama, vLLM)
└── http_provider.py     # Generic HTTP endpoints
```

## Performance

Benchmarks (single request, gpt-4):
- Generation: ~2-4s
- Streaming: ~0.5s to first token
- Batch (10 requests, concurrency=5): ~2x faster than sequential

Optimizations:
- Automatic connection pooling
- Efficient async/await usage
- Configurable timeouts and retries
- Batch processing with concurrency control

## Security

- ✅ No hard-coded secrets
- ✅ Environment-based configuration
- ✅ Fail-closed error handling
- ✅ Input validation
- ✅ Secure logging (no credential leaks)
- ✅ Rate limiting support

## Contributing

1. **Add New Provider**:
   - Create `{name}_provider.py`
   - Implement `N3Provider` interface
   - Register in `factory.py`
   - Add config loader in `config.py`
   - Export in `__init__.py`
   - Add tests
   - Update documentation

2. **Code Standards**:
   - Type hints required
   - Async-first design
   - Comprehensive error handling
   - Logging at appropriate levels
   - Docstrings for public APIs
   - Tests for all functionality

## Troubleshooting

### Provider not found
```python
from namel3ss.providers.factory import list_provider_types
print(list_provider_types())
```

### Configuration not loaded
```python
from namel3ss.providers.config import load_provider_config
config = load_provider_config("openai")
print(config)
```

### Missing dependencies
```bash
pip install httpx google-auth
```

### Async errors
```python
import asyncio
response = asyncio.run(provider.generate(messages))
```

## Changelog

### Version 1.0.0 (2024)
- Initial release
- 6 provider implementations
- Async-first generation
- Streaming support
- Batch processing
- Environment configuration
- Registry system
- Integration adapters
- Comprehensive tests

## License

See LICENSE in repository root.

## Support

- GitHub Issues: [Report bugs or request features]
- Documentation: PROVIDER_SYSTEM.md
- Examples: examples/provider_demo.ai
- Migration: PROVIDER_MIGRATION.md
