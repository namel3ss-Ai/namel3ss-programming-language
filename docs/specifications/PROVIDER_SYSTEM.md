# N3Provider System Documentation

## Overview

The N3Provider system is a production-grade abstraction layer for unified LLM backend management in Namel3ss. It provides a single, consistent interface for all LLM providers with async-first generation, streaming, batching, and environment-based configuration.

## Architecture

### Core Components

```
namel3ss/providers/
├── base.py              # N3Provider interface, base types
├── config.py            # Environment-based configuration
├── factory.py           # Provider instantiation and registry
├── integration.py       # Integration with existing runtime
├── openai_provider.py   # OpenAI implementation
├── anthropic_provider.py # Anthropic Claude implementation
├── google_provider.py   # Google Vertex AI / Gemini implementation
├── azure_openai_provider.py # Azure OpenAI implementation
├── local_provider.py    # Local engines (vLLM, Ollama)
├── http_provider.py     # Generic HTTP endpoints
└── __init__.py          # Public API exports
```

### Key Features

- **Unified Interface**: Single `N3Provider` abstract class for all backends
- **Async-First**: All generation is async with `asyncio`
- **Streaming Support**: Consistent streaming API across providers
- **Batch Generation**: Efficient batch processing with concurrency control
- **Environment Configuration**: No hard-coded secrets, all from env vars
- **Factory Pattern**: Create providers from specs with lazy loading
- **Registry System**: Manage provider lifecycles and reuse instances
- **Production Ready**: Comprehensive error handling, logging, observability

## Quick Start

### Installation

The provider system is included with Namel3ss. Optional provider dependencies:

```bash
# OpenAI
pip install httpx

# Anthropic
pip install httpx

# Google Vertex AI
pip install google-auth httpx

# All providers
pip install httpx google-auth
```

### Basic Usage

```python
from namel3ss.providers import create_provider_from_spec, ProviderMessage

# Create provider (reads from environment)
provider = create_provider_from_spec("openai", "gpt-4")

# Generate response
messages = [
    ProviderMessage(role="system", content="You are helpful."),
    ProviderMessage(role="user", content="What is 2+2?"),
]

response = await provider.generate(messages, temperature=0.7)
print(response.output_text)  # "4"
print(response.total_tokens)  # 45
```

### Streaming

```python
# Stream response chunks
async for chunk in provider.stream(messages):
    print(chunk, end="", flush=True)
```

### Batch Processing

```python
# Process multiple inputs efficiently
batch = [
    [ProviderMessage(role="user", content="Question 1")],
    [ProviderMessage(role="user", content="Question 2")],
    [ProviderMessage(role="user", content="Question 3")],
]

responses = await provider.generate_batch(batch, max_concurrent=5)
for response in responses:
    print(response.output_text)
```

## Configuration

### Environment Variables

All provider configuration uses environment variables with the pattern:

```
NAMEL3SS_PROVIDER_{TYPE}_{KEY}
```

Example:

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

# Local (Ollama)
export NAMEL3SS_PROVIDER_LOCAL_BASE_URL="http://localhost:11434"

# Local (vLLM)
export NAMEL3SS_PROVIDER_LOCAL_BASE_URL="http://localhost:8000"

# HTTP (Generic)
export NAMEL3SS_PROVIDER_HTTP_ENDPOINT="https://my-api.com/generate"
export NAMEL3SS_PROVIDER_HTTP_AUTH_HEADER="Authorization"
export NAMEL3SS_PROVIDER_HTTP_AUTH_VALUE="Bearer token-..."
```

### Configuration Loading

```python
from namel3ss.providers.config import load_provider_config

# Load from environment
config = load_provider_config("openai")
print(config)  # {"api_key": "sk-...", "base_url": "...", ...}

# Load with DSL overrides
dsl_config = {"temperature": 0.9, "max_tokens": 2000}
merged = merge_configs("openai", dsl_config)
```

## Provider Types

### OpenAI

```python
provider = create_provider_from_spec("openai", "gpt-4")
provider = create_provider_from_spec("openai", "gpt-3.5-turbo")
provider = create_provider_from_spec("openai", "gpt-4-turbo-preview")
```

**Configuration:**
- `NAMEL3SS_PROVIDER_OPENAI_API_KEY` (required)
- `NAMEL3SS_PROVIDER_OPENAI_BASE_URL` (default: "https://api.openai.com/v1")
- `NAMEL3SS_PROVIDER_OPENAI_ORG_ID` (optional)

**Features:**
- Full streaming support
- Batch via concurrent requests
- Function calling support (via raw format)

### Anthropic

```python
provider = create_provider_from_spec("anthropic", "claude-3-opus-20240229")
provider = create_provider_from_spec("anthropic", "claude-3-sonnet-20240229")
provider = create_provider_from_spec("anthropic", "claude-3-haiku-20240307")
```

**Configuration:**
- `NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY` (required)
- `NAMEL3SS_PROVIDER_ANTHROPIC_BASE_URL` (default: "https://api.anthropic.com")

**Features:**
- Full streaming support
- System message handling (separate from messages array)
- Usage mapping (input_tokens → prompt_tokens)

### Google (Vertex AI / Gemini)

```python
# Vertex AI (requires GCP credentials)
provider = create_provider_from_spec("google", "gemini-pro")

# Gemini API (requires API key)
provider = create_provider_from_spec("vertex", "gemini-pro")
```

**Configuration (Vertex AI):**
- `NAMEL3SS_PROVIDER_GOOGLE_PROJECT_ID` (required)
- `NAMEL3SS_PROVIDER_GOOGLE_LOCATION` (default: "us-central1")
- `NAMEL3SS_PROVIDER_GOOGLE_USE_VERTEX` (default: "true")

**Configuration (Gemini API):**
- `NAMEL3SS_PROVIDER_GOOGLE_API_KEY` (required)
- `NAMEL3SS_PROVIDER_GOOGLE_USE_VERTEX` (default: "false")

**Features:**
- Dual mode: Vertex AI or Gemini API
- GCP authentication (service account or ADC)
- Full streaming support
- Role mapping (assistant → model)

### Azure OpenAI

```python
provider = create_provider_from_spec("azure_openai", "gpt-4")
```

**Configuration:**
- `NAMEL3SS_PROVIDER_AZURE_OPENAI_API_KEY` (required)
- `NAMEL3SS_PROVIDER_AZURE_OPENAI_ENDPOINT` (required)
- `NAMEL3SS_PROVIDER_AZURE_OPENAI_API_VERSION` (default: "2024-02-15-preview")

**Features:**
- Deployment-based routing
- API versioning in URL
- OpenAI-compatible format
- Full streaming support

### Local (vLLM / Ollama)

```python
# Auto-detects engine type
provider = create_provider_from_spec("local", "llama2")
provider = create_provider_from_spec("ollama", "mistral")
provider = create_provider_from_spec("vllm", "meta-llama/Llama-2-7b-hf")
```

**Configuration:**
- `NAMEL3SS_PROVIDER_LOCAL_BASE_URL` (default: "http://localhost:11434")

**Features:**
- Auto-detection of Ollama vs vLLM
- Format switching based on engine
- Longer timeout (120s default)
- Full streaming support

### HTTP (Generic)

```python
provider = create_provider_from_spec("http", "custom-model")
```

**Configuration:**
- `NAMEL3SS_PROVIDER_HTTP_ENDPOINT` (required)
- `NAMEL3SS_PROVIDER_HTTP_AUTH_HEADER` (optional)
- `NAMEL3SS_PROVIDER_HTTP_AUTH_VALUE` (optional)
- `NAMEL3SS_PROVIDER_HTTP_REQUEST_FORMAT` (default: "openai")
- `NAMEL3SS_PROVIDER_HTTP_RESPONSE_FORMAT` (default: "openai")

**Features:**
- Configurable request/response formats
- Custom auth headers
- OpenAI-compatible default
- Extensible for custom protocols

## Provider Registry

### Basic Registry Usage

```python
from namel3ss.providers import ProviderRegistry

# Create registry
registry = ProviderRegistry()

# Create and register provider
provider = registry.create_and_register(
    name="my_model",
    provider_type="openai",
    model="gpt-4",
    config={"temperature": 0.7},
)

# Get existing provider
provider = registry.get("my_model")

# List all providers
names = registry.list_providers()

# Clear registry
registry.clear()
```

### Context Manager

```python
# Automatic cleanup
async with ProviderRegistry() as registry:
    provider = registry.create_and_register("model1", "openai", "gpt-4")
    
    # Use provider
    response = await provider.generate(messages)
    
# Registry automatically cleared on exit
```

### Global Registry

```python
from namel3ss.providers import register_provider_instance, get_provider_instance

# Register globally
register_provider_instance("chat_model", provider)

# Retrieve anywhere in code
provider = get_provider_instance("chat_model")
```

## Integration with Existing Runtime

### Agent Integration

```python
from namel3ss.providers import create_provider_from_spec
from namel3ss.providers.integration import ProviderLLMBridge
from namel3ss.agents.runtime import AgentRuntime

# Create provider
provider = create_provider_from_spec("openai", "gpt-4")

# Wrap as BaseLLM
llm = ProviderLLMBridge(provider, default_temperature=0.7)

# Use with agent
agent_runtime = AgentRuntime(agent_def, llm, tools)
result = agent_runtime.act("Analyze sales data")
```

### Chain Integration

```python
from namel3ss.providers.integration import run_chain_with_provider

# Execute chain with provider
result = await run_chain_with_provider(
    chain_steps=chain.steps,
    provider=provider,
    initial_input={"question": "What is AI?"},
)

print(result["response"])
```

### RAG Integration

```python
# Use provider for RAG generation
contexts = retrieve_documents(query)

messages = [
    ProviderMessage(role="system", content="Answer using contexts."),
    ProviderMessage(role="user", content=f"Contexts: {contexts}\n\nQuestion: {query}"),
]

response = await provider.generate(messages)
```

### Eval Integration

```python
from namel3ss.providers.integration import ProviderLLMBridge

# Create provider and bridge
provider = create_provider_from_spec("anthropic", "claude-3-sonnet")
llm = ProviderLLMBridge(provider)

# Use with eval suite
eval_runner = EvalRunner(eval_suite, llm)
results = eval_runner.run()
```

## Advanced Usage

### Custom Providers

```python
from namel3ss.providers.base import N3Provider, ProviderResponse

class CustomProvider(N3Provider):
    def __init__(self, name: str, model: str, config=None):
        super().__init__(name, model, config)
        # Initialize client
    
    async def generate(self, messages, **kwargs):
        # Implement generation
        return ProviderResponse(
            model=self.model,
            output_text="...",
            raw={},
            usage={},
        )
    
    def supports_streaming(self):
        return True
    
    async def stream(self, messages, **kwargs):
        # Implement streaming
        yield "chunk 1"
        yield "chunk 2"

# Register custom provider
from namel3ss.providers.factory import register_provider_class
register_provider_class("custom", CustomProvider)

# Use custom provider
provider = create_provider_from_spec("custom", "my-model")
```

### Error Handling

```python
from namel3ss.providers import ProviderError

try:
    response = await provider.generate(messages)
except ProviderError as e:
    print(f"Provider error: {e}")
    print(f"Status: {e.status_code}")
    print(f"Details: {e.details}")
```

### Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("namel3ss.providers")
logger.setLevel(logging.DEBUG)

# Provider operations will log detailed info
provider = create_provider_from_spec("openai", "gpt-4")
response = await provider.generate(messages)
```

### Observability

```python
# Providers automatically emit metrics
# Metrics include:
# - provider.generate.duration
# - provider.generate.tokens
# - provider.stream.duration
# - provider.batch.duration
# - provider.error.count

# Access via observability system
from namel3ss.observability import get_metrics
metrics = get_metrics()
```

## Migration from BaseLLM

### Before (BaseLLM)

```python
from namel3ss.llm.factory import create_llm_from_spec

llm = create_llm_from_spec("openai", "gpt-4", api_key="sk-...")
response = llm.generate_chat(messages)
print(response.text)
```

### After (N3Provider)

```python
from namel3ss.providers import create_provider_from_spec
import asyncio

# Set environment variable first
# export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."

provider = create_provider_from_spec("openai", "gpt-4")
response = asyncio.run(provider.generate(messages))
print(response.output_text)
```

### Compatibility Bridge

```python
from namel3ss.providers.integration import ProviderLLMBridge

# Use provider as BaseLLM (sync compatible)
provider = create_provider_from_spec("openai", "gpt-4")
llm = ProviderLLMBridge(provider)

# Works with existing code
response = llm.generate_chat(messages)  # Sync call
response = await llm.agenerate_chat(messages)  # Async call
```

## Testing

### Unit Tests

```bash
pytest tests/test_providers.py -v
```

### Integration Tests

```bash
pytest tests/test_provider_integration.py -v
```

### With Real APIs

```bash
# Set environment variables
export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."

# Run tests
pytest tests/test_providers.py::test_openai_provider_generate_mock -v
```

## Performance

### Benchmarks

Typical generation times (single request, gpt-4):
- OpenAI: ~2-4s
- Anthropic: ~3-5s
- Google Vertex: ~2-3s
- Azure OpenAI: ~2-4s
- Local (vLLM): ~0.5-2s

Batch processing (10 requests, max_concurrent=5):
- ~2x faster than sequential
- Respects rate limits
- Efficient connection pooling

### Optimization Tips

1. **Reuse Providers**: Create once, use many times
2. **Use Batching**: For multiple inputs, use `generate_batch()`
3. **Stream When Possible**: Lower latency for long responses
4. **Configure Timeouts**: Adjust for model speed
5. **Connection Pooling**: Automatic with httpx.AsyncClient

## Security

### Best Practices

1. **Never Hard-Code Secrets**: Always use environment variables
2. **Use Least Privilege**: API keys should have minimal permissions
3. **Rotate Keys Regularly**: Update credentials periodically
4. **Monitor Usage**: Track API usage and costs
5. **Validate Inputs**: Sanitize user inputs before generation
6. **Handle Errors Gracefully**: Fail closed, don't leak details

### Environment Security

```bash
# Use .env files (never commit)
echo "NAMEL3SS_PROVIDER_OPENAI_API_KEY=sk-..." >> .env
echo ".env" >> .gitignore

# Or use secret management
export NAMEL3SS_PROVIDER_OPENAI_API_KEY=$(vault kv get secret/openai)
```

## Troubleshooting

### Common Issues

**Provider not found:**
```python
# Check registration
from namel3ss.providers.factory import list_provider_types
print(list_provider_types())  # ['openai', 'anthropic', ...]
```

**Configuration not loaded:**
```python
# Debug config loading
from namel3ss.providers.config import load_provider_config
config = load_provider_config("openai")
print(config)  # Check what was loaded
```

**Import errors:**
```bash
# Install optional dependencies
pip install httpx google-auth
```

**Async errors:**
```python
# Use asyncio.run() for top-level
import asyncio
response = asyncio.run(provider.generate(messages))
```

**Timeout errors:**
```python
# Increase timeout
response = await provider.generate(messages, timeout=120)
```

## API Reference

### N3Provider

```python
class N3Provider:
    async def generate(
        self,
        messages: List[ProviderMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ProviderResponse
    
    async def stream(
        self,
        messages: List[ProviderMessage],
        **kwargs,
    ) -> AsyncIterable[str]
    
    async def generate_batch(
        self,
        batch: List[List[ProviderMessage]],
        max_concurrent: int = 5,
        **kwargs,
    ) -> List[ProviderResponse]
    
    def supports_streaming(self) -> bool
```

### ProviderMessage

```python
@dataclass
class ProviderMessage:
    role: str  # "system", "user", "assistant"
    content: str
```

### ProviderResponse

```python
@dataclass
class ProviderResponse:
    model: str
    output_text: str
    raw: Dict[str, Any]
    usage: Dict[str, Any]
    finish_reason: Optional[str] = None
    
    @property
    def prompt_tokens(self) -> int
    
    @property
    def completion_tokens(self) -> int
    
    @property
    def total_tokens(self) -> int
```

### Factory Functions

```python
def create_provider_from_spec(
    provider_type: str,
    model: str,
    config: Optional[Dict[str, Any]] = None,
) -> N3Provider

def register_provider_class(
    provider_type: str,
    provider_class: Type[N3Provider],
) -> None

def get_provider_class(
    provider_type: str,
) -> Type[N3Provider]
```

## Changelog

### Version 1.0.0 (2024)

- Initial production release
- Support for 6 provider types (OpenAI, Anthropic, Google, Azure, Local, HTTP)
- Async-first generation with streaming
- Batch processing with concurrency control
- Environment-based configuration
- Factory and registry patterns
- Integration with existing runtime
- Comprehensive tests and documentation

## Contributing

### Adding New Providers

1. Create `namel3ss/providers/{name}_provider.py`
2. Implement `N3Provider` interface
3. Register in `factory.py`
4. Add configuration loader in `config.py`
5. Export in `__init__.py`
6. Add tests in `tests/test_providers.py`
7. Update documentation

### Code Standards

- Type hints required
- Async-first design
- Comprehensive error handling
- Logging at appropriate levels
- Docstrings for all public APIs
- Tests for all functionality

## License

See LICENSE file in repository root.

## Support

For issues, questions, or contributions:
- GitHub Issues: [github.com/namel3ss/issues]
- Documentation: [docs.namel3ss.dev]
- Community: [discord.gg/namel3ss]
