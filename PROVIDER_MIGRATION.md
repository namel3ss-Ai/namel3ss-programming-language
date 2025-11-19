# Migration Guide: BaseLLM to N3Provider

This guide helps you migrate from the legacy `BaseLLM` system to the new `N3Provider` system in Namel3ss.

## Why Migrate?

The N3Provider system offers:

- **Async-First**: Better performance with native async/await
- **Streaming Support**: Consistent streaming API across all providers
- **Batch Processing**: Efficient concurrent request handling
- **Environment Configuration**: No hard-coded secrets, secure by default
- **Better Error Handling**: Structured errors with retry logic
- **Production Ready**: Comprehensive logging, metrics, and observability
- **Unified Interface**: Single API for all LLM backends

## Migration Overview

| Feature | BaseLLM (Old) | N3Provider (New) |
|---------|---------------|------------------|
| Sync/Async | Sync | Async-first |
| Configuration | Constructor args | Environment variables |
| Streaming | Varies by provider | Unified async iterator |
| Batching | Manual loops | Built-in with concurrency |
| Error Handling | Generic exceptions | Structured ProviderError |
| Type Hints | Partial | Full type annotations |
| Testing | Mocking required | Built-in test providers |

## Step-by-Step Migration

### Step 1: Update Configuration

**Before (BaseLLM):**
```python
from namel3ss.llm.openai_llm import OpenAILLM

llm = OpenAILLM(
    model="gpt-4",
    api_key="sk-...",  # Hard-coded secret!
    organization_id="org-...",
)
```

**After (N3Provider):**
```bash
# Set environment variables (in .env or shell)
export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."
export NAMEL3SS_PROVIDER_OPENAI_ORG_ID="org-..."
```

```python
from namel3ss.providers import create_provider_from_spec

provider = create_provider_from_spec("openai", "gpt-4")
```

### Step 2: Update Generation Calls

**Before (BaseLLM):**
```python
from namel3ss.llm.base import ChatMessage

messages = [
    ChatMessage(role="system", content="You are helpful."),
    ChatMessage(role="user", content="What is 2+2?"),
]

response = llm.generate_chat(messages, temperature=0.7)
print(response.text)
print(response.usage)
```

**After (N3Provider):**
```python
from namel3ss.providers import ProviderMessage
import asyncio

messages = [
    ProviderMessage(role="system", content="You are helpful."),
    ProviderMessage(role="user", content="What is 2+2?"),
]

async def generate():
    response = await provider.generate(messages, temperature=0.7)
    print(response.output_text)
    print(response.usage)

asyncio.run(generate())
```

### Step 3: Update Streaming

**Before (BaseLLM):**
```python
# OpenAI
for chunk in llm.stream_chat(messages):
    print(chunk, end="", flush=True)

# Anthropic (different API)
stream = llm.stream(messages)
for event in stream:
    if event.type == "content_block_delta":
        print(event.delta.text, end="", flush=True)
```

**After (N3Provider):**
```python
async def stream_generate():
    async for chunk in provider.stream(messages):
        print(chunk, end="", flush=True)

asyncio.run(stream_generate())
```

### Step 4: Update Batch Processing

**Before (BaseLLM):**
```python
results = []
for batch_messages in all_messages:
    response = llm.generate_chat(batch_messages)
    results.append(response)
```

**After (N3Provider):**
```python
async def batch_generate():
    responses = await provider.generate_batch(
        all_messages,
        max_concurrent=5,  # Control concurrency
    )
    return responses

results = asyncio.run(batch_generate())
```

### Step 5: Update Error Handling

**Before (BaseLLM):**
```python
try:
    response = llm.generate_chat(messages)
except Exception as e:
    print(f"Error: {e}")
```

**After (N3Provider):**
```python
from namel3ss.providers import ProviderError

async def safe_generate():
    try:
        response = await provider.generate(messages)
    except ProviderError as e:
        print(f"Provider error: {e}")
        print(f"Status code: {e.status_code}")
        print(f"Details: {e.details}")
        # Handle specific error types
        if e.status_code == 429:
            # Rate limit - retry with backoff
            pass
        elif e.status_code == 401:
            # Auth error - check credentials
            pass

asyncio.run(safe_generate())
```

## Integration Patterns

### Pattern 1: Compatibility Bridge (Minimal Changes)

Use `ProviderLLMBridge` to wrap N3Provider as BaseLLM:

```python
from namel3ss.providers import create_provider_from_spec
from namel3ss.providers.integration import ProviderLLMBridge

# Create provider
provider = create_provider_from_spec("openai", "gpt-4")

# Wrap as BaseLLM
llm = ProviderLLMBridge(provider, default_temperature=0.7)

# Use with existing code that expects BaseLLM
agent_runtime = AgentRuntime(agent_def, llm, tools)
```

**Pros:**
- Minimal code changes
- Works with existing BaseLLM-dependent code
- Easy incremental migration

**Cons:**
- Doesn't take full advantage of async
- Bridge overhead
- Some features may not map perfectly

### Pattern 2: Full Async Migration (Recommended)

Migrate to async/await throughout:

```python
from namel3ss.providers import create_provider_from_spec

async def main():
    provider = create_provider_from_spec("openai", "gpt-4")
    
    # Agent execution
    from namel3ss.providers.integration import run_agent_with_provider
    result = await run_agent_with_provider(
        agent_def=agent,
        provider=provider,
        user_input="Analyze data",
        tools=tools,
    )
    
    # Chain execution
    from namel3ss.providers.integration import run_chain_with_provider
    output = await run_chain_with_provider(
        chain_steps=chain.steps,
        provider=provider,
        initial_input={"question": "What is AI?"},
    )

asyncio.run(main())
```

**Pros:**
- Best performance
- Full access to provider features
- Clean, modern code

**Cons:**
- Requires refactoring to async
- All callers must be async

### Pattern 3: Hybrid Approach

Mix sync bridge for compatibility, async for new code:

```python
# Existing code (sync)
llm_bridge = ProviderLLMBridge(provider)
legacy_result = llm_bridge.generate_chat(messages)

# New code (async)
async def new_feature():
    response = await provider.generate(messages)
    return response

# Run new code from sync context
result = asyncio.run(new_feature())
```

## Provider-Specific Migration

### OpenAI

**Before:**
```python
from namel3ss.llm.openai_llm import OpenAILLM

llm = OpenAILLM(
    model="gpt-4",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    organization_id="org-...",
)
```

**After:**
```bash
export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."
export NAMEL3SS_PROVIDER_OPENAI_BASE_URL="https://api.openai.com/v1"
export NAMEL3SS_PROVIDER_OPENAI_ORG_ID="org-..."
```

```python
provider = create_provider_from_spec("openai", "gpt-4")
```

### Anthropic

**Before:**
```python
from namel3ss.llm.anthropic_llm import AnthropicLLM

llm = AnthropicLLM(
    model="claude-3-sonnet-20240229",
    api_key="sk-ant-...",
)
```

**After:**
```bash
export NAMEL3SS_PROVIDER_ANTHROPIC_API_KEY="sk-ant-..."
```

```python
provider = create_provider_from_spec("anthropic", "claude-3-sonnet-20240229")
```

### Google Vertex AI

**Before:**
```python
from namel3ss.llm.vertex_llm import VertexLLM

llm = VertexLLM(
    model="gemini-pro",
    project_id="my-project",
    location="us-central1",
)
```

**After:**
```bash
export NAMEL3SS_PROVIDER_GOOGLE_PROJECT_ID="my-project"
export NAMEL3SS_PROVIDER_GOOGLE_LOCATION="us-central1"
export NAMEL3SS_PROVIDER_GOOGLE_USE_VERTEX="true"
```

```python
provider = create_provider_from_spec("google", "gemini-pro")
```

### Azure OpenAI

**Before:**
```python
from namel3ss.llm.azure_openai_llm import AzureOpenAILLM

llm = AzureOpenAILLM(
    model="gpt-4",
    api_key="...",
    endpoint="https://xxx.openai.azure.com/",
    deployment="gpt-4-prod",
    api_version="2024-02-15-preview",
)
```

**After:**
```bash
export NAMEL3SS_PROVIDER_AZURE_OPENAI_API_KEY="..."
export NAMEL3SS_PROVIDER_AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com/"
export NAMEL3SS_PROVIDER_AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

```python
provider = create_provider_from_spec("azure_openai", "gpt-4")
```

### Local (Ollama)

**Before:**
```python
from namel3ss.llm.ollama_llm import OllamaLLM

llm = OllamaLLM(
    model="llama2",
    base_url="http://localhost:11434",
)
```

**After:**
```bash
export NAMEL3SS_PROVIDER_LOCAL_BASE_URL="http://localhost:11434"
```

```python
provider = create_provider_from_spec("local", "llama2")
```

## Common Migration Issues

### Issue 1: Sync to Async

**Problem:**
```python
# Old sync code
def process_request(text):
    response = llm.generate_chat(messages)
    return response.text
```

**Solution:**
```python
# Convert to async
async def process_request(text):
    response = await provider.generate(messages)
    return response.output_text

# Or use bridge
def process_request(text):
    llm_bridge = ProviderLLMBridge(provider)
    response = llm_bridge.generate_chat(messages)
    return response.text
```

### Issue 2: Different Response Attributes

**Problem:**
```python
# Old: response.text
print(response.text)

# Old: response.usage
print(response.usage["prompt_tokens"])
```

**Solution:**
```python
# New: response.output_text
print(response.output_text)

# New: response.prompt_tokens (property)
print(response.prompt_tokens)
```

### Issue 3: Environment Variables Not Loaded

**Problem:**
```python
# Provider fails with missing API key
provider = create_provider_from_spec("openai", "gpt-4")
# ProviderError: Missing required config: api_key
```

**Solution:**
```bash
# Ensure env vars are set
export NAMEL3SS_PROVIDER_OPENAI_API_KEY="sk-..."

# Or use .env file
echo "NAMEL3SS_PROVIDER_OPENAI_API_KEY=sk-..." >> .env

# Load with python-dotenv
from dotenv import load_dotenv
load_dotenv()
```

### Issue 4: Streaming API Differences

**Problem:**
```python
# Old: Different streaming APIs per provider
if isinstance(llm, OpenAILLM):
    for chunk in llm.stream_chat(messages):
        print(chunk)
elif isinstance(llm, AnthropicLLM):
    stream = llm.stream(messages)
    for event in stream:
        print(event.delta.text)
```

**Solution:**
```python
# New: Unified streaming API
async for chunk in provider.stream(messages):
    print(chunk)
```

## Testing Migration

### Before (BaseLLM):

```python
from unittest.mock import Mock

def test_llm_generation():
    llm = Mock()
    llm.generate_chat.return_value = Mock(
        text="Response",
        usage={"prompt_tokens": 10, "completion_tokens": 20}
    )
    
    response = llm.generate_chat(messages)
    assert response.text == "Response"
```

### After (N3Provider):

```python
from unittest.mock import AsyncMock
import pytest

@pytest.mark.asyncio
async def test_provider_generation():
    provider = AsyncMock()
    provider.generate.return_value = ProviderResponse(
        model="test",
        output_text="Response",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
        raw={},
    )
    
    response = await provider.generate(messages)
    assert response.output_text == "Response"
```

Or use built-in test provider:

```python
from namel3ss.providers.factory import register_provider_class

class TestProvider(N3Provider):
    async def generate(self, messages, **kwargs):
        return ProviderResponse(
            model=self.model,
            output_text="Test response",
            usage={},
            raw={},
        )

register_provider_class("test", TestProvider)
provider = create_provider_from_spec("test", "test-model")
```

## Migration Checklist

- [ ] Set up environment variables for all providers
- [ ] Update imports from `namel3ss.llm` to `namel3ss.providers`
- [ ] Convert `ChatMessage` to `ProviderMessage`
- [ ] Update `generate_chat()` calls to `generate()`
- [ ] Convert sync code to async or use `ProviderLLMBridge`
- [ ] Update streaming to use unified async iterator
- [ ] Replace manual batching with `generate_batch()`
- [ ] Update error handling to use `ProviderError`
- [ ] Update response attribute access (`text` → `output_text`)
- [ ] Update tests to use async fixtures
- [ ] Remove hard-coded API keys from code
- [ ] Update documentation and examples
- [ ] Test thoroughly with real providers
- [ ] Monitor logs and metrics

## Rollback Plan

If you need to rollback:

1. Keep both systems during migration:
```python
from namel3ss.llm.openai_llm import OpenAILLM  # Old
from namel3ss.providers import create_provider_from_spec  # New

# Use feature flags to switch
if USE_NEW_PROVIDER:
    provider = create_provider_from_spec("openai", "gpt-4")
    llm = ProviderLLMBridge(provider)
else:
    llm = OpenAILLM(model="gpt-4", api_key=api_key)
```

2. The old `BaseLLM` system is still available and functional
3. Use `ProviderLLMBridge` for gradual migration
4. Both systems can coexist during transition

## Timeline Recommendation

**Week 1-2: Setup**
- Set up environment variables
- Install updated dependencies
- Create test environments

**Week 3-4: Core Migration**
- Migrate core generation calls
- Update agent and chain execution
- Add async support to main workflows

**Week 5-6: Advanced Features**
- Migrate streaming implementations
- Add batch processing where beneficial
- Update error handling

**Week 7-8: Testing & Validation**
- Comprehensive testing with all providers
- Performance benchmarking
- Documentation updates

**Week 9+: Rollout**
- Gradual rollout to production
- Monitor metrics and errors
- Collect feedback and iterate

## Support

For migration questions or issues:
- Check the [N3Provider documentation](PROVIDER_SYSTEM.md)
- Review [examples](examples/provider_demo.n3)
- Open an issue on GitHub
- Ask in community Discord

## Conclusion

The N3Provider system represents a major improvement in Namel3ss's LLM integration capabilities. While migration requires some effort, the benefits in performance, security, and maintainability are substantial.

Start with the compatibility bridge (`ProviderLLMBridge`) for a low-risk migration, then gradually move to full async implementation for maximum benefit.
