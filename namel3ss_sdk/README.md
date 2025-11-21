# Namel3ss Python SDK

Production-grade Python SDK for integrating Namel3ss into existing applications without migrating your entire stack.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Zero-config defaults** with type-safe configuration
- **Remote + in-process execution** modes
- **Automatic retries** with exponential backoff
- **Circuit breaker** for fault tolerance
- **OpenTelemetry** instrumentation built-in
- **Comprehensive exception hierarchy** with request ID tracking
- **Full async support** for high-performance applications

## Installation

```bash
pip install namel3ss-sdk
```

With OpenTelemetry support:
```bash
pip install namel3ss-sdk[telemetry]
```

## Quick Start

### Remote Execution

Call N3 chains/agents/RAG running on a remote server:

```python
from namel3ss_sdk import N3Client

# Initialize client
client = N3Client(base_url="https://ai.example.com")

# Execute chain
result = client.chains.run(
    "summarize",
    text="Long document here...",
    max_length=100
)
print(result['result'])

# Run agent
agent_result = client.agents.run(
    "support_agent",
    user_input="Reset my password",
    context={"user_id": "123"}
)

# Query RAG pipeline
docs = client.rag.query(
    "knowledge_base",
    query="What are our return policies?",
    top_k=5
)
```

### In-Process Execution

Run N3 workflows directly in your Python process:

```python
from namel3ss_sdk import N3InProcessRuntime

# Load .n3 file
runtime = N3InProcessRuntime("./app.n3")

# Execute chain
result = runtime.chains.run("summarize", text="...")

# Run agent
agent_result = runtime.agents.run(
    "analyst",
    user_input="Analyze sales trends"
)
```

### Async Support

```python
from namel3ss_sdk import N3Client

async def process_requests():
    async with N3Client(base_url="...") as client:
        results = await asyncio.gather(
            client.chains.arun("summarize", text="Doc 1"),
            client.chains.arun("summarize", text="Doc 2"),
            client.chains.arun("summarize", text="Doc 3"),
        )
        return results
```

## Configuration

### From Environment Variables

```bash
export N3_BASE_URL="https://ai.example.com"
export N3_API_TOKEN="your-token"
export N3_TIMEOUT=60.0
export N3_MAX_RETRIES=3
```

```python
from namel3ss_sdk import N3Client

# Auto-loads from environment
client = N3Client()
```

### Explicit Configuration

```python
from namel3ss_sdk import N3Client, N3ClientConfig

config = N3ClientConfig(
    base_url="https://ai.example.com",
    api_token="your-token",
    timeout=60.0,
    max_retries=3,
    retry_backoff_factor=1.0,
    retry_backoff_max=60.0,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0,
    verify_ssl=True,
)

client = N3Client(config=config)
```

### From .env File

```ini
# .env
N3_BASE_URL=https://ai.example.com
N3_API_TOKEN=your-token
N3_TIMEOUT=60.0
N3_MAX_RETRIES=3
```

```python
from namel3ss_sdk import N3Client

client = N3Client()  # Auto-loads .env
```

## Error Handling

The SDK provides comprehensive exception hierarchy:

```python
from namel3ss_sdk import (
    N3Error,              # Base exception
    N3ClientError,        # 4xx errors
    N3ServerError,        # 5xx errors
    N3TimeoutError,       # Request timeout
    N3AuthError,          # Authentication failed
    N3ConnectionError,    # Network error
    N3RateLimitError,     # Rate limit exceeded
    N3RuntimeError,       # Execution error
    N3SchemaError,        # Validation error
    N3CircuitBreakerError,# Circuit breaker open
)

try:
    result = client.chains.run("summarize", text="...")
except N3TimeoutError:
    print("Request timed out - retry with longer timeout")
except N3RateLimitError as e:
    print(f"Rate limited - retry after {e.retry_after} seconds")
except N3ClientError as e:
    print(f"Client error: {e.message} (status={e.status_code})")
except N3Error as e:
    print(f"SDK error: {e.message} (request_id={e.request_id})")
```

## Advanced Features

### Circuit Breaker

Automatically fails fast when service is down:

```python
config = N3ClientConfig(
    base_url="https://ai.example.com",
    circuit_breaker_threshold=5,     # Open after 5 failures
    circuit_breaker_timeout=60.0,    # Test recovery after 60s
)

client = N3Client(config=config)

try:
    result = client.chains.run("my_chain")
except N3CircuitBreakerError:
    # Service is down, use fallback
    result = fallback_implementation()
```

### Retry with Backoff

Automatic retries with exponential backoff:

```python
config = N3ClientConfig(
    max_retries=3,              # Retry up to 3 times
    retry_backoff_factor=1.0,   # Backoff multiplier
    retry_backoff_max=60.0,     # Max 60s backoff
)
```

### OpenTelemetry Tracing

```python
from opentelemetry import trace
from namel3ss_sdk import N3Client, N3ClientConfig

# Setup tracing
tracer = trace.get_tracer(__name__)

config = N3ClientConfig(
    enable_tracing=True,
    service_name="my-app"
)

client = N3Client(config=config)

# Trace N3 calls
with tracer.start_as_current_span("process_order"):
    result = client.chains.run("calculate_total", amount=100)
```

### Custom Timeout

Override default timeout per request:

```python
# Default timeout
result = client.chains.run("fast_chain", text="...")

# Custom timeout for slow operation
result = client.chains.run(
    "slow_chain",
    text="...",
    timeout=120.0  # 2 minutes
)
```

## API Reference

### N3Client

```python
class N3Client:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        config: Optional[N3ClientConfig] = None,
    )
    
    # Chains API
    chains.run(name: str, timeout: Optional[float] = None, **payload) -> Dict
    chains.arun(name: str, timeout: Optional[float] = None, **payload) -> Dict
    
    # Prompts API
    prompts.run(name: str, timeout: Optional[float] = None, **inputs) -> Dict
    prompts.arun(name: str, timeout: Optional[float] = None, **inputs) -> Dict
    
    # Agents API
    agents.run(name: str, user_input: str, context: Optional[Dict] = None,
               max_turns: Optional[int] = None, timeout: Optional[float] = None) -> Dict
    agents.arun(name: str, user_input: str, ...) -> Dict
    
    # RAG API
    rag.query(pipeline: str, query: str, top_k: Optional[int] = None,
              filters: Optional[Dict] = None, timeout: Optional[float] = None) -> Dict
    rag.aquery(pipeline: str, query: str, ...) -> Dict
```

### N3InProcessRuntime

```python
class N3InProcessRuntime:
    def __init__(
        self,
        source_file: Optional[str] = None,
        config: Optional[N3RuntimeConfig] = None,
    )
    
    # Same API as N3Client
    chains.run(name: str, **payload) -> Dict
    prompts.run(name: str, **inputs) -> Dict
    agents.run(name: str, user_input: str, ...) -> Dict
    rag.query(pipeline: str, query: str, ...) -> Dict
    
    # Additional methods
    get_chains() -> Dict[str, Any]
    get_prompts() -> Dict[str, Any]
    get_agents() -> Dict[str, Any]
    get_rag_pipelines() -> Dict[str, Any]
    get_tools() -> Dict[str, Any]
    execute_raw(func_name: str, *args, **kwargs) -> Any
```

## Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from namel3ss_sdk import N3Client

app = FastAPI()
n3 = N3Client(base_url="https://ai.example.com")

@app.post("/api/summarize")
async def summarize(text: str):
    result = await n3.chains.arun("summarize", text=text)
    return {"summary": result['result']}

@app.post("/api/chat")
async def chat(message: str):
    result = await n3.agents.arun(
        "chatbot",
        user_input=message
    )
    return {"response": result['response']}
```

### Django Integration

```python
# views.py
from django.http import JsonResponse
from namel3ss_sdk import N3Client

n3 = N3Client(base_url="https://ai.example.com")

def summarize_article(request):
    text = request.POST.get('text')
    result = n3.chains.run("summarize", text=text)
    return JsonResponse(result)
```

### Background Jobs (Celery)

```python
# tasks.py
from celery import shared_task
from namel3ss_sdk import N3Client

n3 = N3Client(base_url="https://ai.example.com")

@shared_task
def process_document(doc_id):
    doc = Document.objects.get(id=doc_id)
    
    # Run N3 chain
    result = n3.chains.run(
        "analyze_document",
        text=doc.content,
        metadata={"doc_id": doc_id}
    )
    
    doc.analysis = result['result']
    doc.save()
```

## Testing

```python
import pytest
from namel3ss_sdk import N3Client, N3TimeoutError

@pytest.fixture
def client():
    return N3Client(base_url="http://test.example.com")

def test_chain_execution(client):
    result = client.chains.run("test_chain", input="value")
    assert result['status'] == 'success'

def test_timeout(client):
    with pytest.raises(N3TimeoutError):
        client.chains.run("slow_chain", timeout=0.1)

@pytest.mark.asyncio
async def test_async(client):
    result = await client.chains.arun("async_chain", input="value")
    assert result['status'] == 'success'
```

## Security

### Never Log Secrets

```python
# ❌ BAD
logger.info(f"Token: {api_token}")

# ✅ GOOD
logger.info("Making API call", extra={"endpoint": "/chains/summarize"})
```

### Use Environment Variables

```python
import os

# ✅ GOOD
api_token = os.environ['N3_API_TOKEN']

# ❌ BAD
api_token = "hardcoded-secret"
```

### TLS Required

```python
# ✅ GOOD - Production
client = N3Client(
    base_url="https://ai.example.com",
    verify_ssl=True
)

# ❌ ONLY for local development
client = N3Client(
    base_url="http://localhost:8000",
    verify_ssl=False
)
```

## Performance Tips

1. **Use async for concurrent requests**:
   ```python
   results = await asyncio.gather(
       client.chains.arun("chain1", ...),
       client.chains.arun("chain2", ...),
   )
   ```

2. **Reuse client instances**:
   ```python
   # ✅ GOOD - Reuse connection pool
   client = N3Client(...)
   for item in items:
       result = client.chains.run(...)
   ```

3. **Configure timeouts appropriately**:
   ```python
   # Short timeout for fast operations
   result = client.chains.run("fast", timeout=5.0)
   
   # Longer timeout for complex operations
   result = client.chains.run("complex", timeout=120.0)
   ```

4. **Enable circuit breaker for resilience**:
   ```python
   config = N3ClientConfig(
       circuit_breaker_threshold=5,
       circuit_breaker_timeout=60.0
   )
   ```

## Troubleshooting

### Connection Refused

```
N3ConnectionError: Failed to connect to https://ai.example.com
```

**Solution**: Check N3_BASE_URL and network connectivity.

### Authentication Failed

```
N3AuthError: Unauthorized
```

**Solution**: Verify N3_API_TOKEN is correct and not expired.

### Timeout

```
N3TimeoutError: Request timed out after 30.0s
```

**Solution**: Increase timeout or optimize N3 workflow.

### Circuit Breaker Open

```
N3CircuitBreakerError: Circuit breaker is OPEN - too many failures
```

**Solution**: Service is down. Wait 60s or use fallback.

## Migration from Direct API Calls

### Before (Direct HTTP)

```python
import httpx

response = httpx.post(
    "https://ai.example.com/api/chains/summarize/execute",
    json={"text": "..."},
    headers={"Authorization": "Bearer ..."}
)
result = response.json()
```

### After (SDK)

```python
from namel3ss_sdk import N3Client

client = N3Client(base_url="https://ai.example.com")
result = client.chains.run("summarize", text="...")
```

**Benefits**:
- Automatic retries
- Circuit breaker
- Type safety
- Better error handling
- OpenTelemetry tracing
- Less boilerplate

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md).

## License

MIT License - see [LICENSE](../LICENSE).

## Support

- **Issues**: https://github.com/SsebowaDisan/namel3ss-programming-language/issues
- **Docs**: https://github.com/SsebowaDisan/namel3ss-programming-language#readme
- **Examples**: `/examples/sdk/`

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
