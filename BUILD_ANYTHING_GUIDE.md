# Build Anything with Namel3ss - Incremental Adoption Guide

**Status**: Production-Ready  
**Version**: 1.0  
**Last Updated**: November 21, 2025

---

## 🎯 Overview

Namel3ss now provides a **production-grade incremental adoption layer** that enables teams to:

1. **Build anything** - AI workflows, CRUD apps, data pipelines, APIs
2. **Adopt incrementally** - Start small, migrate gradually
3. **Never get trapped** - Always keep escape hatches to Python/external systems

This guide covers the complete architecture, patterns, and best practices for enterprise adoption.

---

## 📦 Components

### 1. N3 Python SDK (`namel3ss_sdk`)

Separate PyPI package for integrating N3 into Python projects.

**Installation**:
```bash
pip install namel3ss-sdk
```

**Key Features**:
- Remote execution (`N3Client`)
- In-process execution (`N3InProcessRuntime`)
- Type-safe configuration
- Comprehensive exception hierarchy
- Retry + circuit breaker
- OpenTelemetry support

### 2. Tool Adapter Framework

First-class adapters for external integrations:

| Adapter | Purpose | Status |
|---------|---------|--------|
| `python` | Call Python functions | ✅ Complete |
| `http` | REST/GraphQL APIs | ✅ Complete |
| `db` | Database queries (SQLAlchemy) | 🚧 In Progress |
| `queue` | Message queues (Celery/RQ/Kafka) | 🚧 In Progress |
| `model` | ML models (OpenAI/Anthropic/HF) | 🚧 In Progress |

### 3. Python FFI Standard

Blessed patterns for calling Python from N3 with type safety.

### 4. SDK Codegen CLI

Generate typed Python clients from N3 schemas:
```bash
namel3ss sdk-sync --backend https://api.example.com --out ./n3_types/
```

---

## 🏗 Deployment Patterns

### Pattern A: N3 as Remote AI Microservice

**Use Case**: Separate AI logic from main application

```python
# Python app
from namel3ss_sdk import N3Client

client = N3Client(base_url="https://ai.example.com")

# Call AI chains
result = client.chains.run("summarize", text=document)
summary = result['result']

# Call agents
agent_result = client.agents.run(
    "support_agent",
    user_input="Reset my password",
    context={"user_id": "123"}
)
```

**Architecture**:
```
┌─────────────────┐          ┌─────────────────┐
│  Python App     │  HTTP    │   N3 Service    │
│  (FastAPI/      │  ───────>│   (Chains,      │
│   Django/Flask) │          │    Agents, RAG) │
└─────────────────┘          └─────────────────┘
```

**When to Use**:
- AI logic is separate concern
- Multiple apps need AI services
- Want to scale AI independently
- Team separation (AI vs app devs)

**Configuration**:
```python
# config.py
from namel3ss_sdk import N3ClientConfig

config = N3ClientConfig(
    base_url="https://ai.example.com",
    api_token=os.environ['N3_API_TOKEN'],
    timeout=60.0,
    max_retries=3,
    verify_ssl=True,
)

client = N3Client(config=config)
```

---

### Pattern B: Embedded N3 Runtime

**Use Case**: Run N3 workflows inside Python process

```python
# Python app
from namel3ss_sdk import N3InProcessRuntime

# Load .n3 file
runtime = N3InProcessRuntime("./workflows.n3")

# Execute chains
result = runtime.chains.run("process_order", order_id=123)

# Run agents
agent_result = runtime.agents.run(
    "analyst",
    user_input="Analyze sales trends"
)

# Query RAG
docs = runtime.rag.query(
    "knowledge_base",
    query="What are our return policies?"
)
```

**Architecture**:
```
┌───────────────────────────────┐
│  Python App                   │
│  ┌─────────────────────────┐  │
│  │  N3 Runtime (embedded)  │  │
│  │  - Chains               │  │
│  │  - Agents               │  │
│  │  - RAG                  │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
```

**When to Use**:
- Simple deployment (no separate service)
- Low latency requirements
- Offline execution needed
- Prototyping/development

**Configuration**:
```python
from namel3ss_sdk import N3RuntimeConfig

config = N3RuntimeConfig(
    source_file="./workflows.n3",
    enable_cache=True,
    cache_size=1000,
    max_turns=10,
    timeout=300.0,
)

runtime = N3InProcessRuntime(config=config)
```

---

### Pattern C: Python-Driven Apps with N3 AI

**Use Case**: Traditional app with AI augmentation

```python
# main Python app
from fastapi import FastAPI
from namel3ss_sdk import N3Client

app = FastAPI()
n3 = N3Client(base_url="https://ai.internal")

@app.post("/api/support/tickets")
async def create_ticket(ticket: Ticket):
    # Core business logic in Python
    ticket_id = db.create_ticket(ticket)
    
    # AI augmentation in N3
    classification = await n3.chains.arun(
        "classify_ticket",
        text=ticket.description
    )
    
    # AI-powered assignment
    assignment = await n3.agents.arun(
        "ticket_router",
        user_input=f"Route ticket: {ticket.description}",
        context={"priority": classification['priority']}
    )
    
    # Back to Python
    db.assign_ticket(ticket_id, assignment['assigned_to'])
    
    return {"ticket_id": ticket_id, "status": "created"}
```

**Architecture**:
```
┌────────────────────────────────────┐
│  Python App (Main Logic)           │
│  - CRUD operations                 │
│  - Business rules                  │
│  - Database access                 │
│  - API endpoints                   │
│          │                          │
│          v (selective AI calls)    │
│  ┌──────────────────┐              │
│  │  N3 Client       │              │
│  │  - AI workflows  │              │
│  │  - Agents        │              │
│  └──────────────────┘              │
└────────────────────────────────────┘
```

**When to Use**:
- Existing Python app
- Add AI features incrementally
- Keep core logic in Python
- AI is enhancement, not core

---

### Pattern D: Full-Stack N3 Apps

**Use Case**: Build entire application in N3

```n3
app "CRM Platform" {
  version: "1.0"
  
  // Data layer
  dataset "customers" {
    source: db("postgresql://...")
    schema: {
      id: int
      name: string
      email: string
      created_at: timestamp
    }
  }
  
  // AI features
  chain "lead_scoring" {
    prompt: "Score this lead: {{customer.name}}"
    model: "gpt-4"
    output: float
  }
  
  agent "sales_assistant" {
    goal: "Help sales team with customer insights"
    tools: ["search_customers", "send_email"]
  }
  
  // API endpoints
  page "customers" {
    route: "/api/customers"
    query: {
      select: customers
      where: {active: true}
      order_by: created_at desc
    }
  }
  
  // UI
  view "dashboard" {
    layout: grid
    widgets: [
      {type: "chart", data: "revenue_by_month"},
      {type: "table", data: "top_customers"}
    ]
  }
}
```

**When to Use**:
- Greenfield projects
- AI-native applications
- Rapid prototyping
- Small teams

---

## 🔧 Tool Adapter Examples

### Python FFI

**Define Python function**:
```python
# myapp/tools/invoicing.py
from pydantic import BaseModel

class InvoiceInput(BaseModel):
    amount: float
    tax_rate: float
    discount: float = 0.0

class InvoiceOutput(BaseModel):
    subtotal: float
    tax: float
    total: float

async def calculate_invoice(input: InvoiceInput) -> InvoiceOutput:
    subtotal = input.amount - input.discount
    tax = subtotal * input.tax_rate
    total = subtotal + tax
    
    return InvoiceOutput(
        subtotal=subtotal,
        tax=tax,
        total=total
    )
```

**Register in N3**:
```n3
tool "calculate_invoice" {
  adapter: "python"
  module: "myapp.tools.invoicing"
  function: "calculate_invoice"
  version: "1.0"
  timeout: 30.0
  retry_policy: {
    enabled: true
    max_attempts: 3
  }
}

chain "process_order" {
  steps: [
    {
      call: "calculate_invoice"
      inputs: {
        amount: {{order.amount}}
        tax_rate: 0.08
        discount: {{order.discount}}
      }
    }
  ]
}
```

**Programmatic registration**:
```python
from namel3ss.adapters import PythonAdapter, PythonAdapterConfig, register_adapter

config = PythonAdapterConfig(
    name="calculate_invoice",
    module="myapp.tools.invoicing",
    function="calculate_invoice",
    version="1.0",
    timeout=30.0,
)

adapter = PythonAdapter(config)
register_adapter("calculate_invoice", adapter)
```

---

### HTTP Adapter

**REST API call**:
```n3
tool "fetch_weather" {
  adapter: "http"
  base_url: "https://api.weather.com"
  endpoint: "/v1/forecast"
  method: "GET"
  auth_type: "api_key"
  auth_header_name: "X-API-Key"
  auth_token: env("WEATHER_API_KEY")
  timeout: 10.0
  retry_policy: {
    enabled: true
    max_attempts: 3
    backoff_factor: 1.0
  }
}

chain "weather_report" {
  call: "fetch_weather"
  inputs: {
    location: {{user_location}}
    days: 7
  }
}
```

**GraphQL call**:
```n3
tool "query_github" {
  adapter: "http"
  base_url: "https://api.github.com"
  endpoint: "/graphql"
  method: "POST"
  auth_type: "bearer"
  auth_token: env("GITHUB_TOKEN")
  request_format: "json"
}
```

---

## 🔐 Security Best Practices

### 1. Never Log Secrets

```python
# ❌ BAD
logger.info(f"Calling API with token: {api_token}")

# ✅ GOOD
logger.info("Calling API", extra={"endpoint": "/api/v1/data"})
```

### 2. Use Environment Variables

```python
# config.py
import os

N3_API_TOKEN = os.environ['N3_API_TOKEN']
DATABASE_URL = os.environ['DATABASE_URL']

# .env (never commit!)
N3_API_TOKEN=secret_token_here
DATABASE_URL=postgresql://...
```

### 3. TLS Required

```python
# ✅ GOOD - TLS enabled
client = N3Client(
    base_url="https://api.example.com",
    verify_ssl=True  # default
)

# ❌ ONLY for local development
client = N3Client(
    base_url="http://localhost:8000",
    verify_ssl=False  # development only!
)
```

### 4. Token Rotation

```python
class TokenProvider:
    def get_token(self) -> str:
        # Fetch from secrets manager
        return secrets_manager.get_token()

client = N3Client(base_url="...")
# Refresh token periodically
token_provider = TokenProvider()
client.config.api_token = token_provider.get_token()
```

---

## 📊 Observability

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Use with N3 SDK
client = N3Client(
    base_url="https://api.example.com",
    config=N3ClientConfig(enable_tracing=True)
)

# Trace N3 calls
with tracer.start_as_current_span("process_order"):
    result = client.chains.run("calculate_total", amount=100)
```

---

## 🧪 Testing

### Unit Tests

```python
import pytest
from namel3ss_sdk import N3Client, N3TimeoutError

def test_chain_execution():
    client = N3Client(base_url="http://test.example.com")
    result = client.chains.run("test_chain", input="value")
    
    assert result['status'] == 'success'
    assert 'result' in result

def test_timeout_handling():
    client = N3Client(base_url="http://slow.example.com")
    
    with pytest.raises(N3TimeoutError):
        client.chains.run("slow_chain", timeout=0.1)

@pytest.mark.asyncio
async def test_async_execution():
    async with N3Client(base_url="http://test.example.com") as client:
        result = await client.chains.arun("async_chain", input="value")
        assert result['status'] == 'success'
```

### Integration Tests

```python
import pytest
from namel3ss_sdk import N3InProcessRuntime

@pytest.fixture
def runtime():
    return N3InProcessRuntime("./test_app.n3")

def test_end_to_end_workflow(runtime):
    # Test complete workflow
    result = runtime.chains.run(
        "process_order",
        order_id=123,
        customer_id=456
    )
    
    assert result['status'] == 'success'
    assert 'order_total' in result['result']

def test_python_ffi(runtime):
    # Test Python function integration
    result = runtime.chains.run(
        "calculate_tax",
        amount=100,
        rate=0.08
    )
    
    assert result['result']['tax'] == 8.0
```

---

## 📈 Migration Strategy

### Phase 1: Proof of Concept (Week 1-2)

1. Install SDK: `pip install namel3ss-sdk`
2. Create simple .n3 file with one chain
3. Call from Python using `N3InProcessRuntime`
4. Validate results

```python
# poc.py
from namel3ss_sdk import N3InProcessRuntime

runtime = N3InProcessRuntime("./poc.n3")
result = runtime.chains.run("hello_world", name="Team")
print(result)
```

### Phase 2: Incremental Integration (Week 3-8)

1. Identify AI-suitable tasks
2. Implement in N3 (chains, agents, RAG)
3. Replace Python AI code with N3 calls
4. Add comprehensive tests
5. Deploy to staging

```python
# Before
def summarize_document(text):
    # Complex OpenAI API calls
    ...

# After
def summarize_document(text):
    return n3_client.chains.run("summarize", text=text)
```

### Phase 3: Production Deployment (Week 9-12)

1. Deploy N3 service (if using Pattern A)
2. Configure monitoring/alerting
3. Enable circuit breakers
4. Set up token rotation
5. Production rollout with feature flags

### Phase 4: Expansion (Ongoing)

1. Add more workflows to N3
2. Build domain-specific agents
3. Integrate with more systems
4. Train team on N3 development

---

## 🎓 Best Practices Summary

### ✅ DO

- Start with embedded runtime for prototyping
- Use remote client for production
- Define clear schemas for all adapters
- Enable retries and circuit breakers
- Implement comprehensive logging
- Use OpenTelemetry for tracing
- Keep secrets in environment variables
- Write tests for all N3 workflows
- Version your N3 contracts
- Document your adapters

### ❌ DON'T

- Log secrets or PII
- Disable TLS in production
- Skip schema validation
- Ignore timeout configurations
- Deploy without circuit breakers
- Hard-code credentials
- Skip error handling
- Use untrusted Python functions
- Ignore security best practices
- Deploy without monitoring

---

## 🆘 Troubleshooting

### Connection Errors

```python
try:
    result = client.chains.run("my_chain")
except N3ConnectionError as e:
    print(f"Failed to connect: {e}")
    print("Check N3_BASE_URL and network connectivity")
```

### Timeout Issues

```python
# Increase timeout for slow operations
result = client.chains.run(
    "slow_chain",
    timeout=120.0  # 2 minutes
)
```

### Circuit Breaker Open

```python
try:
    result = client.chains.run("unreliable_chain")
except N3CircuitBreakerError:
    # Service is down, use fallback
    result = fallback_implementation()
```

---

## 📚 Additional Resources

- **API Reference**: `/namel3ss_sdk/README.md`
- **Examples**: `/examples/sdk/`
- **Migration Guide**: `/docs/MIGRATION.md`
- **Security Guide**: `/docs/SECURITY.md`
- **Performance Tuning**: `/docs/PERFORMANCE.md`

---

## 🤝 Support

- GitHub Issues: https://github.com/SsebowaDisan/namel3ss-programming-language/issues
- Documentation: https://github.com/SsebowaDisan/namel3ss-programming-language#readme
- Community: [Discord/Slack link]

---

**Implementation Status**: Phase 1 Complete (SDK + Adapters)  
**Next Steps**: DB/Queue/Model adapters, SDK codegen CLI, templates
