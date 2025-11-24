# Namel3ss HTTP Runtime

FastAPI/HTTP runtime adapter for the Namel3ss programming language.

## Overview

This package adapts Namel3ss intermediate representation (IR) to HTTP/REST APIs using FastAPI. It's one of several runtime implementations for Namel3ss, demonstrating that **Namel3ss is a language that targets multiple runtimes**, not a single-platform SaaS.

## Installation

```bash
pip install namel3ss-runtime-http
```

Or install with optional features:

```bash
# With WebSocket support
pip install namel3ss-runtime-http[websockets]

# With real-time streaming (WebSocket + Redis)
pip install namel3ss-runtime-http[realtime]

# Development dependencies
pip install namel3ss-runtime-http[dev]
```

## Usage

### Generate FastAPI Backend from IR

```python
from namel3ss import Parser, build_backend_ir
from namel3ss_runtime_http import generate_fastapi_backend

# Parse .ai source
source = '''
app "MyApp" connects to postgres "DB".

prompt "Greet" {
    model: "gpt-4o-mini"
    template: "Say hello to {{name}}."
}
'''

parser = Parser(source)
module = parser.parse()
app = module.body[0]

# Build runtime-agnostic IR
ir = build_backend_ir(app)

# Generate FastAPI backend
generate_fastapi_backend(ir, output_dir="backend/")
```

### Run Development Server

```bash
# Install the backend dependencies
cd backend
pip install -r requirements.txt

# Start uvicorn server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## What Gets Generated

The HTTP runtime generates:

```
backend/
├── main.py                    # FastAPI application entry point
├── database.py                # Database connection setup
├── runtime.py                 # Runtime support (LLM calls, etc.)
├── generated/
│   ├── __init__.py
│   ├── runtime.py             # Generated runtime logic
│   ├── routers/               # API routers
│   │   ├── __init__.py
│   │   ├── prompts.py         # Prompt endpoints
│   │   ├── agents.py          # Agent endpoints
│   │   ├── datasets.py        # Dataset endpoints
│   │   ├── frames.py          # Frame endpoints
│   │   └── pages.py           # Page endpoints
│   ├── helpers/               # Helper utilities
│   └── schemas/               # Pydantic schemas
└── custom/                    # User customization space
    ├── routes/
    └── README.md
```

## API Endpoints

Generated endpoints follow consistent patterns:

- **Prompts:** `POST /api/prompts/{name}` - Execute structured prompts
- **Agents:** `POST /api/agents/{name}` - Run multi-agent workflows
- **Tools:** `POST /api/tools/{name}` - Invoke tools
- **Datasets:** `GET /api/datasets/{name}` - Query datasets
- **Frames:** `GET /api/frames/{name}` - Access frame data
- **Pages:** `GET /pages/{slug}` - Serve page data

## Features

### ✅ HTTP/REST APIs
- Automatic route generation from IR
- Request/response validation
- Error handling
- CORS support

### ✅ Real-time Streaming
- WebSocket support for streaming responses
- Server-sent events (SSE)
- Progress updates during LLM generation

### ✅ Database Integration
- PostgreSQL, MySQL, SQLite support
- Automatic connection pooling
- Migration support

### ✅ Authentication & Security
- API key authentication
- JWT token support
- Rate limiting
- Request validation

### ✅ Observability
- Structured logging
- Metrics endpoints
- Health checks
- OpenTelemetry integration

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...

# Server
PORT=8000
HOST=0.0.0.0
RELOAD=true

# Features
ENABLE_REALTIME=true
ENABLE_CORS=true
```

### Programmatic Configuration

```python
from namel3ss_runtime_http import generate_fastapi_backend

generate_fastapi_backend(
    ir=backend_ir,
    output_dir="backend/",
    enable_realtime=True,      # WebSocket support
    embed_insights=False,       # Insight embedding strategy
    connector_config={
        "retry_limit": 3,
        "timeout": 30,
    },
    export_schemas=True,        # Export OpenAPI schemas
    schema_version="1.0.0",
)
```

## Architecture

### IR → FastAPI Mapping

The HTTP runtime consumes Namel3ss IR and maps it to FastAPI constructs:

| IR Component | FastAPI Construct |
|--------------|-------------------|
| `EndpointIR` | `@app.get/post()` route |
| `PromptSpec` | Async function with LLM call |
| `AgentSpec` | Multi-step orchestration handler |
| `ToolSpec` | Function with input validation |
| `DatasetSpec` | Query endpoint with filtering |
| `MemorySpec` | Session/conversation state management |

### Request Flow

```
HTTP Request
    ↓
FastAPI Router
    ↓
Pydantic Validation
    ↓
Generated Handler (from IR)
    ↓
    ├─→ LLM Call (Prompts/Agents)
    ├─→ Database Query (Datasets/Frames)
    ├─→ Tool Execution
    └─→ Business Logic
    ↓
Response Serialization
    ↓
HTTP Response
```

## Advanced Usage

### Custom Middleware

```python
# backend/custom/middleware.py
from fastapi import Request

async def custom_middleware(request: Request, call_next):
    # Custom logic
    response = await call_next(request)
    return response
```

### Custom Routes

```python
# backend/custom/routes/custom_api.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/custom/hello")
async def hello():
    return {"message": "Hello from custom route"}
```

### Extend Generated Code

The `backend/custom/` directory is preserved across regenerations:

```
backend/custom/
├── __init__.py
├── routes/
│   ├── __init__.py
│   └── custom_api.py      # Your custom routes
├── middleware.py           # Your middleware
└── README.md              # Customization guide
```

## Development

### Run Tests

```bash
cd runtimes/http
pytest
```

### Build Package

```bash
python -m build
```

### Install Locally

```bash
pip install -e .
```

## Relationship to Core

The HTTP runtime **depends on** the Namel3ss language core:

```
namel3ss (core)       ← Provides IR types, parser, type checker
    ↑
    | imports
    |
namel3ss-runtime-http ← Consumes IR, generates FastAPI
```

**Dependency rules:**
- ✅ HTTP runtime can import from `namel3ss` core
- ❌ Core CANNOT import from HTTP runtime
- ✅ HTTP runtime is independent of other runtimes

## Alternative Runtimes

Namel3ss supports multiple runtime targets:

- **namel3ss-runtime-http** (this package) - FastAPI/HTTP
- **namel3ss-runtime-frontend** - Static sites, React apps
- **namel3ss-runtime-deploy** - Docker, Kubernetes, cloud platforms
- **Custom runtimes** - Build your own! (gRPC, GraphQL, serverless, etc.)

See [docs/RUNTIME_GUIDE.md](../../docs/RUNTIME_GUIDE.md) for creating custom runtimes.

## License

MIT License - see LICENSE file for details.

## Links

- **Repository:** https://github.com/SsebowaDisan/namel3ss-programming-language
- **Documentation:** https://github.com/SsebowaDisan/namel3ss-programming-language/tree/main/runtimes/http
- **Issues:** https://github.com/SsebowaDisan/namel3ss-programming-language/issues
- **Language Core:** https://github.com/SsebowaDisan/namel3ss-programming-language/tree/main/namel3ss
