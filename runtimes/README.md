# Namel3ss Runtime Packages

This directory contains runtime implementations for the Namel3ss language.

## Architecture

Namel3ss is a **language that targets multiple runtimes**, not a single-platform SaaS. The language core (in `namel3ss/`) compiles `.ai` programs to an intermediate representation (IR), and runtime packages adapt that IR to concrete execution environments.

## Available Runtimes

### HTTP Runtime (`runtimes/http/`)

Adapts Namel3ss IR to HTTP/REST APIs using FastAPI.

**Responsibilities:**
- Convert IR endpoints to FastAPI routes
- Generate FastAPI application scaffold
- HTTP server management (via uvicorn)
- Request/response handling
- WebSocket support for streaming

**Installation:**
```bash
pip install namel3ss-runtime-http
```

**Usage:**
```python
from namel3ss import build_backend_ir
from namel3ss_runtime_http import generate_fastapi_backend

ir = build_backend_ir(app)
generate_fastapi_backend(ir, output_dir="backend/")
```

### Frontend Runtime (`runtimes/frontend/`)

Generates frontend applications from Namel3ss IR.

**Responsibilities:**
- Convert IR pages to HTML/JS/React
- Static site generation
- React/Vite project scaffolding
- Component generation from IR specs

**Installation:**
```bash
pip install namel3ss-runtime-frontend
```

**Usage:**
```python
from namel3ss import build_frontend_ir
from namel3ss_runtime_frontend import generate_static_site

ir = build_frontend_ir(app)
generate_static_site(ir, output_dir="build/")
```

### Deploy Runtime (`runtimes/deploy/`)

Deployment adapters for Namel3ss applications.

**Responsibilities:**
- Docker image generation
- Kubernetes manifest generation
- Cloud platform adapters (AWS, GCP, Azure)
- Container orchestration

**Installation:**
```bash
pip install namel3ss-runtime-deploy
```

## Dependency Direction

```
namel3ss (language core)
    ↑
    | depends on
    |
runtimes/* (runtime implementations)
```

**Rules:**
- ✅ Runtime packages can import from `namel3ss` core
- ❌ Core package CANNOT import from runtime packages
- ✅ Runtime packages are independent of each other

## Adding a New Runtime

To create a custom runtime (e.g., gRPC, GraphQL, serverless):

1. **Create runtime package:**
   ```bash
   mkdir -p runtimes/my_runtime/namel3ss_runtime_my_runtime
   ```

2. **Create adapter:**
   ```python
   # runtimes/my_runtime/namel3ss_runtime_my_runtime/adapter.py
   from namel3ss import BackendIR
   
   def adapt_to_my_runtime(ir: BackendIR):
       """Convert Namel3ss IR to your runtime"""
       # Implementation
   ```

3. **Create pyproject.toml:**
   ```toml
   [project]
   name = "namel3ss-runtime-my-runtime"
   dependencies = ["namel3ss>=0.5.0"]
   ```

4. **Implement IR consumption:**
   - Read `ir.endpoints` to generate API handlers
   - Read `ir.prompts` to wire LLM calls
   - Read `ir.agents` for orchestration logic
   - etc.

See `docs/RUNTIME_GUIDE.md` for detailed instructions.

## Testing

Each runtime package has its own test suite:

```bash
# Test HTTP runtime
cd runtimes/http
pytest

# Test frontend runtime
cd runtimes/frontend
pytest

# Test deploy runtime
cd runtimes/deploy
pytest
```

## Development

Runtime packages are independently versioned and can evolve separately from the language core, as long as they consume the stable IR format.

**IR Compatibility:**
- Runtime packages depend on IR version (e.g., IR v0.1.0)
- Breaking IR changes require runtime updates
- IR is versioned separately from language syntax
