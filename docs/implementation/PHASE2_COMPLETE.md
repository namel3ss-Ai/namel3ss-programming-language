# Phase 2 Complete: Runtime Package Structure

**Status:** ✅ Complete  
**Date:** January 2025  
**Objective:** Physically separate runtime-specific code into independent packages

## Overview

Phase 2 establishes the physical separation between the Namel3ss language core and runtime implementations by creating dedicated runtime packages with proper dependency boundaries.

## What Was Created

### Runtime Package Structure

```
runtimes/
├── README.md                                    # Runtime architecture documentation
├── http/                                        # HTTP/FastAPI runtime
│   ├── README.md                                # HTTP runtime documentation
│   ├── pyproject.toml                           # HTTP dependencies (FastAPI, uvicorn)
│   └── namel3ss_runtime_http/
│       ├── __init__.py                          # Public API
│       └── adapter.py                           # generate_fastapi_backend()
├── frontend/                                    # Frontend generation runtime
│   ├── README.md                                # Frontend runtime documentation
│   ├── pyproject.toml                           # Frontend dependencies
│   └── namel3ss_runtime_frontend/
│       ├── __init__.py                          # Public API
│       └── adapter.py                           # generate_static_site(), generate_react_app()
└── deploy/                                      # Deployment runtime
    ├── README.md                                # Deploy runtime documentation
    ├── pyproject.toml                           # Deploy dependencies (Docker, K8s)
    └── namel3ss_runtime_deploy/
        ├── __init__.py                          # Public API
        └── adapter.py                           # generate_docker(), generate_kubernetes()
```

### Key Files Created

#### 1. Runtime Adapters (3 files)

**`runtimes/http/namel3ss_runtime_http/adapter.py`** (98 lines)
- `generate_fastapi_backend(ir, output_dir, **kwargs)` - FastAPI backend generation
- Wraps existing `namel3ss.codegen.backend.generate_backend()`
- Uses `_original_app` from IR metadata during Phase 2 transition

**`runtimes/frontend/namel3ss_runtime_frontend/adapter.py`** (105 lines)
- `generate_static_site(ir, output_dir, **kwargs)` - Static HTML/CSS/JS generation
- `generate_react_app(ir, output_dir, **kwargs)` - React + Vite app generation
- Wraps existing `namel3ss.codegen.frontend.generate_site()`
- Uses `_original_app` from IR metadata during Phase 2 transition

**`runtimes/deploy/namel3ss_runtime_deploy/adapter.py`** (281 lines)
- `generate_docker(ir, output_dir, **kwargs)` - Dockerfile and docker-compose.yml
- `generate_kubernetes(ir, output_dir, **kwargs)` - K8s deployment manifests
- `generate_aws_config(ir, output_dir, **kwargs)` - AWS ECS task definitions
- `generate_gcp_config(ir, output_dir, **kwargs)` - Placeholder for GCP Cloud Run
- `generate_azure_config(ir, output_dir, **kwargs)` - Placeholder for Azure

#### 2. Package Configurations (3 files)

**`runtimes/http/pyproject.toml`**
```toml
[project]
name = "namel3ss-runtime-http"
dependencies = [
    "namel3ss>=0.5.0",  # Core language (IR types)
    "fastapi>=0.110.0",
    "uvicorn>=0.30.0",
    "httpx>=0.27.0",
]
```

**`runtimes/frontend/pyproject.toml`**
```toml
[project]
name = "namel3ss-runtime-frontend"
dependencies = [
    "namel3ss>=0.5.0",  # Core language (IR types)
]
```

**`runtimes/deploy/pyproject.toml`**
```toml
[project]
name = "namel3ss-runtime-deploy"
dependencies = [
    "namel3ss>=0.5.0",  # Core language (IR types)
]
[project.optional-dependencies]
aws = ["boto3>=1.34", "awscli>=1.32"]
gcp = ["google-cloud-run>=0.10"]
azure = ["azure-cli>=2.60"]
k8s = ["kubernetes>=29.0"]
```

#### 3. Documentation (4 files)

**`runtimes/README.md`** (220 lines)
- Overview of runtime architecture
- Dependency rules and enforcement
- How to create custom runtimes
- Examples of using runtime adapters

**`runtimes/http/README.md`** (300+ lines)
- HTTP runtime features and usage
- Installation instructions
- Generated file structure
- API endpoint patterns
- Configuration options
- Advanced customization

**`runtimes/frontend/README.md`** (350+ lines)
- Frontend runtime features (static + React)
- Installation and usage
- Generated file structure
- Component architecture
- Theming and styling
- Deployment strategies

**`runtimes/deploy/README.md`** (400+ lines)
- Deployment runtime features
- Docker, Kubernetes, cloud platforms
- Generated configurations
- Multi-environment support
- CI/CD integration
- Infrastructure as code

#### 4. Tests

**`test_phase2_adapters.py`** (221 lines)
- Tests HTTP runtime adapter with FastAPI backend generation
- Tests frontend runtime adapter with static site generation
- Tests frontend runtime adapter with React app generation
- Tests deploy runtime adapter with Docker configuration
- Tests IR metadata bridge (`_original_app` mechanism)
- ✅ All tests passing

## IR Metadata Bridge

To support gradual migration, Phase 2 added a temporary bridge mechanism:

### Implementation

Modified `namel3ss/ir/builder.py` to store original App AST:

```python
def build_backend_ir(app: App) -> BackendIR:
    # ... existing code ...
    return BackendIR(
        # ... all IR fields ...
        metadata={
            # ... existing metadata ...
            "_original_app": app,  # ← Phase 2 bridge
        },
    )
```

### Usage in Runtime Adapters

```python
def generate_fastapi_backend(ir: BackendIR, output_dir: str, **kwargs):
    # Get original App from IR metadata
    app = ir.metadata.get("_original_app")
    
    if app is None:
        raise ValueError("IR missing original app reference")
    
    # Use existing codegen (temporary)
    from namel3ss.codegen.backend import generate_backend
    generate_backend(app, output_dir, **kwargs)
```

### Why This Bridge Exists

- Existing codegen functions require the full `App` AST
- IR doesn't yet contain all information codegen needs
- Bridge allows Phase 2 to proceed without rewriting all codegen
- Will be removed in Phase 3 after codegen is migrated to consume IR directly

## Dependency Architecture

Phase 2 establishes strict dependency boundaries:

```
┌─────────────────────────────────────────────────────────┐
│                    namel3ss (core)                      │
│  - Parser                                               │
│  - AST types                                            │
│  - IR types (BackendIR, FrontendIR)                     │
│  - Type checker                                         │
│  - NO runtime dependencies (FastAPI, etc.)              │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ imports
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────┴────────────┐         ┌───────────┴────────────┐
│ namel3ss-runtime-  │         │  namel3ss-runtime-     │
│       http         │         │      frontend          │
│                    │         │                        │
│ - FastAPI adapter  │         │ - Static site adapter  │
│ - Depends on core  │         │ - React adapter        │
│ - Depends on       │         │ - Depends on core      │
│   FastAPI          │         │                        │
└────────────────────┘         └────────────────────────┘
                          
         ┌────────────────────────────────┐
         │  namel3ss-runtime-deploy       │
         │                                │
         │ - Docker adapter               │
         │ - Kubernetes adapter           │
         │ - Cloud adapters               │
         │ - Depends on core              │
         └────────────────────────────────┘
```

**Rules:**
- ✅ Runtimes CAN import from `namel3ss` core
- ❌ Core CANNOT import from runtimes
- ✅ Runtimes are independent of each other

## Test Results

```
======================================================================
PHASE 2: Runtime Adapter Tests
======================================================================

=== Testing HTTP Runtime Adapter ===
✓ Built BackendIR: TestApp
  - Endpoints: 1
  - Prompts: 1
✓ Generated FastAPI backend
  Files: 21 Python files

=== Testing Frontend Runtime Adapter (Static) ===
✓ Built FrontendIR: TestApp
  - Pages: 1
✓ Generated static site
  Files: 3 total files

=== Testing Frontend Runtime Adapter (React) ===
✓ Built FrontendIR: TestApp
✓ Generated React app
  Files: 18 total files

=== Testing Deploy Runtime Adapter ===
✓ Built BackendIR for deployment
✓ Generated Docker config
  Files: Dockerfile, docker-compose.yml, .dockerignore

=== Testing IR Metadata Bridge ===
✓ BackendIR metadata includes '_original_app'
✓ FrontendIR metadata includes '_original_app'
  Original app type: App

======================================================================
✅ ALL PHASE 2 ADAPTER TESTS PASSED
======================================================================
```

## API Examples

### HTTP Runtime

```python
from namel3ss import build_backend_ir
from namel3ss.parser import Parser
from namel3ss_runtime_http import generate_fastapi_backend

# Parse source
parser = Parser(source_code)
module = parser.parse()
app = module.body[0]

# Build IR
ir = build_backend_ir(app)

# Generate FastAPI backend
generate_fastapi_backend(ir, output_dir="backend/")
```

### Frontend Runtime

```python
from namel3ss import build_frontend_ir
from namel3ss_runtime_frontend import generate_static_site, generate_react_app

# Build IR
ir = build_frontend_ir(app)

# Generate static site
generate_static_site(ir, output_dir="frontend_static/")

# OR generate React app
generate_react_app(ir, output_dir="frontend_react/")
```

### Deploy Runtime

```python
from namel3ss import build_backend_ir
from namel3ss_runtime_deploy import generate_docker, generate_kubernetes

# Build IR
ir = build_backend_ir(app)

# Generate Docker configuration
generate_docker(ir, output_dir="deploy/docker/")

# Generate Kubernetes manifests
generate_kubernetes(
    ir,
    output_dir="deploy/k8s/",
    replicas=3,
    enable_hpa=True,
    domain="myapp.example.com",
)
```

## Achievements

✅ **Physical separation** - Runtime code isolated in separate packages  
✅ **Dependency boundaries** - Core has no runtime dependencies  
✅ **Independent packages** - Each runtime can be installed separately  
✅ **Backward compatibility** - Existing codegen still works via bridge  
✅ **Comprehensive docs** - Each runtime has detailed README  
✅ **Test coverage** - All adapters tested and passing  
✅ **Bridge mechanism** - Temporary `_original_app` enables gradual migration  

## Known Limitations (Phase 2)

### Temporary Bridge

- Runtime adapters still use existing codegen via `_original_app` bridge
- Codegen hasn't been moved into runtime packages yet
- **Resolution:** Phase 3 will move codegen code and remove bridge

### Import Paths

- Old import paths still work: `from namel3ss.codegen.backend import generate_backend`
- New paths exist but are wrappers: `from namel3ss_runtime_http import generate_fastapi_backend`
- **Resolution:** Phase 3 will update all imports and deprecate old paths

### CLI Integration

- CLI still uses old codegen paths directly
- CLI doesn't yet use runtime adapters
- **Resolution:** Phase 3 will refactor CLI to use runtime adapters

## Next Steps: Phase 3

### Phase 3 Goals

1. **Move Codegen Code**
   - Move `namel3ss/codegen/backend/` → `runtimes/http/namel3ss_runtime_http/codegen/`
   - Move `namel3ss/codegen/frontend/` → `runtimes/frontend/namel3ss_runtime_frontend/codegen/`
   - Keep only runtime-agnostic IR building in core

2. **Remove Bridge**
   - Update codegen to consume IR directly (not App AST)
   - Remove `_original_app` from IR metadata
   - Ensure codegen uses only IR types

3. **Update Imports**
   - Replace all `from namel3ss.codegen.backend` with runtime imports
   - Update CLI to use `namel3ss_runtime_http`
   - Deprecate old import paths

4. **Refactor CLI**
   - Create `namel3ss_cli` package
   - Separate language operations from runtime operations
   - CLI imports from runtime packages, not internal codegen

5. **Backward Compatibility**
   - Run full test suite after changes
   - Ensure generated code is identical
   - No breaking changes for users

## Files Modified in Phase 2

### Core
- `namel3ss/ir/builder.py` - Added `_original_app` to metadata (2 changes)

### Runtimes (New)
- `runtimes/README.md` - Runtime architecture docs
- `runtimes/http/README.md` - HTTP runtime docs
- `runtimes/http/pyproject.toml` - HTTP package config
- `runtimes/http/namel3ss_runtime_http/__init__.py` - HTTP public API
- `runtimes/http/namel3ss_runtime_http/adapter.py` - HTTP adapter implementation
- `runtimes/frontend/README.md` - Frontend runtime docs
- `runtimes/frontend/pyproject.toml` - Frontend package config
- `runtimes/frontend/namel3ss_runtime_frontend/__init__.py` - Frontend public API
- `runtimes/frontend/namel3ss_runtime_frontend/adapter.py` - Frontend adapter implementation
- `runtimes/deploy/README.md` - Deploy runtime docs
- `runtimes/deploy/pyproject.toml` - Deploy package config
- `runtimes/deploy/namel3ss_runtime_deploy/__init__.py` - Deploy public API
- `runtimes/deploy/namel3ss_runtime_deploy/adapter.py` - Deploy adapter implementation

### Tests
- `test_phase2_adapters.py` - Phase 2 validation tests

**Total:** 14 new files, 2 modified files

## Summary

Phase 2 successfully established the physical structure for runtime separation:

- ✅ **3 runtime packages** created with proper package structure
- ✅ **Strict dependency boundaries** enforced (runtimes depend on core, not reverse)
- ✅ **Comprehensive documentation** for each runtime (1000+ lines total)
- ✅ **Bridge mechanism** enables gradual migration without breaking changes
- ✅ **All tests passing** - HTTP, frontend, and deploy adapters working
- ✅ **Foundation laid** for Phase 3 codegen migration

**Phase 2 Status: Complete ✅**

Ready to proceed to Phase 3: Move codegen code and remove bridge.
