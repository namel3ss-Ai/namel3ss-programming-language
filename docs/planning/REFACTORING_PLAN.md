# Namel3ss Architecture Refactoring Plan

**Date:** November 24, 2025  
**Objective:** Transform Namel3ss from a monolithic language+runtime into a cleanly separated language core with multiple runtime implementations

---

## Executive Summary

This document outlines a comprehensive refactoring to establish Namel3ss as **a language that targets multiple runtimes**, not a single-platform SaaS. The refactoring will:

1. **Extract Language Core** - Parser, AST, typechecker, resolver, IR (no runtime dependencies)
2. **Create Runtime Packages** - HTTP/FastAPI adapter, frontend generators, deployment tooling
3. **Establish Clear Boundaries** - One-way dependencies, public APIs, enforceable contracts
4. **Maintain Compatibility** - All tests pass, existing workflows continue to work

---

## Part 1: Current State Analysis

### 1.1 Core Language Components (✓ Well-Structured)

These modules form the language core and are **mostly clean**:

```
namel3ss/
├── parser/                    # ✓ Grammar, tokenizer, parser logic
├── ast/                       # ✓ AST node definitions  
├── types/                     # ✓ Type system, type checker
├── resolver.py                # ✓ Symbol resolution
├── resolver_symbolic.py       # ✓ Logic programming resolver
├── errors.py                  # ✓ Error types
├── lang/                      # ✓ Core language features
├── linter/                    # ✓ Static analysis
├── formatting/                # ✓ Code formatter
└── testing/                   # ✓ Language-level test infrastructure
```

**Status:** These are language-level concerns and should remain in core.

### 1.2 Runtime-Specific Code (⚠️ Entangled with Core)

Code that currently lives alongside core but is **runtime-specific**:

```
namel3ss/
├── codegen/
│   ├── backend/               # ⚠️ Generates FastAPI servers directly
│   │   ├── core/
│   │   │   ├── app_module.py       # IMPORTS: from fastapi import FastAPI
│   │   │   ├── packages.py         # IMPORTS: from fastapi import APIRouter
│   │   │   ├── routers_pkg/        # IMPORTS: FastAPI throughout
│   │   │   ├── runtime/            # IMPORTS: FastAPI, HTTPException
│   │   │   └── ...
│   │   └── state/             # ⚠️ Encodes app to FastAPI-specific structure
│   └── frontend/              # ⚠️ Generates React/HTML/JS directly
├── cli/
│   ├── commands/
│   │   ├── run.py             # ⚠️ IMPORTS: uvicorn, starts servers
│   │   ├── deploy.py          # ⚠️ Docker, cloud deployment
│   │   └── ...
├── devserver.py               # ⚠️ Multi-app dev server orchestration
├── project_templates/         # ⚠️ CRUD service templates with FastAPI
└── sdk_sync/                  # ⚠️ FastAPI schema export
```

**Key Issues:**

1. **`namel3ss/codegen/backend/`** - Directly generates FastAPI code instead of an IR
   - `app_module.py` line 18: `from fastapi import FastAPI, Request`
   - `routers_pkg/*.py` - Every router imports FastAPI types
   - `runtime/header.py` - Injects `from fastapi import HTTPException`

2. **`namel3ss/cli/commands/run.py`** - Language CLI directly imports runtime tools
   - Line 75: `import uvicorn`
   - Line 137: `uvicorn.run("main:app", ...)`
   - Responsibility blur: build command both compiles **and** serves apps

3. **`namel3ss/sdk_sync/exporter.py`** - Exports FastAPI schemas specifically
   - Line 425: `from fastapi import APIRouter, HTTPException`
   - Could export generic OpenAPI/IR instead

### 1.3 Optional Integrations (✓ Loosely Coupled)

Already well-separated:

```
demo-vscode-extension/         # ✓ VS Code extension (separate)
vscode-extension/              # ✓ VS Code LSP server
editor/vscode/                 # ✓ Editor support
yjs-server/                    # ✓ Collaborative editing server
frontend_live/                 # ✓ Web-based editor/IDE
```

**Status:** Keep these as optional tools/integrations.

### 1.4 Test Structure (✓ Comprehensive)

```
tests/
├── parser/                    # ✓ Language core tests
├── codegen/                   # ✓ Codegen tests (need reclassification)
├── backend/                   # ⚠️ Runtime integration tests
├── integration/               # ⚠️ End-to-end tests (runtime-dependent)
├── e2e/                       # ⚠️ Full stack tests
└── ...
```

**Need:** Reclassify into `tests/core/` vs `tests/runtimes/` structure.

---

## Part 2: Blurred Boundaries Analysis

### 2.1 Dependency Violations

**Problem:** Core language code imports runtime frameworks.

| Module | Violation | Impact |
|--------|-----------|---------|
| `codegen/backend/core/app_module.py` | `from fastapi import FastAPI` | Core codegen locked to FastAPI |
| `codegen/backend/core/routers_pkg/*.py` | FastAPI imports throughout | Cannot target other runtimes |
| `cli/commands/run.py` | `import uvicorn` | Language CLI depends on runtime server |
| `sdk_sync/exporter.py` | `from fastapi import APIRouter` | Schema export FastAPI-specific |
| `project_templates/crud_service/` | FastAPI templates baked in | Templates are runtime-specific |

**Consequence:** Cannot build alternative runtimes (e.g., AWS Lambda adapter, gRPC service, CLI tool runner) without rewriting core codegen.

### 2.2 Codegen Architecture Problem

**Current (Tightly Coupled):**

```
.ai source → Parser → AST → BackendState → FastAPI .py files
                                              ↳ Directly generates:
                                                 from fastapi import FastAPI
                                                 app = FastAPI()
                                                 @app.get("/api/...")
```

**What We Need (Decoupled via IR):**

```
.ai source → Parser → AST → Resolver → BackendState (IR)
                                            ↓
                                       [Runtime Adapters]
                                            ↓
                        ┌──────────────────┼──────────────────┐
                        ↓                  ↓                  ↓
                  FastAPI Runtime    Lambda Runtime    gRPC Runtime
                  (HTTP/JSON)        (Serverless)      (Protocol Buffers)
```

**IR Structure Needed:**

```python
# namel3ss/ir/backend.py (NEW)
@dataclass
class EndpointIR:
    """Runtime-agnostic endpoint specification"""
    path: str
    method: str  # GET, POST, etc.
    input_schema: TypeSpec
    output_schema: TypeSpec
    handler: CallableRef  # Reference to prompt/agent/tool

@dataclass
class BackendIR:
    """Complete backend specification (IR)"""
    endpoints: List[EndpointIR]
    agents: List[AgentSpec]
    prompts: List[PromptSpec]
    datasets: List[DatasetSpec]
    memory: List[MemorySpec]
    # No FastAPI, no HTTP - just specifications
```

### 2.3 CLI Responsibility Blur

**Problem:** `namel3ss build` and `namel3ss run` mix language concerns with runtime concerns.

**Current `namel3ss run`:**
```python
# namel3ss/cli/commands/run.py
def cmd_run(args):
    app = parse_and_typecheck(source)        # ✓ Language concern
    generate_backend(app, backend_dir)        # ✓ Language concern (should emit IR)
    import uvicorn                            # ✗ Runtime concern
    uvicorn.run("main:app", host=..., port=...)  # ✗ Runtime concern
```

**What We Need:**
```python
# Core language CLI (namel3ss-core)
def cmd_build(args):
    app = parse_and_typecheck(source)
    ir = compile_to_ir(app)
    write_ir(ir, output_dir)  # Emit IR artifacts
    
# Runtime CLI (namel3ss-runtime-http)
def cmd_serve(args):
    ir = load_ir(input_dir)
    fastapi_app = adapt_ir_to_fastapi(ir)
    uvicorn.run(fastapi_app, host=..., port=...)
```

---

## Part 3: Target Architecture

### 3.1 Package Structure

```
namel3ss-programming-language/
│
├── namel3ss/                          # LANGUAGE CORE PACKAGE
│   ├── __init__.py
│   ├── parser/                        # ✓ Keep
│   ├── ast/                           # ✓ Keep  
│   ├── types/                         # ✓ Keep
│   ├── resolver.py                    # ✓ Keep
│   ├── errors.py                      # ✓ Keep
│   ├── lang/                          # ✓ Keep
│   ├── linter/                        # ✓ Keep
│   ├── formatting/                    # ✓ Keep
│   ├── testing/                       # ✓ Keep (mock LLM, test DSL)
│   ├── ir/                            # ✓ NEW - Intermediate Representation
│   │   ├── __init__.py
│   │   ├── spec.py                    # IR dataclasses
│   │   ├── builder.py                 # AST → IR transformation
│   │   └── serialization.py           # IR ↔ JSON
│   ├── compiler/                      # ✓ NEW - Compilation pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py                # Parse → Resolve → Type Check → IR
│   │   └── validation.py              # IR validation
│   └── cli/                           # ✓ Keep (language-level commands only)
│       ├── __init__.py
│       ├── commands/
│       │   ├── build.py               # ✓ Modified - emits IR, no server startup
│       │   ├── lint.py                # ✓ Keep
│       │   ├── fmt.py                 # ✓ Keep
│       │   ├── test.py                # ✓ Keep (language-level tests)
│       │   ├── typecheck.py           # ✓ Keep
│       │   └── trace.py               # ✓ Keep (language tracing)
│       └── ...
│
├── runtimes/                          # RUNTIME IMPLEMENTATIONS (NEW)
│   ├── __init__.py
│   ├── http/                          # HTTP/REST runtime
│   │   ├── __init__.py
│   │   ├── adapter.py                 # IR → HTTP server
│   │   ├── fastapi_adapter.py         # IR → FastAPI app
│   │   ├── codegen/                   # FastAPI-specific code generation
│   │   │   ├── app_module.py          # ← MOVE from namel3ss/codegen/backend
│   │   │   ├── routers.py
│   │   │   └── ...
│   │   ├── cli.py                     # `serve`, `dev` commands
│   │   └── templates/                 # FastAPI project templates
│   ├── frontend/                      # Frontend generation runtime
│   │   ├── __init__.py
│   │   ├── adapter.py                 # IR → Frontend artifacts
│   │   ├── generators/
│   │   │   ├── static.py              # ← MOVE from namel3ss/codegen/frontend
│   │   │   ├── react.py
│   │   │   └── ...
│   │   └── templates/
│   └── deploy/                        # Deployment runtime
│       ├── __init__.py
│       ├── docker.py                  # Docker image generation
│       ├── k8s.py                     # Kubernetes manifests
│       └── cloud/                     # Cloud provider adapters
│           ├── aws.py
│           ├── gcp.py
│           └── azure.py
│
├── tools/                             # OPTIONAL INTEGRATIONS
│   ├── vscode_extension/              # ← MOVE from demo-vscode-extension/
│   ├── web_editor/                    # ← MOVE from frontend_live/
│   ├── yjs_server/                    # ← MOVE from yjs-server/
│   └── lsp_server/                    # LSP implementation
│
├── tests/                             # TESTS (REORGANIZED)
│   ├── core/                          # Language core tests
│   │   ├── parser/                    # ← MOVE from tests/parser/
│   │   ├── typechecker/               # ← MOVE from tests/types/
│   │   ├── resolver/
│   │   ├── linter/
│   │   ├── formatting/
│   │   └── ir/                        # IR generation tests
│   ├── runtimes/                      # Runtime implementation tests
│   │   ├── http/                      # ← MOVE from tests/backend/
│   │   ├── frontend/                  # ← MOVE from tests/codegen/
│   │   └── integration/               # ← MOVE from tests/e2e/
│   └── tools/                         # Optional tool tests
│       ├── vscode/
│       └── web_editor/
│
├── docs/                              # DOCUMENTATION (UPDATED)
│   ├── ARCHITECTURE.md                # ← NEW - High-level architecture
│   ├── LANGUAGE_REFERENCE.md          # ✓ Keep
│   ├── IR_SPECIFICATION.md            # ← NEW - IR format specification
│   ├── RUNTIME_GUIDE.md               # ← NEW - How to build runtimes
│   ├── core/                          # ← NEW - Core language docs
│   └── runtimes/                      # ← NEW - Runtime-specific docs
│
├── pyproject.toml                     # MODIFIED - Core dependencies only
├── runtimes/http/pyproject.toml       # ← NEW - FastAPI, uvicorn
├── runtimes/frontend/pyproject.toml   # ← NEW - Frontend deps
└── README.md                          # UPDATED - Reflects new architecture
```

### 3.2 Dependency Rules

**Strict One-Way Dependencies:**

```
Core Language ← NO dependencies on Runtimes or Tools
    ↑
    │ (depends on)
    │
Runtimes ← NO dependencies on Tools
    ↑
    │ (optionally depends on)
    │
Tools
```

**Package Dependencies:**

| Package | Can Import | Cannot Import |
|---------|------------|---------------|
| `namel3ss` (core) | stdlib only | `fastapi`, `uvicorn`, `react`, runtime frameworks |
| `runtimes/http` | `namel3ss` core, `fastapi`, `uvicorn` | `runtimes/frontend`, `tools/*` |
| `runtimes/frontend` | `namel3ss` core, `jinja2` | `runtimes/http`, `tools/*` |
| `tools/vscode` | `namel3ss` core, `pygls` | `runtimes/*` (except for preview) |

### 3.3 Public Core API

**`namel3ss/__init__.py`** - Core public API:

```python
"""
Namel3ss Language Core

This package provides the core language implementation:
- Parser, AST, type checker, resolver
- Intermediate representation (IR) generation
- Language-level tooling (linter, formatter, test runner)

To host Namel3ss programs, use a runtime package:
- namel3ss.runtimes.http - FastAPI HTTP server
- namel3ss.runtimes.frontend - Frontend generation
"""

# Core language features
from .parser import Parser, N3SyntaxError
from .ast import App, Module, Page, Prompt, Agent, Dataset, Frame
from .types import TypeChecker, TypeError
from .resolver import Resolver, ResolverError
from .errors import N3Error, N3CompilationError

# Intermediate Representation
from .ir import BackendIR, FrontendIR, compile_to_ir

# Compilation pipeline
from .compiler import compile_program, CompilationResult

__version__ = "0.5.0"
__all__ = [
    "Parser",
    "N3SyntaxError",
    "TypeChecker",
    "TypeError",
    "Resolver",
    "ResolverError",
    "BackendIR",
    "FrontendIR",
    "compile_to_ir",
    "compile_program",
    "CompilationResult",
]
```

**Runtime Adapters Import Core:**

```python
# runtimes/http/adapter.py
from namel3ss import BackendIR, compile_to_ir
from namel3ss.ast import App

def adapt_to_fastapi(ir: BackendIR) -> FastAPI:
    """Convert language IR to FastAPI application"""
    from fastapi import FastAPI
    app = FastAPI()
    # ... generate routes from IR ...
    return app
```

---

## Part 4: Implementation Phases

### Phase 1: Create IR Layer (No Behavioral Change)

**Goal:** Introduce IR as intermediate step, but keep existing codegen working.

**Steps:**

1. Create `namel3ss/ir/` package:
   ```python
   # namel3ss/ir/spec.py
   @dataclass
   class EndpointIR:
       path: str
       method: str
       input_schema: TypeSpec
       output_schema: TypeSpec
       handler_ref: str  # e.g., "prompts.MyPrompt"
   
   @dataclass
   class BackendIR:
       endpoints: List[EndpointIR]
       agents: List[AgentSpec]
       prompts: List[PromptSpec]
       # ... no FastAPI-specific types
   ```

2. Create `namel3ss/ir/builder.py`:
   ```python
   def build_ir(app: App) -> BackendIR:
       """Convert AST to runtime-agnostic IR"""
       # Current BackendState logic, minus FastAPI specifics
   ```

3. Refactor `namel3ss/codegen/backend/core/generator.py`:
   ```python
   def generate_backend(app: App, out_dir: Path, ...) -> None:
       # Step 1: Build IR (new)
       ir = build_ir(app)
       
       # Step 2: Adapt IR to FastAPI (existing logic)
       generate_fastapi_from_ir(ir, out_dir, ...)
   ```

**Testing:** All existing tests pass. No user-visible change.

---

### Phase 2: Move Runtime Code to `runtimes/`

**Goal:** Physical separation of core and runtime code.

**Steps:**

1. Create `runtimes/http/` package:
   ```bash
   mkdir -p runtimes/http/namel3ss_runtime_http
   ```

2. Move FastAPI-specific codegen:
   ```bash
   # Move backend generation
   mv namel3ss/codegen/backend/core/app_module.py \
      runtimes/http/namel3ss_runtime_http/codegen/
   mv namel3ss/codegen/backend/core/routers_pkg/ \
      runtimes/http/namel3ss_runtime_http/codegen/
   # ... etc
   ```

3. Update imports:
   ```python
   # Before (in core):
   from namel3ss.codegen.backend.core.app_module import _render_app_module
   
   # After (in runtime):
   from namel3ss_runtime_http.codegen.app_module import render_app_module
   ```

4. Create `runtimes/http/pyproject.toml`:
   ```toml
   [project]
   name = "namel3ss-runtime-http"
   version = "0.5.0"
   dependencies = [
       "namel3ss>=0.5.0",  # Core language
       "fastapi>=0.110,<1.0",
       "uvicorn>=0.30,<0.31",
   ]
   ```

**Testing:** Update import paths in tests. All tests pass.

---

### Phase 3: Refactor CLI (Language vs Runtime Commands)

**Goal:** Core CLI has only language commands. Runtime commands move to runtime packages.

**Core CLI Commands (Keep in `namel3ss/cli/`):**

```python
# namel3ss/cli/commands/build.py (MODIFIED)
def cmd_build(args):
    """Build .ai source to IR artifacts"""
    app = load_and_compile(args.file)
    ir = compile_to_ir(app)
    write_ir(ir, args.out / "ir.json")
    print(f"✓ IR artifacts written to {args.out}/ir.json")
    
    # Optional: If runtime package installed, offer to generate
    if has_runtime("http"):
        if args.with_http:
            generate_http_runtime(ir, args.out / "backend")
```

**Runtime CLI Commands (Move to `runtimes/http/cli.py`):**

```python
# runtimes/http/namel3ss_runtime_http/cli.py (NEW)
def cmd_serve(args):
    """Start HTTP development server"""
    ir = load_ir(args.ir_file)
    app = adapt_to_fastapi(ir)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
```

**Unified CLI Entry Point (Optional):**

```python
# namel3ss/cli/__init__.py
def main():
    # Register core commands
    parser.add_subcommand("build", cmd_build)
    parser.add_subcommand("lint", cmd_lint)
    parser.add_subcommand("test", cmd_test)
    
    # Discover and register runtime commands (if installed)
    for runtime in discover_runtime_plugins():
        runtime.register_commands(parser)
    
    args = parser.parse_args()
    args.func(args)
```

**Testing:** CLI tests updated. Existing workflows work via new command structure.

---

### Phase 4: Reorganize Tests

**Goal:** Tests reflect core vs runtime distinction.

**Steps:**

1. Create new test structure:
   ```bash
   mkdir -p tests/core tests/runtimes/http tests/runtimes/frontend
   ```

2. Move core tests:
   ```bash
   mv tests/parser tests/core/
   mv tests/types tests/core/typechecker
   mv tests/linter tests/core/
   # ... etc
   ```

3. Move runtime tests:
   ```bash
   mv tests/backend tests/runtimes/http/
   mv tests/integration tests/runtimes/http/integration
   mv tests/e2e tests/runtimes/integration/e2e
   ```

4. Update test fixtures:
   - Core tests use mock LLMs only
   - Runtime tests can use real HTTP servers, databases

**Testing:** Run full test suite. All tests pass in new locations.

---

### Phase 5: Add Dependency Enforcement

**Goal:** Prevent core from importing runtime code.

**Implementation:**

1. Create `scripts/check_dependencies.py`:
   ```python
   """Enforce core → runtime dependency rules"""
   
   CORE_PACKAGES = ["namel3ss"]
   RUNTIME_PACKAGES = ["runtimes"]
   FORBIDDEN_CORE_IMPORTS = ["fastapi", "uvicorn", "flask", "django"]
   
   def check_core_imports():
       for py_file in core_files():
           imports = extract_imports(py_file)
           for imp in imports:
               if imp in FORBIDDEN_CORE_IMPORTS:
                   raise DependencyViolation(
                       f"{py_file} imports runtime dependency {imp}"
                   )
   ```

2. Add to CI pipeline:
   ```yaml
   # .github/workflows/test.yml
   - name: Check dependency rules
     run: python scripts/check_dependencies.py
   ```

3. Add pre-commit hook:
   ```yaml
   # .pre-commit-config.yaml
   - repo: local
     hooks:
       - id: check-dependencies
         name: Check dependency rules
         entry: python scripts/check_dependencies.py
         language: system
   ```

**Testing:** Intentionally add a `from fastapi import FastAPI` to core code. Verify check fails.

---

### Phase 6: Update Documentation

**Goal:** Documentation reflects new architecture.

**New Documentation:**

1. **`docs/ARCHITECTURE.md`** (NEW):
   ```markdown
   # Namel3ss Architecture
   
   Namel3ss is an AI programming language with a **clean separation** between:
   
   1. **Language Core** - Parser, type checker, IR generation
   2. **Runtime Implementations** - HTTP servers, frontend generators, deployment adapters
   3. **Optional Tools** - VS Code extension, web editor, LSP server
   
   ## Core → Runtime Dataflow
   
   ```
   .ai source → Parser → AST → Resolver → Type Checker → IR
                                                          ↓
                                                    [Runtime Adapters]
                                                          ↓
                                       ┌──────────────────┼──────────────────┐
                                       ↓                  ↓                  ↓
                                 FastAPI HTTP         React Frontend    Lambda Serverless
   ```
   
   ## Adding a New Runtime
   
   To create a new runtime (e.g., gRPC, GraphQL, serverless):
   
   1. Create runtime package: `runtimes/grpc/`
   2. Import core: `from namel3ss import BackendIR`
   3. Implement adapter: `def adapt_to_grpc(ir: BackendIR) -> grpc.Server`
   4. Register CLI commands (optional)
   ```

2. **`docs/IR_SPECIFICATION.md`** (NEW):
   ```markdown
   # Intermediate Representation (IR) Specification
   
   The Namel3ss IR is a runtime-agnostic representation of compiled programs.
   
   ## IR Structure
   
   ### EndpointIR
   
   Represents an API endpoint:
   
   ```python
   @dataclass
   class EndpointIR:
       path: str              # e.g., "/api/classify"
       method: str            # "GET", "POST", etc.
       input_schema: TypeSpec  # Input type specification
       output_schema: TypeSpec # Output type specification
       handler_ref: str       # Reference to prompt/agent/tool
   ```
   
   ### AgentSpec, PromptSpec, etc.
   
   ... (document all IR types)
   ```

3. **`docs/RUNTIME_GUIDE.md`** (NEW):
   ```markdown
   # Building a Namel3ss Runtime
   
   This guide shows how to create a custom runtime for Namel3ss.
   
   ## Step 1: Create Runtime Package
   
   ```bash
   mkdir -p runtimes/my_runtime/namel3ss_runtime_my_runtime
   ```
   
   ## Step 2: Implement IR Adapter
   
   ```python
   # runtimes/my_runtime/namel3ss_runtime_my_runtime/adapter.py
   from namel3ss import BackendIR, compile_to_ir
   
   def adapt_to_my_runtime(ir: BackendIR) -> MyRuntimeApp:
       app = MyRuntimeApp()
       for endpoint in ir.endpoints:
           app.add_route(
               path=endpoint.path,
               method=endpoint.method,
               handler=create_handler(endpoint)
           )
       return app
   ```
   
   ## Step 3: Register CLI Commands (Optional)
   
   ... (guide for CLI integration)
   ```

4. Update `README.md`:
   ```markdown
   # Namel3ss - The AI Programming Language
   
   **Namel3ss is a language that can target many runtimes, not a SaaS you're locked into.**
   
   ## Architecture
   
   - **Language Core** (`namel3ss`) - Parser, type checker, IR generation
   - **Runtimes** - Adapters to host Namel3ss programs:
     - `namel3ss-runtime-http` - FastAPI HTTP server
     - `namel3ss-runtime-frontend` - React/static site generator
     - `namel3ss-runtime-deploy` - Docker, K8s, cloud deployment
   - **Tools** - Optional editor integrations (VS Code, web editor)
   
   ## Installation
   
   ```bash
   # Install language core
   pip install namel3ss
   
   # Install HTTP runtime (optional)
   pip install namel3ss-runtime-http
   
   # Install frontend runtime (optional)
   pip install namel3ss-runtime-frontend
   ```
   
   ## Usage
   
   ```bash
   # Compile to IR
   namel3ss build app.ai --out build/
   
   # Run HTTP server (requires namel3ss-runtime-http)
   namel3ss serve build/ir.json --host 127.0.0.1 --port 8000
   ```
   ```

**Testing:** Documentation builds without errors. Links work.

---

### Phase 7: Final Validation

**Goal:** Ensure all goals met, tests pass, documentation accurate.

**Checklist:**

- [ ] Core package has NO imports of `fastapi`, `uvicorn`, `flask`, etc.
- [ ] Runtime packages depend on core, not vice versa
- [ ] All existing CLI commands work (possibly with new names/structure)
- [ ] All tests pass in new locations (`tests/core/`, `tests/runtimes/`)
- [ ] Dependency enforcement CI check passes
- [ ] Documentation reflects new architecture
- [ ] `README.md` clearly states: "Namel3ss is a language, not a platform"

---

## Part 5: Migration Guide for Users

### For Existing Users

**No breaking changes for basic workflows:**

```bash
# These still work (backward compatible):
namel3ss build app.ai
namel3ss run app.ai
namel3ss test app.ai
```

**New recommended workflows:**

```bash
# Compile to IR (core language operation)
namel3ss build app.ai --out build/

# Start HTTP server (runtime operation)
namel3ss serve build/ir.json

# Or use runtime-specific CLI:
namel3ss-http serve build/ir.json --dev
```

### For Plugin/Extension Developers

**Before:**
```python
from namel3ss.codegen.backend import generate_backend

def my_plugin(app):
    generate_backend(app, "my_output/")
```

**After:**
```python
from namel3ss import compile_to_ir
from namel3ss_runtime_http import generate_http_backend

def my_plugin(app):
    ir = compile_to_ir(app)
    generate_http_backend(ir, "my_output/")
```

### For Runtime Developers

**New capability:** Create custom runtimes for Namel3ss!

```python
# my_custom_runtime/adapter.py
from namel3ss import BackendIR

def adapt_to_lambda(ir: BackendIR) -> dict:
    """Convert Namel3ss IR to AWS Lambda handler"""
    return {
        "handler": generate_lambda_handler(ir),
        "config": generate_sam_template(ir),
    }
```

---

## Part 6: Success Metrics

### Technical Metrics

- ✅ **Zero circular dependencies** between core ↔ runtimes
- ✅ **100% test pass rate** in new structure
- ✅ **Dependency check CI** passes
- ✅ **Core package size** < 50% of current (runtime code moved out)
- ✅ **IR specification** documented and validated

### Architectural Metrics

- ✅ **Core package** imports no runtime frameworks (fastapi, uvicorn, etc.)
- ✅ **Multiple runtimes** possible (HTTP, frontend, deploy)
- ✅ **Plugin architecture** for runtime discovery
- ✅ **Public API** clearly defined in `namel3ss/__init__.py`

### User Experience Metrics

- ✅ **Backward compatibility** - existing commands work
- ✅ **Clear documentation** - architecture guide, runtime guide, IR spec
- ✅ **Developer narrative** - "Namel3ss is a language, not a platform"
- ✅ **Migration guide** - smooth transition for existing users

---

## Part 7: Risk Mitigation

### Risk 1: Breaking Existing Code

**Mitigation:**
- Maintain backward-compatible imports via `__init__.py` wrappers
- Keep deprecated paths working with deprecation warnings
- Extensive testing before release

### Risk 2: Performance Regression

**Mitigation:**
- IR layer is simple dataclasses, minimal overhead
- Benchmark before/after
- Optimize IR serialization if needed

### Risk 3: Incomplete Refactor

**Mitigation:**
- Phase-by-phase approach with validation checkpoints
- Each phase maintains working tests
- Can stop at any phase with partial benefits

### Risk 4: Community Confusion

**Mitigation:**
- Clear communication in CHANGELOG
- Updated documentation at each phase
- Migration guide with examples
- Blog post explaining rationale and benefits

---

## Appendix A: File Move Checklist

### Core Language (Stay in `namel3ss/`)

- [x] `namel3ss/parser/`
- [x] `namel3ss/ast/`
- [x] `namel3ss/types/`
- [x] `namel3ss/resolver.py`
- [x] `namel3ss/errors.py`
- [x] `namel3ss/lang/`
- [x] `namel3ss/linter/`
- [x] `namel3ss/formatting/`
- [x] `namel3ss/testing/` (mock infrastructure)

### Move to `runtimes/http/`

- [ ] `namel3ss/codegen/backend/` → `runtimes/http/namel3ss_runtime_http/codegen/`
- [ ] `namel3ss/cli/commands/run.py` → `runtimes/http/namel3ss_runtime_http/cli.py`
- [ ] `namel3ss/devserver.py` → `runtimes/http/namel3ss_runtime_http/devserver.py`
- [ ] `namel3ss/project_templates/crud_service/` → `runtimes/http/templates/`

### Move to `runtimes/frontend/`

- [ ] `namel3ss/codegen/frontend/` → `runtimes/frontend/namel3ss_runtime_frontend/generators/`
- [ ] Frontend templates

### Move to `runtimes/deploy/`

- [ ] `namel3ss/cli/commands/deploy.py` → `runtimes/deploy/cli.py`
- [ ] Docker, K8s generation logic

### Move to `tools/`

- [ ] `demo-vscode-extension/` → `tools/vscode_extension/`
- [ ] `frontend_live/` → `tools/web_editor/`
- [ ] `yjs-server/` → `tools/yjs_server/`

---

## Appendix B: Import Mapping Table

| Old Import | New Import | Notes |
|------------|------------|-------|
| `from namel3ss.codegen.backend import generate_backend` | `from namel3ss_runtime_http import generate_http_backend` | Runtime-specific |
| `from namel3ss.codegen.backend.state import BackendState` | `from namel3ss.ir import BackendIR` | Renamed for clarity |
| `from namel3ss.cli.commands.run import run_dev_server` | `from namel3ss_runtime_http.cli import serve` | Moved to runtime |
| `from namel3ss.parser import Parser` | `from namel3ss import Parser` | Core API |
| `from namel3ss.ast import App` | `from namel3ss import App` | Core API |

---

## Conclusion

This refactoring transforms Namel3ss from a monolithic "language+runtime" into a **modular, extensible ecosystem** with:

1. **Clean Language Core** - Reusable by any runtime
2. **Multiple Runtime Implementations** - HTTP, frontend, serverless, etc.
3. **Clear Boundaries** - Enforced dependencies, public APIs
4. **Maintainable Architecture** - Production-grade separation of concerns

**Result:** Namel3ss becomes **a language that can target many runtimes**, not a platform you're locked into.

---

**Next Steps:** Review this plan, provide feedback, then proceed with Phase 1 implementation.
