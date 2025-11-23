# Namel3ss CLI Reference

The Namel3ss CLI compiles `.n3` source into a runnable [Runtime](docs/reference/GLOSSARY.md#runtime), serves it for development, and executes evaluation hooks. This guide targets experienced engineers and uses terminology defined in the [Glossary](docs/reference/GLOSSARY.md).

## Installation
```bash
pip install namel3ss
pip install uvicorn[standard]  # required for dev server
```

## Command Summary
- `namel3ss build`: Compile `.n3` into frontend and backend runtime assets.
- `namel3ss run`: Compile and start a development server with reload.
- `namel3ss eval`: Execute experiment definitions.
- `namel3ss train`: Invoke training hooks bound to model definitions.
- `namel3ss deploy`: Invoke deployment hooks bound to model definitions.

## namel3ss build
Compile an [Application](docs/reference/GLOSSARY.md#application) into artifacts.

**Usage**
```bash
namel3ss build <file> [options]
```

**Key Options**
- `--out, -o <dir>`: Frontend output directory (default: `build`).
- `--backend-out <dir>`: Backend output directory (default: `backend_build`).
- `--build-backend`: Generate backend in addition to the frontend.
- `--backend-only`: Generate only the backend.
- `--target <static|react-vite>`: Select frontend generator (default: `static`).
- `--realtime`: Include WebSocket/SSE helpers.
- `--print-ast`: Emit parsed AST as JSON and exit.
- `--env KEY=VALUE`: Set environment variables during compilation (repeatable).

**Examples**
```bash
# Frontend only
namel3ss build app.n3
namel3ss build app.n3 --out dist

# Backend + frontend
namel3ss build app.n3 --build-backend --backend-out api

# Backend only
namel3ss build app.n3 --backend-only --backend-out runtime_api
```

> **Best Practice**  
> Use `--env KEY=VALUE` to provide secrets at build time. Do not hard-code provider credentials in `.n3` files.

## namel3ss run
Compile and start a development server for a single `.n3` file.

**Usage**
```bash
namel3ss run <file> [options]
```

**Key Options**
- `--backend-out <dir>`: Backend output directory (default: temp directory).
- `--host <host>` / `--port <port>`: Bind address (default: `127.0.0.1:8000`).
- `--no-reload`: Disable hot reload.
- `--dev`: Force dev mode even when targeting a Chain.
- `--json`: Emit structured JSON when executing a Chain.
- `--realtime`: Enable WebSocket/SSE helpers in the generated backend.
- `--embed-insights`: Inline insight payloads into dataset responses.
- `--env KEY=VALUE`: Set environment variables before launch.

**Lifecycle**
1. Parse `.n3`.
2. Generate a FastAPI backend scaffold.
3. Start uvicorn with reload unless disabled.

**Examples**
```bash
namel3ss run app.n3
namel3ss run app.n3 --port 3000 --host 0.0.0.0
namel3ss run app.n3 --backend-out dev_backend --no-reload
```

> **Warning**  
> `uvicorn` must be available (`pip install uvicorn[standard]`). The command fails fast if missing.

## namel3ss eval
Execute experiment variants declared in `.n3`.

**Usage**
```bash
namel3ss eval <experiment> [-f <file>] [--format <json|text>]
```

**Options**
- `-f, --file <path>`: Path to `.n3` (default: first `.n3` in the working directory).
- `--format <json|text>`: Output format (default: `json`).

On failure, the command returns structured JSON with `status="error"` and diagnostics.

## namel3ss train
Invoke training hooks bound to a model definition.

**Usage**
```bash
namel3ss train <file> --model <name>
```

If the model has no `trainer` hook, the command exits with structured JSON describing the missing configuration.

## namel3ss deploy
Invoke deployment hooks bound to a model definition.

**Usage**
```bash
namel3ss deploy <file> --model <name>
```

If deployment hooks are absent or fail, the command returns structured JSON diagnostics.

> **See Also:** [Style Guide](docs/STYLE_GUIDE.md), [Glossary](docs/reference/GLOSSARY.md)

### `namel3ss doctor`

Inspect optional dependencies and extras required for connectors, caches, and other integrations.

**Usage:**
```bash
namel3ss doctor
```

The command prints a human-readable summary highlighting missing packages and suggesting extras to install.

## Backward Compatibility

The CLI maintains backward compatibility with the original single-command interface:

```bash
# Old style (still works, with a deprecation notice)
namel3ss app.n3 --out dist

# New style (recommended)
namel3ss build app.n3 --out dist
```

## Error Handling

The CLI provides clear error messages for common issues:

### File Not Found
```
Error: file not found: nonexistent.n3
```

### Syntax Error
```
Syntax error on line 5: Expected ':' after if condition
if user.role == "admin"
```

### Missing uvicorn (for run command)
```
Error: uvicorn is not installed.
Please install it with: pip install uvicorn[standard]
```

## Workflow Examples

### Development Workflow

1. Create your `.n3` file:
```n3
app "My App" connects to postgres "DB".

page "Home" at "/":
  show text "Hello World"
```

2. Start the dev server:
```bash
namel3ss run myapp.n3
```

3. Make changes to your `.n3` file - the server will automatically reload

4. Visit http://127.0.0.1:8000 to see your API

### Production Build Workflow

1. Build static site and backend:
```bash
namel3ss build myapp.n3 --build-backend --out frontend --backend-out backend
```

2. Deploy the `frontend/` directory to a static host
3. Deploy the `backend/` directory to a Python hosting service

### Testing Workflow

Print the AST to understand how your code is parsed:
```bash
namel3ss build myapp.n3 --print-ast > ast.json
```

## Tips

1. **Use `run` during development** - it's faster and supports hot reload
2. **Use `build` for production** - generates optimized output
3. **Keep backends separate** - use `--backend-out` to organize your project
4. **Check syntax early** - use `--print-ast` to debug parsing issues
5. **Customize ports** - use `--port` to avoid conflicts with other services

## Advanced Usage

### Custom Backend Location for Development
```bash
namel3ss run app.n3 --backend-out ./dev_backend
```

## Environment Variables

Configure the generated backend by exporting one or more of the following variables before running `namel3ss run` or deploying the scaffolded FastAPI project:

| Variable | Description |
| --- | --- |
| `NAMEL3SS_API_KEY` | Require an API key via the `X-API-Key` header or bearer token. |
| `NAMEL3SS_AUTH_MODE` | `disabled`, `optional`, or `required` JWT enforcement. |
| `NAMEL3SS_JWT_SECRET` | Shared secret for HS256/384/512 token validation. |
| `NAMEL3SS_JWT_ALGORITHMS` | Comma-separated list of allowed HMAC algorithms. |
| `NAMEL3SS_JWT_AUDIENCE` / `NAMEL3SS_JWT_ISSUER` | Restrict valid `aud` / `iss` claims. |
| `NAMEL3SS_JWT_LEEWAY` | Leeway (seconds) for `exp` / `nbf` claim validation. |
| `NAMEL3SS_ENABLE_TENANT_RESOLUTION` | Resolve tenants from headers or JWT claims when truthy. |
| `NAMEL3SS_TENANT_HEADER` | Header name checked for tenant IDs (default `X-Tenant-Id`). |
| `NAMEL3SS_TENANT_CLAIM` | Claim names that may contain the tenant identifier. |
| `NAMEL3SS_REQUIRE_TENANT` | Reject requests that do not resolve a tenant. |
| `NAMEL3SS_ALLOW_ANONYMOUS` | Permit anonymous access even when JWTs are otherwise required. |
| `NAMEL3SS_ALLOW_STUBS` | Enable deterministic stubs for connectors and AI helpers. |
| `NAMEL3SS_RUNTIME_CACHE` | Select the runtime cache backend (`memory`, `redis`). |
| `NAMEL3SS_REDIS_URL` | Redis connection string when Redis caching is enabled. |
| `NAMEL3SS_RUNTIME_ASYNC` | Set to `1` to allow async dataset execution. |
| `NAMEL3SS_RUNTIME_CACHE_TTL` | Default TTL in seconds for dataset cache entries. |

This allows you to:
- Inspect the generated backend code
- Customize it manually if needed
- Version control the generated code

### Backend-Only Builds
```bash
namel3ss build app.n3 --backend-only --backend-out api
```

Useful when:
- You have a custom frontend
- You only need the API
- You're testing backend generation

### Multiple Apps
Run multiple apps on different ports:
```bash
namel3ss run app1.n3 --port 8000 &
namel3ss run app2.n3 --port 8001 &
```

## Help

Get help for any command:
```bash
namel3ss --help
namel3ss build --help
namel3ss run --help
```
