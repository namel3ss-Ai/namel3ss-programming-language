# Namel3ss CLI Documentation

## Overview

The Namel3ss CLI provides a complete toolset for building and running full-stack applications from `.n3` source files.

## Installation

```bash
pip install namel3ss
```

For development server support, also install uvicorn:
```bash
pip install uvicorn[standard]
```

## Commands

### `namel3ss build`

Generate static site and/or backend scaffold from a `.n3` file.

**Usage:**
```bash
namel3ss build <file> [options]
```

**Options:**
- `--out, -o <dir>` - Output directory for static files (default: `build`)
- `--print-ast` - Print the parsed AST as JSON and exit
- `--build-backend` - Also generate FastAPI backend scaffold
- `--backend-only` - Only generate backend, skip static site
- `--backend-out <dir>` - Output directory for backend scaffold (default: `backend_build`)

**Examples:**

Generate static site only:
```bash
namel3ss build app.n3
namel3ss build app.n3 --out dist
```

Generate both static site and backend:
```bash
namel3ss build app.n3 --build-backend
namel3ss build app.n3 --build-backend --backend-out api
```

Generate backend only:
```bash
namel3ss build app.n3 --backend-only --backend-out backend
```

Print AST for debugging:
```bash
namel3ss build app.n3 --print-ast
```

### `namel3ss run`

Start a development server with hot reload for rapid iteration.

**Usage:**
```bash
namel3ss run <file> [options]
```

**Options:**
- `--backend-out <dir>` - Output directory for backend scaffold (default: temp directory)
- `--host <host>` - Host to bind server to (default: `127.0.0.1`)
- `--port <port>` - Port to bind server to (default: `8000`)
- `--no-reload` - Disable hot reload

**What it does:**
1. Parses your `.n3` file
2. Generates a FastAPI backend scaffold
3. Starts a uvicorn development server with hot reload
4. Displays the server URL

**Examples:**

Basic dev server (uses temp directory):
```bash
namel3ss run app.n3
```

Custom port and host:
```bash
namel3ss run app.n3 --port 3000 --host 0.0.0.0
```

Specify backend directory:
```bash
namel3ss run app.n3 --backend-out dev_backend
```

Disable hot reload:
```bash
namel3ss run app.n3 --no-reload
```

**Requirements:**
- `uvicorn` must be installed: `pip install uvicorn[standard]`
- If uvicorn is not installed, you'll get a helpful error message

**Output:**
```
✓ Parsed: My App
✓ Backend generated in: /tmp/.namel3ss_dev_backend

Namel3ss dev server running at http://127.0.0.1:8000
Press CTRL+C to stop
```

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
