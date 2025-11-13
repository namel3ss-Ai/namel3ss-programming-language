# Namel3ss CLI Quick Reference

## Commands

### Build
```bash
# Basic build (static site)
namel3ss build app.n3

# With backend
namel3ss build app.n3 --build-backend

# Backend only
namel3ss build app.n3 --backend-only

# Custom directories
namel3ss build app.n3 --out frontend --backend-out api

# React frontend target
namel3ss build app.n3 --target=react-vite --out app-react
```

### Run (Development Server)
```bash
# Start dev server
namel3ss run app.n3

# Custom port
namel3ss run app.n3 --port 3000

# Custom backend directory
namel3ss run app.n3 --backend-out ./dev_backend

# Public access
namel3ss run app.n3 --host 0.0.0.0 --port 8000
```

## Common Options

### Build Command
- `--out <dir>` - Static output directory
- `--build-backend` - Generate backend too
- `--backend-only` - Skip static site
- `--backend-out <dir>` - Backend directory
- `--target <static|react-vite>` - Select frontend generator
- `--print-ast` - Debug: print AST

### Run Command
- `--backend-out <dir>` - Backend directory (default: temp)
- `--host <host>` - Server host (default: 127.0.0.1)
- `--port <port>` - Server port (default: 8000)
- `--no-reload` - Disable hot reload

## Workflows

### Development
```bash
# 1. Create app.n3
# 2. Start dev server
namel3ss run app.n3

# 3. Edit app.n3 (auto-reloads)
# 4. Visit http://127.0.0.1:8000
```

### Production
```bash
# Generate everything
namel3ss build app.n3 --build-backend --out dist --backend-out api

# Deploy dist/ (static)
# Deploy api/ (backend)
```

### Testing
```bash
# Check AST
namel3ss build app.n3 --print-ast

# Quick backend test
namel3ss run app.n3 --no-reload
```

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| File not found | Missing .n3 file | Check file path |
| Syntax error | Invalid N3 syntax | Fix syntax (error shows line) |
| uvicorn not installed | Missing dependency | `pip install uvicorn[standard]` |

## Tips

✅ Use `run` during development (hot reload)
✅ Use `build` for production (optimized)
✅ `--backend-only` for API-only projects
✅ `--print-ast` for debugging syntax
✅ Legacy mode still works but deprecated

## Examples

```bash
# Multi-app development
namel3ss run app1.n3 --port 8000 &
namel3ss run app2.n3 --port 8001 &

# Frontend + Backend split
namel3ss build app.n3 --build-backend --out www --backend-out server

# Inspect generated code
namel3ss build app.n3 --backend-only --backend-out inspect_me
```

## Help
```bash
namel3ss --help
namel3ss build --help
namel3ss run --help
```
