# Namel3ss CLI Quick Reference# Namel3ss CLI Quick Reference



## Commands## Commands



### Build### Build

```bash```bash

# Basic build (static site)# Basic build (static site)

namel3ss build app.n3namel3ss build app.n3



# With backend# With backend

namel3ss build app.n3 --build-backend `--realtime` - Include realtime websocket scaffolding in the backend

 `--embed-insights` - Inline insight results into dataset payloads

# Backend only `--env KEY=VALUE` - Set env vars for generation (repeatable)

namel3ss build app.n3 --backend-onlynamel3ss build app.n3 --out frontend --backend-out api



# Custom directories `--dev` - Force dev-server mode when invoking with a chain name

namel3ss build app.n3 --out frontend --backend-out api `--json` - Emit structured JSON when executing a chain

 `--realtime` - Enable realtime websocket scaffolding in the generated backend

# React frontend target `--embed-insights` - Inline insight results into dataset payloads

namel3ss build app.n3 --target=react-vite --out app-react `--env KEY=VALUE` - Set env vars before launching the server (repeatable)

```### Run (Development Server)

```bash

### Run (Development Server)# Start dev server

```bashnamel3ss run app.n3

# Start dev server

namel3ss run app.n3# Custom port

namel3ss run app.n3 --port 3000

# Custom port

namel3ss run app.n3 --port 3000# Custom backend directory



# Custom backend directory# Inspect experiment behaviour

namel3ss run app.n3 --backend-out ./dev_backendnamel3ss eval my_experiment -f app.n3 --format=text

namel3ss run app.n3 --backend-out ./dev_backend

# Public access

namel3ss run app.n3 --host 0.0.0.0 --port 8000# Public access

 `eval`, `train`, `deploy`, and `doctor` subcommands are available for advanced workflows

# Execute a chain and emit JSONnamel3ss run app.n3 --host 0.0.0.0 --port 8000

namel3ss run summarize --json```

```

## Common Options

### Eval Experiments

```bash### Build Command

# Evaluate an experiment defined in app.n3- `--out <dir>` - Static output directory

namel3ss eval my_experiment -f app.n3 --format=textnamel3ss eval --help

```- `--build-backend` - Generate backend too

- `--backend-only` - Skip static site

### Train Models- `--backend-out <dir>` - Backend directory

```bash- `--target <static|react-vite>` - Select frontend generator

# Invoke custom training hook for a model- `--print-ast` - Debug: print AST

namel3ss train app.n3 --model my_model

```### Run Command

- `--backend-out <dir>` - Backend directory (default: temp)

### Deploy Models- `--host <host>` - Server host (default: 127.0.0.1)

```bash- `--port <port>` - Server port (default: 8000)

# Invoke custom deploy hook for a model- `--no-reload` - Disable hot reload

namel3ss deploy app.n3 --model my_model

```## Workflows



### Doctor### Development

```bash```bash

# Inspect optional dependencies and extras# 1. Create app.n3

namel3ss doctor# 2. Start dev server

```namel3ss run app.n3



## Common Options# 3. Edit app.n3 (auto-reloads)

# 4. Visit http://127.0.0.1:8000

### Build Command```

- `--out <dir>` - Static output directory

- `--build-backend` - Generate backend too### Production

- `--backend-only` - Skip static site```bash

- `--backend-out <dir>` - Backend directory# Generate everything

- `--target <static|react-vite>` - Select frontend generatornamel3ss build app.n3 --build-backend --out dist --backend-out api

- `--print-ast` - Debug: print AST

- `--realtime` - Include realtime websocket scaffolding in the backend# Deploy dist/ (static)

- `--embed-insights` - Inline insight results into dataset payloads# Deploy api/ (backend)

- `--env KEY=VALUE` - Set env vars for generation (repeatable)```



### Run Command### Testing

- `--backend-out <dir>` - Backend directory (default: temp)```bash

- `--host <host>` - Server host (default: 127.0.0.1)# Check AST

- `--port <port>` - Server port (default: 8000)namel3ss build app.n3 --print-ast

- `--no-reload` - Disable hot reload

- `--dev` - Force dev server mode even if the target looks like a chain# Quick backend test

- `--json` - Emit structured JSON when executing a chainnamel3ss run app.n3 --no-reload

- `--realtime` - Enable realtime websocket scaffolding in the generated backend```

- `--embed-insights` - Inline insight results into dataset payloads

- `--env KEY=VALUE` - Set env vars before launching the server (repeatable)## Error Messages



## Workflows| Error | Cause | Solution |

|-------|-------|----------|

### Development| File not found | Missing .n3 file | Check file path |

```bash| Syntax error | Invalid N3 syntax | Fix syntax (error shows line) |

# 1. Create app.n3| uvicorn not installed | Missing dependency | `pip install uvicorn[standard]` |

# 2. Start dev server

namel3ss run app.n3## Tips



# 3. Edit app.n3 (auto-reloads)✅ Use `run` during development (hot reload)

# 4. Visit http://127.0.0.1:8000✅ Use `build` for production (optimized)

```✅ `--backend-only` for API-only projects

✅ `--print-ast` for debugging syntax

### Production✅ Legacy mode still works but deprecated

```bash

# Generate everything## Examples

namel3ss build app.n3 --build-backend --out dist --backend-out api

```bash

# Deploy dist/ (static)# Multi-app development

# Deploy api/ (backend)namel3ss run app1.n3 --port 8000 &

```namel3ss run app2.n3 --port 8001 &



### Testing# Frontend + Backend split

```bashnamel3ss build app.n3 --build-backend --out www --backend-out server

# Check AST

namel3ss build app.n3 --print-ast# Inspect generated code

namel3ss build app.n3 --backend-only --backend-out inspect_me

# Quick backend test```

namel3ss run app.n3 --no-reload

## Help

# Inspect experiment behaviour```bash

namel3ss eval my_experiment -f app.n3 --format=textnamel3ss --help

```namel3ss build --help

namel3ss run --help

## Error Messages```


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
✅ `eval`, `train`, `deploy`, and `doctor` cover advanced workflows

## Examples

```bash
# Multi-app development
namel3ss run app1.n3 --port 8000 &
namel3ss run app2.n3 --port 8001 &

# Frontend + Backend split
namel3ss build app.n3 --build-backend --out www --backend-out server

# Inspect generated code
namel3ss build app.n3 --backend-only --backend-out inspect_me

# Doctor check
namel3ss doctor
```

## Help
```bash
namel3ss --help
namel3ss build --help
namel3ss run --help
namel3ss eval --help
namel3ss train --help
```
