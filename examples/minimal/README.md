# Minimal Example

A basic N3 application demonstrating core language features.

## What it demonstrates:
- Basic app structure with description
- LLM configuration with OpenAI provider
- Simple prompt template with variable substitution

## How to build:
```bash
# Backend only
namel3ss build examples/minimal/app.ai --backend-only

# Full build (frontend + backend)
namel3ss build examples/minimal/app.ai
```

## How to run:
```bash
namel3ss run examples/minimal/app.ai
```

## Key concepts:
- **App declaration**: Basic app metadata
- **LLM setup**: Provider, model, and parameter configuration
- **Prompt templates**: Variable interpolation with `{{variable}}` syntax