# Research Assistant Example

A multi-turn research assistant that synthesizes information and tracks conversation history.

## What it demonstrates:
- Multi-turn conversation workflows
- Research and information synthesis
- Advanced prompt templates with history tracking
- Agent memory management
- Structured research methodology

## How to build:
```bash
# Backend only
namel3ss build examples/research-assistant/app.ai --backend-only

# Full build (frontend + backend)
namel3ss build examples/research-assistant/app.ai
```

## How to run:
```bash
namel3ss run examples/research-assistant/app.ai
```

## Key concepts:
- **Research workflows**: Systematic information gathering
- **Conversation tracking**: Maintaining research context across turns
- **Information synthesis**: Combining multiple sources
- **Structured prompts**: Complex template with multiple variables
- **Agent memory**: `conversation_window` for research continuity