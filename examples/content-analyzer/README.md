# Content Analyzer Example

An AI-powered content analysis system using agents and memory.

## What it demonstrates:
- Agent definitions with LLM integration
- Memory policies for conversation tracking
- System prompts for AI behavior control
- Structured JSON output generation
- Content analysis workflows

## How to build:
```bash
# Backend only
namel3ss build examples/content-analyzer/app.ai --backend-only

# Full build (frontend + backend)
namel3ss build examples/content-analyzer/app.ai
```

## How to run:
```bash
namel3ss run examples/content-analyzer/app.ai
```

## Key concepts:
- **Agent system**: AI agents with specific roles and capabilities
- **Memory policies**: `conversation_window` for maintaining context
- **System prompts**: Detailed AI behavior instructions
- **Structured output**: JSON response formatting
- **Content analysis**: Topic detection, sentiment analysis, risk assessment