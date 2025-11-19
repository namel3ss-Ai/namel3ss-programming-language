# Structured Prompts Implementation Guide

## Overview

Namel3ss now supports **structured prompts** with typed arguments and output schemas. This enables:

- **Type-safe prompt templates** with explicit argument declarations
- **Structured JSON outputs** with validation against declared schemas
- **Enum constraints** for classification tasks
- **Provider integration** for JSON mode / tool calling (OpenAI, Anthropic, etc.)
- **Runtime validation** of LLM outputs

## Syntax

### Basic Structure

```n3
prompt "prompt_name" {
    args: {
        arg_name: type [= default_value],
        ...
    }
    output_schema: {
        field_name: type,
        ...
    }
    model: "model_name"
    template: """
    Your prompt template with {arg_name} placeholders
    """
}
```

### Supported Types

**Argument Types:**
- `string` - Text
- `int` - Integer number
- `float` - Floating point number
- `bool` - Boolean (true/false)
- `list` - List/array
- `object` - Dictionary/object

**Output Schema Types:**
- Primitives: `string`, `int`, `float`, `bool`
- Enums: `enum["value1", "value2", ...]`
- Lists: `list[string]`, `list[int]`, etc.
- Nullable: Append `?` like `string?`

## Examples

### Example 1: Classification with Enum Output

```n3
llm "gpt4" {
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.7
}

prompt "classify_ticket" {
    args: {
        text: string
    }
    output_schema: {
        category: enum["billing", "technical", "account", "other"],
        urgency: enum["low", "medium", "high"],
        needs_handoff: bool
    }
    model: "gpt4"
    template: """
You are a support triage agent. Analyze the following support ticket and classify it.

Ticket: {text}

Provide a JSON response with:
- category: the main category of the issue
- urgency: the urgency level
- needs_handoff: whether this needs to be handed off to a human agent
"""
}
```

### Example 2: Prompt with Default Arguments

```n3
prompt "summarize" {
    args: {
        text: string,
        max_words: int = 100,
        style: string = "concise"
    }
    output_schema: {
        summary: string,
        key_points: list[string],
        word_count: int
    }
    model: "gpt4"
    template: """
Summarize the following text in approximately {max_words} words using a {style} style.

Text: {text}

Provide:
- summary: the main summary
- key_points: list of 3-5 key points
- word_count: actual word count of your summary
"""
}
```

### Example 3: Using in Chains

```n3
chain "triage_and_route" {
    step classification = prompt.classify_ticket(text = input.ticket_text) | llm.gpt4
    step summary = prompt.summarize(text = input.ticket_text, max_words = 50) | llm.gpt4
    
    output = {
        category: classification.category,
        urgency: classification.urgency,
        needs_handoff: classification.needs_handoff,
        summary: summary.summary
    }
}
```

## Implementation Details

### AST Nodes

**New AST Nodes:**
- `PromptArgument` - Typed argument with default value support
- `OutputFieldType` - Type specification (primitive, enum, list, object)
- `OutputField` - Field in output schema
- `OutputSchema` - Complete output schema with JSON Schema generation

### Parser

The parser handles:
- `args: { name: type = default }` blocks
- `output_schema: { field: type }` blocks
- `enum["val1", "val2"]` syntax
- `list[type]` syntax
- Inline or block templates

### Analyzer Validation

The resolver validates:
- Argument names are unique
- Argument types are valid
- Output schema field names are unique
- Output schema types are valid
- Enum values are non-empty and unique
- Template placeholders match defined args
- Model references exist

### Runtime Components

**PromptProgram** (`namel3ss/prompts/runtime.py`):
- `render_prompt(args)` - Validates args, applies defaults, renders template
- `get_output_schema()` - Returns JSON Schema dict
- Argument coercion and validation

**OutputValidator** (`namel3ss/prompts/validator.py`):
- `validate(output)` - Validates LLM output against schema
- Type checking (string, int, float, bool, list, enum)
- Required field validation
- Enum value validation
- Nested structure support

## Provider Integration (Next Steps)

To complete structured output support, providers need:

1. **JSON Mode Support** - Use provider's JSON mode when available
2. **Tool Calling** - Convert schema to function/tool definition
3. **Fallback** - Add format instructions to prompt for providers without JSON mode

### OpenAI Integration

```python
# In openai_llm.py
def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs):
    payload = {
        "model": self.model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        # Or use function calling:
        # "functions": [{"name": "output", "parameters": schema}]
    }
    # ... make API call, parse response
```

### Anthropic Integration

```python
# In anthropic_llm.py  
def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs):
    # Anthropic uses tools for structured output
    tool = {
        "name": "structured_output",
        "description": "Generate structured output",
        "input_schema": schema
    }
    # ... make API call with tools parameter
```

## Best Practices

1. **Keep schemas focused** - Small, task-specific outputs are easier to validate
2. **Use enums liberally** - Reduces hallucinations and validates categories
3. **Provide clear field descriptions** - Helps LLM understand requirements
4. **Set appropriate defaults** - Makes prompts easier to use
5. **Validate outputs** - Always use OutputValidator before consuming results
6. **Handle validation errors** - Implement retry logic or fallbacks

## Migration from Legacy Prompts

Legacy prompts using `input` and `output` blocks with `PromptField` are still supported for backward compatibility. To migrate:

**Before (Legacy):**
```n3
prompt "summarize" {
    input:
        text: text
        max_length: int
    output:
        summary: text
    using model "gpt4": """
    Summarize: {text}
    """
}
```

**After (Structured):**
```n3
prompt "summarize" {
    args: {
        text: string,
        max_length: int = 100
    }
    output_schema: {
        summary: string
    }
    model: "gpt4"
    template: """
    Summarize: {text}
    """
}
```

## Error Messages

Common validation errors:

- `"missing required argument: text"` - Required arg not provided
- `"unknown arguments: foo"` - Extra args passed
- `"cannot coerce value to type 'int'"` - Type mismatch
- `"Missing required field: category"` - LLM didn't return required field
- `"invalid enum value 'foo'"` - LLM returned value not in enum
- `"template references undefined arguments"` - Placeholder doesn't match arg

## Testing

Test files are located in `tests/`:
- `test_structured_prompts_parser.py` - Parser tests
- `test_structured_prompts_analyzer.py` - Analyzer validation tests
- `test_prompt_program.py` - Runtime tests
- `test_output_validator.py` - Validation tests

Run tests:
```bash
python -m pytest tests/test_structured_prompts_*.py -v
```

## Future Enhancements

- [ ] Nested object types in output schemas
- [ ] Cross-field validation rules
- [ ] Schema inheritance/composition
- [ ] Conditional required fields
- [ ] Provider-specific optimizations
- [ ] Automatic retry on validation failure
- [ ] Cost tracking for structured outputs
- [ ] Performance metrics
