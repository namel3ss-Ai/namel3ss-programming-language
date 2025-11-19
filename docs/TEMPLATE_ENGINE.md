# Template Engine Guide

Namel3ss includes a production-grade template engine for building AI prompts with advanced features like conditionals, loops, filters, and nested object access. The template engine is built on Jinja2 with security sandboxing to prevent code injection.

## Table of Contents

- [Quick Start](#quick-start)
- [Syntax Overview](#syntax-overview)
- [Security Model](#security-model)
- [Variables](#variables)
- [Conditionals](#conditionals)
- [Loops](#loops)
- [Filters](#filters)
- [Best Practices](#best-practices)
- [Advanced Features](#advanced-features)

## Quick Start

```python
from namel3ss.prompts.template_prompt import TemplatePrompt

# Simple variable substitution
prompt = TemplatePrompt(
    name="greeting",
    template="Hello {{ name }}!",
)
result = prompt.render(name="World")
print(result.rendered)  # "Hello World!"
```

## Syntax Overview

Namel3ss templates use Jinja2 syntax:

- **Variables**: `{{ variable_name }}`
- **Conditionals**: `{% if condition %} ... {% endif %}`
- **Loops**: `{% for item in items %} ... {% endfor %}`
- **Filters**: `{{ value|filter_name }}`
- **Comments**: `{# This is a comment #}`

## Security Model

The template engine runs in a **sandboxed environment** with the following security features:

### What's Blocked

- ❌ No `__import__`, `eval()`, `exec()`, `compile()`
- ❌ No `__builtins__`, `__class__`, `__bases__`, `__subclasses__`
- ❌ No filesystem access (`open()`)
- ❌ No arbitrary code execution
- ❌ Compile-time validation catches dangerous patterns

### What's Allowed

- ✅ Safe built-in functions: `range`, `len`, `str`, `int`, `float`, `bool`, `list`, `dict`, `min`, `max`, `sum`, `sorted`, `enumerate`, `zip`
- ✅ Variable access and nested object access
- ✅ Conditionals and loops
- ✅ Custom filters (see below)

### Security Example

```python
# This will raise TemplateSecurityError at compile time
prompt = TemplatePrompt(
    name="dangerous",
    template="{{ __import__('os').system('ls') }}",
)
# TemplateSecurityError: Template contains dangerous pattern: __import__
```

## Variables

### Simple Variables

```python
prompt = TemplatePrompt(
    name="intro",
    template="Hello {{ name }}, welcome to {{ place }}!",
)
result = prompt.render(name="Alice", place="Wonderland")
# "Hello Alice, welcome to Wonderland!"
```

### Nested Objects

Access nested dictionary and object properties:

```python
prompt = TemplatePrompt(
    name="profile",
    template="User: {{ user.profile.name }}, Age: {{ user.profile.age }}",
)
result = prompt.render(
    user={
        "profile": {
            "name": "Alice",
            "age": 30,
        }
    }
)
# "User: Alice, Age: 30"
```

### List Access

```python
prompt = TemplatePrompt(
    name="items",
    template="First: {{ items[0] }}, Last: {{ items[-1] }}",
)
result = prompt.render(items=["apple", "banana", "cherry"])
# "First: apple, Last: cherry"
```

### Default Values

```python
prompt = TemplatePrompt(
    name="greeting",
    template="Hello {{ name }}!",
    args={
        "name": {"default": "World"},
    },
)
result = prompt.render()  # Uses default
# "Hello World!"
```

## Conditionals

### If/Else

```python
prompt = TemplatePrompt(
    name="access",
    template="""
{% if premium %}
Welcome, Premium Member! Access all features.
{% else %}
Welcome! Upgrade to Premium for full access.
{% endif %}
""",
)

result = prompt.render(premium=True)
# "Welcome, Premium Member! Access all features."
```

### If/Elif/Else

```python
prompt = TemplatePrompt(
    name="grade",
    template="""
Grade: {% if score >= 90 %}A{% elif score >= 80 %}B{% elif score >= 70 %}C{% else %}F{% endif %}
""",
)

result = prompt.render(score=85)
# "Grade: B"
```

### Conditional Instructions (AI Prompts)

```python
prompt = TemplatePrompt(
    name="ai_instruct",
    template="""
You are a helpful assistant.

{% if formal %}
Use formal, professional language.
{% else %}
Use casual, friendly language.
{% endif %}

{% if max_words %}
Limit your response to {{ max_words }} words.
{% endif %}

Answer: {{ question }}
""",
)

result = prompt.render(
    formal=True,
    max_words=100,
    question="What is Python?",
)
```

## Loops

### For Loop Over List

```python
prompt = TemplatePrompt(
    name="list_items",
    template="""
Items:
{% for item in items %}
- {{ item }}
{% endfor %}
""",
)

result = prompt.render(items=["apple", "banana", "cherry"])
# "Items:\n- apple\n- banana\n- cherry"
```

### For Loop with Index

```python
prompt = TemplatePrompt(
    name="numbered",
    template="""
{% for item in items %}
{{ loop.index }}. {{ item }}
{% endfor %}
""",
)

result = prompt.render(items=["first", "second", "third"])
# "1. first\n2. second\n3. third"
```

### Loop Variables

Within loops, you have access to:

- `loop.index` - 1-based index
- `loop.index0` - 0-based index
- `loop.first` - True if first iteration
- `loop.last` - True if last iteration
- `loop.length` - Total number of items

### RAG Context Loop

```python
prompt = TemplatePrompt(
    name="rag_qa",
    template="""
Answer the question based on these documents:

{% for doc in documents %}
Document {{ loop.index }}:
{{ doc.content }}
Source: {{ doc.metadata.source }}

{% endfor %}

Question: {{ question }}
""",
)

result = prompt.render(
    documents=[
        {"content": "Python is a language.", "metadata": {"source": "doc1.txt"}},
        {"content": "Created by Guido.", "metadata": {"source": "doc2.txt"}},
    ],
    question="What is Python?",
)
```

## Filters

Filters modify variables: `{{ value|filter_name(args) }}`

### Built-in Filters

#### `truncate(length, suffix='...')`

Truncate text to specified length:

```python
{{ text|truncate(50) }}
{{ long_text|truncate(100, '… [more]') }}
```

#### `uppercase`, `lowercase`, `title`

Case conversion:

```python
{{ name|uppercase }}  # "ALICE"
{{ name|lowercase }}  # "alice"
{{ name|title }}      # "Alice Smith"
```

#### `strip`

Remove leading/trailing whitespace:

```python
{{ "  hello  "|strip }}  # "hello"
```

#### `json_encode(indent=None)`

Convert to JSON:

```python
{{ data|json_encode }}
{{ data|json_encode(2) }}  # Pretty-printed
```

#### `list_join(separator=', ')`

Join list into string:

```python
{{ tags|list_join(', ') }}
{{ items|list_join(' | ') }}
```

#### `default(default_value='')`

Provide default for None or empty:

```python
{{ value|default('N/A') }}
{{ optional|default('Not provided') }}
```

#### `length`

Get length of collection or string:

```python
{{ items|length }}  # "5"
{{ "hello"|length }}  # "5"
```

#### `format_date(format='%Y-%m-%d')`

Format datetime objects:

```python
{{ created_at|format_date('%Y-%m-%d') }}
{{ timestamp|format_date('%B %d, %Y') }}
```

### Chaining Filters

```python
{{ text|strip|uppercase|truncate(50) }}
{{ user_input|default('Anonymous')|title }}
```

### Filters in AI Prompts

```python
prompt = TemplatePrompt(
    name="summary",
    template="""
Documents ({{ documents|length }} total):

{% for doc in documents %}
- {{ doc.title|title }}: {{ doc.content|truncate(100) }}
  Tags: {{ doc.tags|list_join(', ') }}
{% endfor %}

Summarize the above {{ documents|length }} documents.
""",
)
```

## Best Practices

### 1. Use Compile-Time Validation

Templates are compiled at initialization, catching syntax errors early:

```python
# Syntax error caught immediately
prompt = TemplatePrompt(
    name="invalid",
    template="{% if True %}unclosed",
)
# TemplateCompilationError: Template syntax error: unexpected end of template
```

### 2. Provide Defaults for Optional Variables

```python
prompt = TemplatePrompt(
    name="flexible",
    template="""
{{ greeting|default('Hello') }} {{ name }}!
{% if context %}Context: {{ context }}{% endif %}
""",
    args={
        "greeting": {"default": "Hello", "required": False},
        "context": {"default": None, "required": False},
    },
)

# Works with minimal input
result = prompt.render(name="Alice")
```

### 3. Use Conditionals for Optional Content

```python
template="""
System: {{ system_message }}

{% if examples %}
Examples:
{% for ex in examples %}
Input: {{ ex.input }}
Output: {{ ex.output }}
{% endfor %}
{% endif %}

User: {{ user_message }}
"""
```

### 4. Structure RAG Prompts with Loops

```python
template="""
Answer based on context:

{% for doc in context %}
Document {{ loop.index }}:
{{ doc.content|truncate(500) }}
{% if doc.score %}Relevance: {{ doc.score }}{% endif %}

{% endfor %}

Question: {{ question }}
"""
```

### 5. Validate Required Variables

```python
prompt = TemplatePrompt(
    name="strict",
    template="{{ required_field }}",
    args={
        "required_field": {"required": True},
    },
)

# This will raise PromptError
result = prompt.render()  # Missing required_field
```

## Advanced Features

### Nested Loops

```python
template="""
{% for category in categories %}
Category: {{ category.name }}
  Items:
  {% for item in category.items %}
  - {{ item }}
  {% endfor %}
{% endfor %}
"""
```

### Complex Conditionals

```python
template="""
{% if user.role == 'admin' and user.verified %}
Admin Panel Access
{% elif user.role == 'user' and user.verified %}
User Dashboard
{% else %}
Please verify your account
{% endif %}
"""
```

### Combining All Features

Few-shot learning prompt:

```python
prompt = TemplatePrompt(
    name="few_shot",
    template="""
Task: {{ task }}

{% if instructions %}
Instructions:
{% for instruction in instructions %}
- {{ instruction }}
{% endfor %}
{% endif %}

{% if examples %}
Examples ({{ examples|length }} total):
{% for ex in examples %}
{{ loop.index }}. Input: {{ ex.input }}
   Output: {{ ex.output }}
{% endfor %}
{% endif %}

Now complete this task:
Input: {{ input }}
Output: {{ output|default('[Generate output here]') }}
""",
)

result = prompt.render(
    task="Classify sentiment",
    instructions=[
        "Analyze the emotional tone",
        "Consider context and sarcasm",
        "Output: positive, negative, or neutral",
    ],
    examples=[
        {"input": "I love this!", "output": "positive"},
        {"input": "This is terrible", "output": "negative"},
    ],
    input="This is amazing!",
)
```

## Error Handling

### Compilation Errors

```python
try:
    prompt = TemplatePrompt(
        name="test",
        template="{% if True %}unclosed",
    )
except PromptError as e:
    print(f"Compilation failed: {e}")
    # Template name: test
```

### Rendering Errors

```python
try:
    result = prompt.render(missing_var="value")
except PromptError as e:
    print(f"Missing: {', '.join(e.missing_vars)}")
    print(f"Prompt: {e.prompt_name}")
```

### Security Errors

```python
try:
    prompt = TemplatePrompt(
        name="dangerous",
        template="{{ __import__('os') }}",
    )
except PromptError as e:
    print(f"Security violation: {e}")
```

## Integration with Namel3ss DSL

Templates integrate seamlessly with Namel3ss AI primitives:

```n3
# Define prompt with template
prompt MyPrompt(query: str, context: list):
    template: """
    Context:
    {% for doc in context %}
    - {{ doc.content|truncate(200) }}
    {% endfor %}
    
    Question: {{ query }}
    """
    model: "gpt-4"

# Use in chain
chain QAChain:
    docs = retrieve(query)
    answer = MyPrompt(query=query, context=docs)
    return answer
```

## Migration from Old Format

### Before (naive string substitution)

```python
template = "Hello {name}!"  # Basic .format() style
```

### After (Jinja2 with advanced features)

```python
template = """
Hello {{ name|title }}!

{% if premium %}
You have access to premium features.
{% endif %}

Recent items:
{% for item in items %}
- {{ item|truncate(50) }}
{% endfor %}
"""
```

## Performance Notes

- Templates are **compiled once** at initialization
- Compiled templates are **reused** for multiple renders
- No runtime compilation overhead
- Security validation happens at compile time

## See Also

- [Prompt System Documentation](./PROMPTS.md)
- [RAG Pipeline Guide](./RAG_RERANKING.md)
- [AI DSL Syntax](./CONTROL_FLOW_SYNTAX.md)
