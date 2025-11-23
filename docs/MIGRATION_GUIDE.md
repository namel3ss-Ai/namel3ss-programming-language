# Namel3ss Migration Guide

**Version:** 1.0.0 → Unified Parser  
**Date:** November 21, 2025

This guide helps you migrate your N3 code from legacy syntax to the new canonical syntax.

---

## Overview

The Namel3ss compiler has been unified into a single, deterministic parser with consistent syntax. All legacy syntax variations have been deprecated and replaced with a canonical brace-based syntax.

### Key Changes

1. **Mandatory quoted names**: All declaration names must be in quotes
2. **Brace-based blocks**: All declarations use `{ }` instead of `:` with indentation
3. **No dual-parser fallback**: Parser strictly enforces canonical syntax
4. **Improved error messages**: Line numbers, expected tokens, and suggestions

---

## Syntax Migration

### 1. Application Declarations

**Before (Legacy):**
```n3
app "My App" connects to postgres "db".

app My App.
```

**After (Canonical):**
```n3
app "My App" connects to postgres "db" {
  description: "Application description"
  version: "1.0.0"
}
```

**Rules:**
- Names must be quoted
- Use `{ }` for configuration block
- No trailing `.` after app name
- Empty app bodies still need `{ }`

---

### 2. LLM Declarations

**Before (Legacy):**
```n3
llm gpt4:
    provider: openai
    model: gpt-4
    temperature: 0.7

llm chat_model {
    provider: "openai"
}
```

**After (Canonical):**
```n3
llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
}

llm "chat_model" {
  provider: "openai"
}
```

**Rules:**
- LLM name must be quoted
- Always use `{ }` (no colon syntax)
- Indent consistently (2 or 4 spaces)

---

### 3. Prompt Declarations

**Before (Legacy):**
```n3
prompt greeting:
    name: "Greeting"
    template: "Hello {{ name }}"

prompt simple_prompt:
    template: """
    System prompt here
    """
```

**After (Canonical):**
```n3
prompt "greeting" {
  input: [
    - name: text (required)
  ]
  
  output: [
    - greeting: text
  ]
  
  template: "Hello {{ name }}"
}

prompt "simple_prompt" {
  template: """
  System prompt here
  """
}
```

**Rules:**
- Prompt name must be quoted
- Use structured `input`/`output` schemas
- `template` is a field in the block

---

### 4. Agent Declarations

**Before (Legacy):**
```n3
agent support_agent:
    llm: gpt4
    tools: [search, create_ticket]
```

**After (Canonical):**
```n3
agent "support_agent" {
  llm: "gpt4"
  system_prompt: "You are a helpful agent."
  tools: ["search", "create_ticket"]
  memory: "conversation_history"
}
```

**Rules:**
- Agent name must be quoted
- Tool names must be quoted in array
- Always use brace syntax

---

### 5. Memory Declarations

**Before (Legacy):**
```n3
memory chat_history:
    scope: user
    kind: list
```

**After (Canonical):**
```n3
memory "chat_history" {
  scope: "user"
  kind: "list"
  max_items: 100
}
```

---

### 6. Dataset Declarations

**Before (Legacy):**
```n3
dataset active_users from table users:
    filter by: status == "active"

dataset users from postgres users.
```

**After (Canonical):**
```n3
dataset "active_users" from table users {
  filter: fn(user) => user.status == "active"
}

dataset "users" from table users {
  cache_ttl: 3600
}
```

**Rules:**
- Dataset name must be quoted
- `filter` is now a lambda expression
- Use `{ }` for configuration

---

### 7. Page Declarations

**Before (Legacy):**
```n3
page Home at "/":
    show text "Welcome"
    show table "Users" from dataset users

page "Dashboard":
    ...
```

**After (Canonical):**
```n3
page "Home" at "/" {
  show text {
    title: "Welcome"
    content: "Welcome message"
  }
  
  show table {
    title: "Users"
    source: "users"
  }
}

page "Dashboard" at "/dashboard" {
  # Page content
}
```

**Rules:**
- Page name must be quoted
- `at` keyword required with route
- `show` statements use config objects
- Always use `{ }` for page body

---

### 8. Chain Declarations

**Before (Legacy):**
```n3
chain analysis:
    input -> preprocess -> analyze -> output
```

**After (Canonical):**
```n3
chain "analysis" {
  steps: ["input", "preprocess", "analyze", "output"]
}

# Or with inline step definitions
chain "pipeline" {
  input -> rag "retrieval" -> llm "gpt4" -> output
}
```

---

### 9. RAG Pipeline Declarations

**Before (Legacy):**
```n3
index docs:
    source: documentation
    embedding: text-embedding-3-small

rag_pipeline retrieval:
    index: docs
    top_k: 5
```

**After (Canonical):**
```n3
index "docs" {
  source_dataset: "documentation"
  embedding_model: "text-embedding-3-small"
  chunk_size: 512
  backend: "pgvector"
}

rag_pipeline "retrieval" {
  query_encoder: "text-embedding-3-small"
  index: "docs"
  top_k: 5
  distance_metric: "cosine"
}
```

---

## Automated Migration

### Migration Script

```python
#!/usr/bin/env python3
"""Migrate N3 files from legacy to canonical syntax."""

import re
from pathlib import Path

def migrate_file(content: str) -> str:
    """Migrate N3 content to canonical syntax."""
    
    # Fix app declarations
    content = re.sub(
        r'app "([^"]+)"\.', 
        r'app "\1" {\n  description: "Migrated app"\n}',
        content
    )
    
    # Fix LLM declarations with colon
    content = re.sub(
        r'llm (\w+):', 
        r'llm "\1" {',
        content
    )
    
    # Fix prompt declarations
    content = re.sub(
        r'prompt (\w+):', 
        r'prompt "\1" {',
        content
    )
    
    # Fix agent declarations
    content = re.sub(
        r'agent (\w+):', 
        r'agent "\1" {',
        content
    )
    
    # Fix memory declarations
    content = re.sub(
        r'memory (\w+):', 
        r'memory "\1" {',
        content
    )
    
    # Fix dataset declarations
    content = re.sub(
        r'dataset (\w+) from', 
        r'dataset "\1" from',
        content
    )
    
    # Fix chain declarations
    content = re.sub(
        r'chain (\w+):', 
        r'chain "\1" {',
        content
    )
    
    # Fix page declarations
    content = re.sub(
        r'page (\w+) at', 
        r'page "\1" at',
        content
    )
    
    return content

def migrate_directory(path: Path):
    """Migrate all N3 files in directory."""
    for file in path.rglob("*.ai"):
        print(f"Migrating {file}...")
        content = file.read_text()
        migrated = migrate_file(content)
        
        # Backup original
        backup = file.with_suffix('.ai.bak')
        file.rename(backup)
        
        # Write migrated version
        file.write_text(migrated)
        print(f"  ✓ Migrated (backup: {backup})")

if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    migrate_directory(path)
    print("\n✅ Migration complete!")
```

**Usage:**
```bash
python migrate.py /path/to/n3/files
```

---

## Common Migration Issues

### Issue 1: Unquoted Names

**Error:**
```
Line 4:8 | Expected: string
Found: identifier
Suggestion: Did you mean "my_llm"?
```

**Fix:**
Add quotes around declaration names:
```n3
# Before
llm my_llm {

# After
llm "my_llm" {
```

---

### Issue 2: Missing Braces

**Error:**
```
Line 3:20 | Expected: lbrace
Found: newline
```

**Fix:**
Replace colon with braces:
```n3
# Before
app "My App" connects to postgres "db".

# After
app "My App" connects to postgres "db" {
  description: "My application"
}
```

---

### Issue 3: Inconsistent Indentation

**Error:**
```
Line 8:5 | Inconsistent indentation: 5 spaces
```

**Fix:**
Use consistent indentation (2 or 4 spaces, not mixed):
```n3
# Before
app "Test" {
  field1: "value"
     field2: "value"  # Wrong: 5 spaces

# After
app "Test" {
  field1: "value"
  field2: "value"  # Correct: 2 spaces
}
```

---

## Validation

After migration, validate your N3 files:

```bash
# Parse and check syntax
python -c "
from namel3ss.lang.parser import parse_module

with open('your_file.ai') as f:
    module = parse_module(f.read(), 'your_file.ai')
    print('✓ Syntax valid!')
"

# Run test suite
pytest tests/test_official_examples.py -v
```

---

## Breaking Changes Summary

| Feature | Legacy | Canonical | Breaking |
|---------|--------|-----------|----------|
| Declaration names | Optional quotes | Required quotes | ✅ Yes |
| Block syntax | `:` with indent | `{ }` always | ✅ Yes |
| Trailing `.` | Allowed | Not allowed | ✅ Yes |
| App body | Optional | Required (can be empty) | ✅ Yes |
| Filter syntax | `filter by:` | `filter:` (lambda) | ✅ Yes |
| Show statements | String arg | Config object | ⚠️ Partial |

---

## Deprecation Timeline

- **v1.0** (Nov 2025): Unified parser released, legacy syntax deprecated
- **v1.1** (Dec 2025): Warning on legacy syntax
- **v2.0** (Jan 2026): Legacy syntax removed completely

---

## Support

If you encounter migration issues:

1. Check error messages - they include suggestions
2. Review [Language Reference](LANGUAGE_REFERENCE.md)
3. Check [Grammar Specification](GRAMMAR.md)
4. File an issue with example code

---

## Examples

### Complete Before/After

**Before:**
```n3
app CustomerPortal connects to postgres customer_db.

llm gpt4:
    provider: openai
    model: gpt-4

agent support:
    llm: gpt4
    tools: [search]

page Home at "/":
    show text "Welcome"
```

**After:**
```n3
app "CustomerPortal" connects to postgres "customer_db" {
  description: "Customer portal application"
  version: "1.0.0"
}

llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
}

agent "support" {
  llm: "gpt4"
  system_prompt: "You are a support agent."
  tools: ["search"]
}

page "Home" at "/" {
  show text {
    content: "Welcome"
  }
}
```

---

**End of Migration Guide**
