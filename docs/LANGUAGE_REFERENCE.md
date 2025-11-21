# Namel3ss Language Reference

**Version:** 1.0.0  
**Date:** November 21, 2025

Complete reference guide for the Namel3ss (N3) programming language - an AI-native DSL for building production backends and frontends.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Module Structure](#module-structure)
3. [Declarations](#declarations)
4. [Expressions](#expressions)
5. [Control Flow](#control-flow)
6. [Type System](#type-system)
7. [Best Practices](#best-practices)

---

## Introduction

Namel3ss is an English-like DSL designed for rapid application development with built-in AI capabilities. Key features:

- **Declarative Syntax**: Describe what you want, not how to build it
- **AI-Native**: First-class support for LLMs, agents, RAG, and training
- **Type-Safe**: Optional type annotations with inference
- **Production-Ready**: Compiles to optimized backend/frontend code

---

## Module Structure

### Module Declaration

```n3
module "my_app"

import ai.models as models
import data.processing

language_version: "1.0"
```

**Rules:**
- Module declaration must be first (if present)
- Imports must come before other declarations
- `language_version` specifies N3 version

---

## Declarations

### Application

The root container for your application:

```n3
app "Customer Portal" connects to postgres "customer_db" {
  description: "Customer-facing web application"
  version: "2.0.0"
}
```

**Fields:**
- `name` (required): Application name
- `connects to`: Database connections
- `description`: Human-readable description
- `version`: Semantic version

### Page

UI pages with declarative components:

```n3
page "Dashboard" at "/dashboard" {
  show text {
    title: "Welcome"
    content: "Dashboard overview"
  }
  
  show table {
    title: "Recent Orders"
    source: dataset("orders")
  }
  
  if user.role == "admin" {
    show button {
      label: "Admin Panel"
      action: goto("/admin")
    }
  }
}
```

**Fields:**
- `name` (required): Page identifier
- `route` (required): URL path
- `statements`: UI components and logic

**Component Types:**
- `text`, `table`, `chart`, `form`, `button`, `input`, `select`

### LLM

Large Language Model configuration:

```n3
llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
  api_key: env.OPENAI_API_KEY
  temperature: 0.7
  max_tokens: 2000
  top_p: 0.9
}
```

**Fields:**
- `provider`: `"openai"`, `"anthropic"`, `"cohere"`, `"ollama"`
- `model`: Model identifier
- `api_key`: API key (use `env.VAR_NAME` for environment variables)
- `temperature`: 0.0-2.0 (creativity)
- `max_tokens`: Response length limit
- `top_p`: Nucleus sampling parameter

### Agent

AI agents with tools and memory:

```n3
agent "customer_support" {
  llm: "gpt4"
  system_prompt: "You are a helpful customer support agent."
  tools: ["search_kb", "create_ticket", "check_order"]
  memory: "conversation_history"
  max_iterations: 10
  temperature: 0.7
}
```

**Fields:**
- `llm`: Reference to LLM declaration
- `system_prompt`: Agent instructions
- `tools`: List of tool names
- `memory`: Reference to memory store
- `max_iterations`: Max tool calls per turn

### Prompt

Structured prompt templates:

```n3
prompt "summarize_text" {
  input: [
    - text: text (required)
    - max_length: number (optional)
  ]
  
  output: [
    - summary: text
    - key_points: array
  ]
  
  template: """
    Summarize the following text in {max_length} words or less:
    
    {text}
    
    Provide:
    1. A concise summary
    2. Key points as a bullet list
  """
}
```

**Sections:**
- `input`: Input schema (array or object)
- `output`: Output schema
- `template`: Prompt template with `{var}` substitution

### Chain

Sequential or parallel workflows:

```n3
chain "analysis_pipeline" {
  input -> preprocess -> analyze -> summarize -> output
}

# With parallel steps
chain "parallel_analysis" {
  input -> (sentiment | entities | topics) -> merge -> output
}
```

**Connectors:**
- `->`: Sequential execution
- `|`: Parallel execution

### RAG Pipeline

Retrieval-Augmented Generation:

```n3
index "docs_index" {
  source_dataset: "documentation"
  embedding_model: "text-embedding-3-small"
  chunk_size: 512
  overlap: 64
  backend: "pgvector"
  table_name: "doc_embeddings"
}

rag_pipeline "doc_retrieval" {
  query_encoder: "text-embedding-3-small"
  index: "docs_index"
  top_k: 5
  distance_metric: "cosine"
  reranker: "cross_encoder"
}
```

**Index Fields:**
- `source_dataset`: Data source
- `embedding_model`: Vector embedding model
- `chunk_size`: Document chunk size
- `overlap`: Chunk overlap
- `backend`: `"pgvector"`, `"faiss"`, `"weaviate"`

**Pipeline Fields:**
- `query_encoder`: Query embedding model
- `index`: Reference to index
- `top_k`: Number of results
- `distance_metric`: `"cosine"`, `"euclidean"`, `"dot"`
- `reranker`: Optional reranker model

### Memory

Stateful storage for agents:

```n3
memory "conversation_history" {
  scope: "user"
  kind: "list"
  max_items: 100
  metadata: {
    description: "Chat message history"
    retention_days: 30
  }
}
```

**Scopes:**
- `"user"`: Per-user storage
- `"session"`: Per-session (cleared on logout)
- `"global"`: Application-wide
- `"buffer"`: Temporary (in-memory)

**Kinds:**
- `"list"`: Ordered collection
- `"key_value"`: Dictionary
- `"vector"`: Vector store
- `"graph"`: Graph structure

### Dataset

Data source declarations:

```n3
dataset "active_users" from postgres table users {
  filter: fn(user) => user.status == "active"
  cache_ttl: 3600
}
```

### Function

User-defined functions:

```n3
fn greet(name: text) => "Hello, " + name + "!"

fn calculate_tax(amount: number, rate: number): number =>
  amount * rate

fn process_data(items: array) => {
  let filtered = filter(items, fn(x) => x.active)
  let mapped = map(filtered, fn(x) => x.value)
  return sum(mapped)
}
```

---

## Expressions

### Literals

```n3
# Strings
"hello"
'world'
"""
Multi-line
string
"""

# Numbers
42
3.14
1.5e-10

# Booleans
true
false

# Null
null
```

### Operators

```n3
# Arithmetic
x + y    # Addition
x - y    # Subtraction
x * y    # Multiplication
x / y    # Division
x % y    # Modulo
x ** y   # Exponentiation

# Comparison
x == y   # Equal
x != y   # Not equal
x < y    # Less than
x > y    # Greater than
x <= y   # Less than or equal
x >= y   # Greater than or equal

# Logical
x && y   # AND
x || y   # OR
!x       # NOT

# Member access
obj.field
obj.method()
arr[0]
```

### Lambda Expressions

```n3
fn(x) => x * 2
fn(a, b) => a + b
fn(user) => user.age > 18 && user.verified
```

### Let Expressions

```n3
let x = 10 in
let y = 20 in
x + y
```

### Match Expressions

```n3
match status {
  case "pending" => "Waiting"
  case "approved" => "Confirmed"
  case "rejected" => "Denied"
  case _ => "Unknown"
}

# Array patterns
match items {
  case [] => "empty"
  case [x] => "single: " + str(x)
  case [first, ...rest] => "many, first: " + str(first)
}
```

---

## Control Flow

### If-Else

```n3
if condition {
  # then block
}

if x > 10 {
  show text "Large"
} else {
  show text "Small"
}
```

### For Loops

```n3
for item in items {
  show text item.name
}
```

### While Loops

```n3
while count < 10 {
  count = count + 1
}
```

---

## Type System

### Primitive Types

- `text`: String values
- `number`: Integer or float
- `boolean`: `true` or `false`
- `null`: Null value
- `any`: Any type

### Composite Types

```n3
# Arrays
number[]
text[]

# Objects
{
  name: text
  age: number
  active: boolean
}

# Functions
(text, number) => boolean

# Union types
text | number | null
```

### Type Annotations

```n3
fn add(a: number, b: number): number => a + b

let name: text = "Alice"
let count: number = 42
```

---

## Best Practices

### 1. Use Quoted Names

```n3
# ✅ Good
llm "gpt4" { }
agent "support_bot" { }

# ❌ Bad
llm gpt4 { }
agent support_bot { }
```

### 2. Prefer Block Syntax

```n3
# ✅ Good
llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
}

# ❌ Deprecated
llm gpt4:
  provider: openai
  model: gpt-4
```

### 3. Use Environment Variables for Secrets

```n3
# ✅ Good
llm "gpt4" {
  api_key: env.OPENAI_API_KEY
}

# ❌ Bad (hardcoded secret)
llm "gpt4" {
  api_key: "sk-1234..."
}
```

### 4. Type Your Functions

```n3
# ✅ Good
fn calculate(x: number, y: number): number => x + y

# ⚠️ Works but less safe
fn calculate(x, y) => x + y
```

### 5. Use Descriptive Names

```n3
# ✅ Good
agent "customer_support_agent" { }
memory "user_conversation_history" { }

# ❌ Bad
agent "agent1" { }
memory "mem" { }
```

### 6. Organize Code Logically

```n3
# Module directives first
module "my_app"
import ai.models

# App declaration
app "My App" { }

# LLMs and tools
llm "gpt4" { }
tool "search" { }

# Agents and chains
agent "support" { }
chain "pipeline" { }

# Pages last
page "Home" at "/" { }
```

---

## Advanced Features

### RAG + Agent Integration

```n3
index "kb_index" {
  source_dataset: "knowledge_base"
  embedding_model: "text-embedding-3-small"
}

rag_pipeline "kb_retrieval" {
  index: "kb_index"
  top_k: 3
}

agent "qa_agent" {
  llm: "gpt4"
  tools: ["kb_retrieval", "web_search"]
  memory: "conversation_history"
}

page "Q&A" at "/qa" {
  show chat {
    agent: "qa_agent"
    placeholder: "Ask me anything..."
  }
}
```

### Multi-Agent Workflows

```n3
agent "researcher" {
  llm: "gpt4"
  tools: ["web_search", "arxiv_search"]
}

agent "writer" {
  llm: "gpt4"
  tools: ["grammar_check"]
}

agent "reviewer" {
  llm: "gpt4"
  tools: ["fact_check"]
}

graph "research_workflow" {
  nodes: ["researcher", "writer", "reviewer"]
  edges: [
    ["researcher", "writer"],
    ["writer", "reviewer"],
    ["reviewer", "writer"]
  ]
  entry: "researcher"
  max_iterations: 5
}
```

---

## Appendix: Complete Example

```n3
module "customer_portal"

import ai.models as models

language_version: "1.0"

app "Customer Portal" connects to postgres "customer_db" {
  description: "AI-powered customer support portal"
  version: "1.0.0"
}

llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
  api_key: env.OPENAI_API_KEY
  temperature: 0.7
}

memory "conversation_history" {
  scope: "user"
  kind: "list"
  max_items: 50
}

index "support_kb" {
  source_dataset: "knowledge_base"
  embedding_model: "text-embedding-3-small"
  chunk_size: 512
}

rag_pipeline "kb_retrieval" {
  index: "support_kb"
  top_k: 3
}

agent "support_agent" {
  llm: "gpt4"
  system_prompt: "You are a helpful customer support agent."
  tools: ["kb_retrieval", "create_ticket"]
  memory: "conversation_history"
}

page "Home" at "/" {
  show text {
    title: "Welcome to Customer Portal"
    content: "How can we help you today?"
  }
  
  show chat {
    agent: "support_agent"
    placeholder: "Type your question..."
  }
}
```

---

**For more information:**
- [Grammar Specification](GRAMMAR.md)
- [Parser Internals](PARSER_INTERNALS.md)
- [Migration Guide](MIGRATION_GUIDE.md)
