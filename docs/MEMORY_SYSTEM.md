## Memory System in Namel3ss

# Production-Ready Memory for Stateful LLM Applications

## Overview

Namel3ss provides a **first-class memory system** that enables stateful interactions with LLMs, persistent conversation history, user preferences, and contextual data management. Memory is a declarative language construct with proper typing, scoping, and safety guarantees.

### Key Features

- **Declarative DSL**: Define memory stores as first-class language constructs
- **Multiple Scopes**: Session, user, conversation, page, and global scopes
- **Type Safety**: List, key-value, buffer, vector, and conversation types
- **Capacity Management**: Automatic eviction with configurable limits
- **Prompt Integration**: Reference memory directly in templates
- **Chain Integration**: Read/write memory as explicit chain steps
- **Observability**: Logging and metrics for memory operations
- **Pluggable Backends**: In-memory, Redis, database, or custom

---

## Memory DSL Syntax

### Basic Memory Definition

```n3
memory "conversation_history" {
  scope: "session"
  kind: "list"
  max_items: 100
  metadata: {
    description: "Stores conversation turns for current session"
  }
}
```

### Memory Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Unique identifier for the memory |
| `scope` | No | string | Scope of persistence (default: "session") |
| `kind` | No | string | Data structure type (default: "list") |
| `max_items` | No | int | Maximum items before eviction |
| `config` | No | dict | Backend-specific configuration |
| `metadata` | No | dict | Arbitrary metadata for documentation |

### Supported Scopes

- **`session`**: Per-session storage, cleared when session ends
- **`user`**: Per-user storage, persists across sessions
- **`conversation`**: Per-conversation storage for chat apps
- **`page`**: Per-page storage for UI state
- **`thread`**: Per-thread storage for concurrent workflows
- **`global`**: Shared across all users and sessions

### Supported Kinds

- **`list`**: Ordered collection with append/prepend operations
- **`conversation`**: Specialized list for chat messages (alias for list)
- **`key_value`** (or `kv`): Dictionary/map structure
- **`buffer`**: Single-value storage with overwrite
- **`vector`**: Vector embeddings for semantic search (future)

---

## Using Memory in Prompts

### Template References

Reference memory directly in prompt templates using `{memory.name}` syntax:

```n3
prompt "chat_reply" {
  model: "gpt4"
  
  args: {
    user_input: string
  }
  
  template: """
  You are a helpful assistant.
  
  Previous conversation:
  {memory.conversation_history}
  
  User: {user_input}
  Assistant:
  """
  
  output_schema: {
    response: string
  }
}
```

### Memory with Limits

Limit the number of items returned from list-type memory:

```n3
template: """
Recent context (last 5 turns):
{memory.conversation_history:5}

User: {user_input}
"""
```

### How It Works

1. At runtime, `{memory.conversation_history}` is resolved to actual memory contents
2. For list-type memory, items are formatted as numbered entries
3. For dict-type memory, contents are formatted as key-value pairs
4. Memory resolution happens **before** argument substitution

---

## Using Memory in Chains

### Reading Memory

Read memory into chain context at the start of execution:

```n3
define chain "chat_session" {
  steps:
    - step "load_history" {
        kind: "memory_read"
        target: "conversation_history"
        limit: 50
        options: {
          assign_to: "context.history"
        }
      }
    
    - step "generate_reply" {
        kind: "prompt"
        target: "chat_reply"
        options: {
          user_input: input.message
          history: context.history
        }
      }
}
```

### Writing Memory

Store chain step outputs to memory:

```n3
define chain "save_conversation" {
  steps:
    - step "generate_reply" {
        kind: "prompt"
        target: "chat_reply"
        options: {
          user_input: input.message
        }
      }
    
    - step "save_user_message" {
        kind: "memory_write"
        target: "conversation_history"
        options: {
          value: {
            role: "user"
            content: input.message
          }
        }
      }
    
    - step "save_assistant_message" {
        kind: "memory_write"
        target: "conversation_history"
        options: {
          value: {
            role: "assistant"
            content: steps.generate_reply.output.response
          }
        }
      }
}
```

### Inline Memory Operations

Use `read_memory` and `write_memory` options on any step:

```n3
- step "chat" {
    kind: "prompt"
    target: "chat_reply"
    options: {
      user_input: input.message
    }
    read_memory: ["conversation_history"]
    write_memory: ["conversation_history"]
  }
```

---

## Built-In Memory Functions

For advanced use cases, use Python-level memory functions:

### `read_memory(name, limit=None)`

Read contents from a memory store:

```python
history = await read_memory("conversation_history", limit=10)
```

### `write_memory(name, value)`

Replace entire memory contents:

```python
await write_memory("user_profile", {
    "name": "Alice",
    "preferences": {"theme": "dark"}
})
```

### `append_memory(name, item)`

Append item to list-type memory:

```python
await append_memory("conversation_history", {
    "role": "user",
    "content": "Hello!"
})
```

### `set_memory(name, value)`

Alias for `write_memory`:

```python
await set_memory("session_state", {"active": True})
```

### `clear_memory(name)`

Clear all data from a memory store:

```python
await clear_memory("conversation_history")
```

### `update_memory(name, fn)`

Apply transformation function to memory:

```python
await update_memory("counter", lambda data: {
    **data,
    "count": data["count"] + 1
})
```

### `get_key(memory_name, key, default=None)`

Get value from key-value memory:

```python
theme = await get_key("user_settings", "theme", default="light")
```

### `set_key(memory_name, key, value)`

Set key in key-value memory:

```python
await set_key("user_settings", "theme", "dark")
```

---

## Memory in PromptProgram

When using `PromptProgram` directly in Python:

```python
from namel3ss.prompts.runtime import PromptProgram
from namel3ss.codegen.backend.core.runtime.memory import get_memory_registry

# Get the global memory registry
registry = get_memory_registry()

# Create prompt program with memory support
program = PromptProgram(
    definition=prompt_definition,
    memory_registry=registry,
    scope_context={"session_id": "session-123", "user_id": "user-456"}
)

# Render prompt (memory references resolved automatically)
rendered = await program.render_prompt({"user_input": "Hello"})

# Or use memory methods directly
history = await program.read_memory("conversation_history", limit=10)
await program.append_memory("conversation_history", new_message)
```

---

## Complete Example

### Full Chat Application with Memory

```n3
# Define LLM
llm "gpt4" {
  provider: "openai"
  model: "gpt-4"
  api_key: env.OPENAI_API_KEY
  temperature: 0.7
}

# Define conversation memory
memory "chat_history" {
  scope: "user"
  kind: "list"
  max_items: 100
  metadata: {
    description: "Persistent chat history per user"
    eviction_policy: "oldest_first"
  }
}

# Define user preferences memory
memory "user_preferences" {
  scope: "user"
  kind: "key_value"
  metadata: {
    description: "User settings and preferences"
  }
}

# Chat reply prompt
prompt "chat_response" {
  model: "gpt4"
  
  args: {
    user_message: string,
    user_name: string = "User"
  }
  
  template: """
  You are a helpful AI assistant. Respond naturally and helpfully.
  
  Conversation history:
  {memory.chat_history:10}
  
  {user_name}: {user_message}
  Assistant:
  """
  
  output_schema: {
    response: string,
    sentiment: enum["positive", "neutral", "negative"],
    requires_followup: bool
  }
}

# Main chat chain
define chain "chat" {
  steps:
    # Load user preferences
    - step "load_prefs" {
        kind: "memory_read"
        target: "user_preferences"
        options: {
          assign_to: "context.prefs"
        }
      }
    
    # Generate response
    - step "generate" {
        kind: "prompt"
        target: "chat_response"
        options: {
          user_message: input.message
          user_name: context.prefs.name
        }
      }
    
    # Save user message to history
    - step "save_user_msg" {
        kind: "memory_write"
        target: "chat_history"
        options: {
          value: {
            role: "user"
            content: input.message
            timestamp: context.now
          }
        }
      }
    
    # Save assistant response to history
    - step "save_assistant_msg" {
        kind: "memory_write"
        target: "chat_history"
        options: {
          value: {
            role: "assistant"
            content: steps.generate.output.response
            sentiment: steps.generate.output.sentiment
            timestamp: context.now
          }
        }
      }
}
```

---

## Configuration

### Environment Variables

- `NAMEL3SS_MEMORY_BACKEND`: Backend type (`local`, `redis`, `db`)
- `NAMEL3SS_MEMORY_DEFAULT_SCOPE`: Default scope if not specified
- `NAMEL3SS_MEMORY_MAX_SIZE`: Global memory size limit

### Runtime Configuration

Set scope context at runtime:

```python
from namel3ss.codegen.backend.core.runtime.memory_functions import set_scope_context

# Called at request start
set_scope_context({
    "session_id": request.session_id,
    "user_id": request.user.id,
    "conversation_id": request.conversation_id,
})
```

---

## Best Practices

### 1. Use Appropriate Scopes

- **Session**: Temporary data like current form state
- **User**: Persistent data like preferences and history
- **Global**: Shared configuration or feature flags
- **Conversation**: Multi-turn chat within a specific conversation

### 2. Set Capacity Limits

Always set `max_items` for list-type memory to prevent unbounded growth:

```n3
memory "events" {
  kind: "list"
  max_items: 1000  # Keep last 1000 events
}
```

### 3. Use Descriptive Names

```n3
# Good
memory "user_conversation_history"
memory "session_form_state"
memory "global_feature_flags"

# Avoid
memory "data"
memory "temp"
```

### 4. Document Memory Purpose

Use metadata to document what each memory stores:

```n3
memory "analytics_events" {
  kind: "list"
  max_items: 500
  metadata: {
    description: "User interaction events for analytics"
    retention_days: 30
    privacy_level: "anonymized"
  }
}
```

### 5. Consider Memory Size

For large memories:
- Use limits to control size
- Consider summarization strategies
- Use pagination when reading

```n3
# Read recent items only
{memory.large_history:20}
```

### 6. Handle Missing Memory Gracefully

Prompts should handle empty memory:

```n3
template: """
{{#if memory.history}}
Previous context: {memory.history:5}
{{else}}
(No previous context)
{{/if}}

User: {user_input}
"""
```

---

## Security Considerations

### 1. Scope Isolation

Memory is automatically isolated by scope. Verify scope contexts are set correctly:

```python
# Ensure user_id is authenticated
set_scope_context({
    "user_id": authenticated_user.id  # From auth system
})
```

### 2. Don't Store Sensitive Data

Avoid storing:
- Raw passwords or API keys
- Credit card numbers
- Personal identifiable information (PII) without encryption

### 3. Implement Retention Policies

Clear memory based on retention requirements:

```python
# Clear session memory on logout
await clear_memory("session_state")

# Implement TTL for compliance
if user_inactive_for_days > 90:
    await clear_memory("user_conversation_history")
```

### 4. Validate Memory Content

Validate data before storing:

```python
# Sanitize user input
sanitized = sanitize_input(user_message)
await append_memory("chat_history", {
    "role": "user",
    "content": sanitized
})
```

---

## Troubleshooting

### Memory Not Found

**Error**: `MemoryNotFoundError: Memory 'xyz' not found`

**Solution**: Ensure memory is defined in your `.n3` file:

```n3
memory "xyz" {
  scope: "session"
  kind: "list"
}
```

### Type Mismatch

**Error**: `MemoryError: expected list but got dict`

**Solution**: Verify memory kind matches usage:

```n3
# For list operations (append)
memory "items" { kind: "list" }

# For dict operations (set_key/get_key)
memory "config" { kind: "key_value" }
```

### Scope Context Missing

**Error**: Memory reads return wrong data or are empty

**Solution**: Set scope context at request start:

```python
set_scope_context({
    "session_id": get_session_id(request),
    "user_id": get_user_id(request),
})
```

### Memory Grows Too Large

**Problem**: Memory consumes too much space

**Solution**: Set `max_items` with appropriate limit:

```n3
memory "logs" {
  kind: "list"
  max_items: 1000  # Evicts oldest items
}
```

---

## Advanced Topics

### Custom Memory Backends

Implement custom storage (e.g., Redis):

```python
from namel3ss.codegen.backend.core.runtime.memory import MemoryBackend

class RedisBackend(MemoryBackend):
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def read(self, key: str):
        data = await self.redis.get(key)
        return json.loads(data) if data else None
    
    async def write(self, key: str, value: Any):
        await self.redis.set(key, json.dumps(value))
    
    # Implement other methods...
```

### Memory Summarization

For long conversations, implement summarization:

```python
async def summarize_and_compress_memory(memory_name: str):
    history = await read_memory(memory_name)
    
    if len(history) > 50:
        # Get oldest 30 messages
        old_messages = history[:30]
        
        # Summarize using LLM
        summary = await llm.generate(f"Summarize: {old_messages}")
        
        # Keep summary + recent 20
        compressed = [
            {"role": "system", "content": f"Previous context: {summary}"}
        ] + history[-20:]
        
        await write_memory(memory_name, compressed)
```

### Conditional Memory Storage

Store to memory conditionally:

```n3
- step "chat" {
    kind: "prompt"
    target: "chat_response"
    options: {
      user_input: input.message
    }
  }

- step "save_if_important" {
    kind: "if"
    condition: steps.chat.output.requires_followup
    then:
      - step "save" {
          kind: "memory_write"
          target: "important_messages"
          options: {
            value: steps.chat.output
          }
        }
  }
```

---

## Appendix: Memory API Reference

### MemoryRegistry

```python
class MemoryRegistry:
    def register(spec: Dict[str, Any]) -> None
    def get(name: str, scope_context: Dict[str, str]) -> MemoryHandle
    def list_memories() -> List[str]
    def get_spec(name: str) -> Optional[MemorySpec]
```

### MemoryHandle

```python
class MemoryHandle:
    async def read(limit: Optional[int] = None, reverse: bool = False) -> Any
    async def write(value: Any) -> None
    async def append(item: Any) -> None
    async def update(fn: Callable[[Any], Any]) -> None
    async def clear() -> None
    async def set_key(key: str, value: Any) -> None
    async def get_key(key: str, default: Any = None) -> Any
```

### Memory Functions

```python
async def read_memory(name: str, limit: Optional[int] = None) -> Any
async def write_memory(name: str, value: Any) -> None
async def append_memory(name: str, item: Any) -> None
async def set_memory(name: str, value: Any) -> None
async def clear_memory(name: str) -> None
async def update_memory(name: str, fn: Callable) -> None
async def get_key(memory_name: str, key: str, default: Any = None) -> Any
async def set_key(memory_name: str, key: str, value: Any) -> None
```

---

## Summary

Namel3ss provides a **production-ready memory system** that:

✅ Integrates as a first-class language construct  
✅ Supports multiple scopes and data types  
✅ Works seamlessly with prompts and chains  
✅ Enforces capacity limits and type safety  
✅ Provides observability and error handling  
✅ Allows pluggable storage backends  

Memory enables building **stateful AI applications** with:
- Persistent conversation history
- User preferences and personalization
- Context management across sessions
- Workflow state tracking

For questions or issues, refer to the [Namel3ss documentation](https://github.com/namel3ss-ai/docs) or file an issue on GitHub.
