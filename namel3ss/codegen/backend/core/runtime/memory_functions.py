"""
Built-in memory functions for Namel3ss runtime.

Provides high-level, easy-to-use functions for memory operations
that can be used in expressions, chains, and prompts.
"""

from typing import Any, Callable, Dict, List, Optional

from .memory import MemoryRegistry, get_memory_registry


# Global scope context (updated per request/session)
_SCOPE_CONTEXT: Dict[str, str] = {}


def set_scope_context(context: Dict[str, str]) -> None:
    """
    Set the scope context for memory operations.
    
    Should be called at the start of each request with:
    - session_id: Unique session identifier
    - user_id: User identifier (if authenticated)
    - conversation_id: Conversation identifier (for chat apps)
    - page_id: Page identifier (for page-scoped memory)
    
    Args:
        context: Dictionary of scope identifiers
    """
    global _SCOPE_CONTEXT
    _SCOPE_CONTEXT = context


def get_scope_context() -> Dict[str, str]:
    """Get the current scope context."""
    return dict(_SCOPE_CONTEXT)


async def read_memory(name: str, *, limit: Optional[int] = None, registry: Optional[MemoryRegistry] = None) -> Any:
    """
    Read from a memory store.
    
    Args:
        name: Memory name
        limit: Optional limit on items returned (for list-type memory)
        registry: Optional custom registry (defaults to global)
    
    Returns:
        Memory contents
    
    Example:
        history = await read_memory("conversation_history", limit=10)
    """
    reg = registry or get_memory_registry()
    handle = reg.get(name, scope_context=_SCOPE_CONTEXT)
    return await handle.read(limit=limit)


async def write_memory(name: str, value: Any, *, registry: Optional[MemoryRegistry] = None) -> None:
    """
    Write to a memory store, replacing existing content.
    
    Args:
        name: Memory name
        value: Value to write
        registry: Optional custom registry (defaults to global)
    
    Example:
        await write_memory("user_profile", {"name": "Alice", "preferences": {...}})
    """
    reg = registry or get_memory_registry()
    handle = reg.get(name, scope_context=_SCOPE_CONTEXT)
    await handle.write(value)


async def append_memory(name: str, item: Any, *, registry: Optional[MemoryRegistry] = None) -> None:
    """
    Append item to a list-type memory store.
    
    Args:
        name: Memory name
        item: Item to append
        registry: Optional custom registry (defaults to global)
    
    Example:
        await append_memory("conversation_history", {
            "role": "user",
            "content": "Hello!"
        })
    """
    reg = registry or get_memory_registry()
    handle = reg.get(name, scope_context=_SCOPE_CONTEXT)
    await handle.append(item)


async def set_memory(name: str, value: Any, *, registry: Optional[MemoryRegistry] = None) -> None:
    """
    Alias for write_memory - sets memory content.
    
    Args:
        name: Memory name
        value: Value to write
        registry: Optional custom registry (defaults to global)
    """
    await write_memory(name, value, registry=registry)


async def clear_memory(name: str, *, registry: Optional[MemoryRegistry] = None) -> None:
    """
    Clear all data from a memory store.
    
    Args:
        name: Memory name
        registry: Optional custom registry (defaults to global)
    
    Example:
        await clear_memory("conversation_history")
    """
    reg = registry or get_memory_registry()
    handle = reg.get(name, scope_context=_SCOPE_CONTEXT)
    await handle.clear()


async def update_memory(name: str, fn: Callable[[Any], Any], *, registry: Optional[MemoryRegistry] = None) -> None:
    """
    Update memory by applying a transformation function.
    
    Args:
        name: Memory name
        fn: Function that takes current value and returns new value
        registry: Optional custom registry (defaults to global)
    
    Example:
        # Append to list
        await update_memory("items", lambda items: items + [new_item])
        
        # Update dict
        await update_memory("config", lambda cfg: {**cfg, "updated": True})
    """
    reg = registry or get_memory_registry()
    handle = reg.get(name, scope_context=_SCOPE_CONTEXT)
    await handle.update(fn)


async def get_key(memory_name: str, key: str, default: Any = None, *, registry: Optional[MemoryRegistry] = None) -> Any:
    """
    Get a value by key from key_value type memory.
    
    Args:
        memory_name: Memory name
        key: Key name
        default: Default value if key not found
        registry: Optional custom registry (defaults to global)
    
    Returns:
        Value associated with key, or default
    
    Example:
        pref = await get_key("user_settings", "theme", default="dark")
    """
    reg = registry or get_memory_registry()
    handle = reg.get(memory_name, scope_context=_SCOPE_CONTEXT)
    return await handle.get_key(key, default)


async def set_key(memory_name: str, key: str, value: Any, *, registry: Optional[MemoryRegistry] = None) -> None:
    """
    Set a key-value pair in key_value type memory.
    
    Args:
        memory_name: Memory name
        key: Key name
        value: Value to store
        registry: Optional custom registry (defaults to global)
    
    Example:
        await set_key("user_settings", "theme", "dark")
    """
    reg = registry or get_memory_registry()
    handle = reg.get(memory_name, scope_context=_SCOPE_CONTEXT)
    await handle.set_key(key, value)


# Export all public functions
__all__ = [
    "set_scope_context",
    "get_scope_context",
    "read_memory",
    "write_memory",
    "append_memory",
    "set_memory",
    "clear_memory",
    "update_memory",
    "get_key",
    "set_key",
]
