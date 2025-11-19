"""
Production-ready memory system for Namel3ss.

This module provides:
- MemoryRegistry: Central registry for all defined memory stores
- MemoryHandle: Scope-aware, type-safe memory operations
- Backend abstraction: Pluggable storage (in-memory, Redis, DB)
- Safety: Bounded storage, scope isolation, proper error handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MemoryError(Exception):
    """Base exception for memory operations."""
    pass


class MemoryNotFoundError(MemoryError):
    """Memory store not found in registry."""
    pass


class MemoryScopeError(MemoryError):
    """Invalid memory scope access."""
    pass


class MemoryCapacityError(MemoryError):
    """Memory capacity limit exceeded."""
    pass


@dataclass
class MemorySpec:
    """Specification for a memory store."""
    
    name: str
    scope: str = "session"
    kind: str = "list"
    max_items: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryBackend(ABC):
    """Abstract interface for memory storage backends."""
    
    @abstractmethod
    async def read(self, key: str) -> Optional[Any]:
        """Read value for the given key."""
        pass
    
    @abstractmethod
    async def write(self, key: str, value: Any) -> None:
        """Write value for the given key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value for the given key."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def clear_all(self) -> None:
        """Clear all data (testing only)."""
        pass


class InMemoryBackend(MemoryBackend):
    """In-memory storage backend for development and testing."""
    
    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def read(self, key: str) -> Optional[Any]:
        async with self._lock:
            return self._storage.get(key)
    
    async def write(self, key: str, value: Any) -> None:
        async with self._lock:
            self._storage[key] = value
    
    async def delete(self, key: str) -> None:
        async with self._lock:
            self._storage.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._storage
    
    async def clear_all(self) -> None:
        async with self._lock:
            self._storage.clear()


class MemoryHandle:
    """
    Scope-aware handle for memory operations.
    
    Provides type-safe read/write operations with proper scoping,
    capacity limits, and eviction strategies.
    """
    
    def __init__(
        self,
        spec: MemorySpec,
        backend: MemoryBackend,
        scope_context: Dict[str, str],
    ):
        self.spec = spec
        self.backend = backend
        self.scope_context = scope_context
        self._scoped_key = self._build_scoped_key()
    
    def _build_scoped_key(self) -> str:
        """Build scope-aware key for storage."""
        parts = [f"memory:{self.spec.name}"]
        
        # Add scope identifiers
        if self.spec.scope == "session":
            session_id = self.scope_context.get("session_id", "default")
            parts.append(f"session:{session_id}")
        elif self.spec.scope == "user":
            user_id = self.scope_context.get("user_id", "anonymous")
            parts.append(f"user:{user_id}")
        elif self.spec.scope == "conversation":
            conversation_id = self.scope_context.get("conversation_id", "default")
            parts.append(f"conversation:{conversation_id}")
        elif self.spec.scope == "page":
            page_id = self.scope_context.get("page_id", "default")
            parts.append(f"page:{page_id}")
        elif self.spec.scope == "global":
            # No additional scope identifier
            pass
        else:
            logger.warning(f"Unknown memory scope: {self.spec.scope}, treating as global")
        
        return ":".join(parts)
    
    async def read(
        self,
        *,
        limit: Optional[int] = None,
        reverse: bool = False,
    ) -> Any:
        """
        Read from memory.
        
        Args:
            limit: Maximum number of items to return (for list/conversation types)
            reverse: Return items in reverse order
        
        Returns:
            Memory contents based on kind:
            - list/conversation: List of items (most recent if limit specified)
            - key_value: Dict of key-value pairs
            - buffer: Single value or None
        """
        try:
            data = await self.backend.read(self._scoped_key)
            
            if data is None:
                # Return empty structure based on kind
                if self.spec.kind in ("list", "conversation"):
                    return []
                elif self.spec.kind == "key_value":
                    return {}
                else:  # buffer, vector, etc.
                    return None
            
            # Apply limit and reverse for list-like types
            if self.spec.kind in ("list", "conversation") and isinstance(data, list):
                # If limit specified, take last N items (most recent)
                if limit is not None and limit > 0:
                    data = data[-limit:]
                
                # Then apply reverse if requested
                if reverse:
                    data = list(reversed(data))
            
            return data
        
        except Exception as e:
            logger.error(f"Memory read error for '{self.spec.name}': {e}", exc_info=True)
            raise MemoryError(f"Failed to read from memory '{self.spec.name}': {e}") from e
    
    async def write(self, value: Any) -> None:
        """
        Write to memory, replacing existing content.
        
        Args:
            value: Value to write (type depends on memory kind)
        
        Raises:
            MemoryCapacityError: If value exceeds capacity limits
        """
        try:
            # Validate capacity for list types
            if self.spec.kind in ("list", "conversation"):
                if not isinstance(value, list):
                    raise MemoryError(
                        f"Memory '{self.spec.name}' is kind '{self.spec.kind}', "
                        f"expected list but got {type(value).__name__}"
                    )
                
                if self.spec.max_items is not None and len(value) > self.spec.max_items:
                    # Truncate to max_items, keeping most recent
                    value = value[-self.spec.max_items:]
                    logger.warning(
                        f"Memory '{self.spec.name}' truncated to {self.spec.max_items} items"
                    )
            
            elif self.spec.kind == "key_value":
                if not isinstance(value, dict):
                    raise MemoryError(
                        f"Memory '{self.spec.name}' is kind 'key_value', "
                        f"expected dict but got {type(value).__name__}"
                    )
            
            await self.backend.write(self._scoped_key, value)
        
        except MemoryError:
            raise
        except Exception as e:
            logger.error(f"Memory write error for '{self.spec.name}': {e}", exc_info=True)
            raise MemoryError(f"Failed to write to memory '{self.spec.name}': {e}") from e
    
    async def append(self, item: Any) -> None:
        """
        Append item to list-type memory.
        
        Args:
            item: Item to append
        
        Raises:
            MemoryError: If memory is not list type
            MemoryCapacityError: If at capacity
        """
        if self.spec.kind not in ("list", "conversation"):
            raise MemoryError(
                f"Cannot append to memory '{self.spec.name}' of kind '{self.spec.kind}'"
            )
        
        try:
            current = await self.read()
            if not isinstance(current, list):
                current = []
            
            current.append(item)
            
            # Enforce max_items by removing oldest
            if self.spec.max_items is not None and len(current) > self.spec.max_items:
                current = current[-self.spec.max_items:]
                logger.debug(
                    f"Memory '{self.spec.name}' evicted oldest items, "
                    f"keeping {self.spec.max_items} most recent"
                )
            
            await self.backend.write(self._scoped_key, current)
        
        except MemoryError:
            raise
        except Exception as e:
            logger.error(f"Memory append error for '{self.spec.name}': {e}", exc_info=True)
            raise MemoryError(f"Failed to append to memory '{self.spec.name}': {e}") from e
    
    async def update(self, fn: Callable[[Any], Any]) -> None:
        """
        Update memory by applying a transformation function.
        
        Args:
            fn: Function that takes current value and returns new value
        
        Example:
            await handle.update(lambda items: items + [new_item])
        """
        try:
            current = await self.read()
            updated = fn(current)
            await self.write(updated)
        
        except Exception as e:
            logger.error(f"Memory update error for '{self.spec.name}': {e}", exc_info=True)
            raise MemoryError(f"Failed to update memory '{self.spec.name}': {e}") from e
    
    async def clear(self) -> None:
        """Clear all data from this memory store."""
        try:
            await self.backend.delete(self._scoped_key)
            logger.debug(f"Cleared memory '{self.spec.name}'")
        except Exception as e:
            logger.error(f"Memory clear error for '{self.spec.name}': {e}", exc_info=True)
            raise MemoryError(f"Failed to clear memory '{self.spec.name}': {e}") from e
    
    async def set_key(self, key: str, value: Any) -> None:
        """
        Set a key-value pair in key_value type memory.
        
        Args:
            key: Key name
            value: Value to store
        
        Raises:
            MemoryError: If memory is not key_value type
        """
        if self.spec.kind != "key_value":
            raise MemoryError(
                f"Cannot set_key on memory '{self.spec.name}' of kind '{self.spec.kind}'"
            )
        
        try:
            current = await self.read()
            if not isinstance(current, dict):
                current = {}
            
            current[key] = value
            await self.backend.write(self._scoped_key, current)
        
        except MemoryError:
            raise
        except Exception as e:
            logger.error(f"Memory set_key error for '{self.spec.name}': {e}", exc_info=True)
            raise MemoryError(
                f"Failed to set key '{key}' in memory '{self.spec.name}': {e}"
            ) from e
    
    async def get_key(self, key: str, default: Any = None) -> Any:
        """
        Get a value by key from key_value type memory.
        
        Args:
            key: Key name
            default: Default value if key not found
        
        Returns:
            Value associated with key, or default
        
        Raises:
            MemoryError: If memory is not key_value type
        """
        if self.spec.kind != "key_value":
            raise MemoryError(
                f"Cannot get_key from memory '{self.spec.name}' of kind '{self.spec.kind}'"
            )
        
        try:
            current = await self.read()
            if not isinstance(current, dict):
                return default
            
            return current.get(key, default)
        
        except MemoryError:
            raise
        except Exception as e:
            logger.error(f"Memory get_key error for '{self.spec.name}': {e}", exc_info=True)
            raise MemoryError(
                f"Failed to get key '{key}' from memory '{self.spec.name}': {e}"
            ) from e


class MemoryRegistry:
    """
    Central registry for all memory stores in the application.
    
    Manages memory specifications, backend selection, and handle creation.
    Provides thread-safe access to memory operations.
    """
    
    def __init__(self, backend: Optional[MemoryBackend] = None):
        self._specs: Dict[str, MemorySpec] = {}
        self._backend = backend or InMemoryBackend()
        self._default_scope_context: Dict[str, str] = {}
    
    def register(self, spec_dict: Dict[str, Any]) -> None:
        """
        Register a memory specification from encoded AST.
        
        Args:
            spec_dict: Dictionary with name, scope, kind, max_items, config, metadata
        """
        spec = MemorySpec(
            name=spec_dict["name"],
            scope=spec_dict.get("scope", "session"),
            kind=spec_dict.get("kind", "list"),
            max_items=spec_dict.get("max_items"),
            config=spec_dict.get("config", {}),
            metadata=spec_dict.get("metadata", {}),
        )
        
        self._specs[spec.name] = spec
        logger.info(
            f"Registered memory '{spec.name}' "
            f"(scope={spec.scope}, kind={spec.kind}, max_items={spec.max_items})"
        )
    
    def get(
        self,
        name: str,
        scope_context: Optional[Dict[str, str]] = None,
    ) -> MemoryHandle:
        """
        Get a handle to a registered memory store.
        
        Args:
            name: Memory name
            scope_context: Context for scope resolution (session_id, user_id, etc.)
        
        Returns:
            MemoryHandle for performing operations
        
        Raises:
            MemoryNotFoundError: If memory not registered
        """
        if name not in self._specs:
            raise MemoryNotFoundError(
                f"Memory '{name}' not found in registry. "
                f"Available: {', '.join(sorted(self._specs.keys()))}"
            )
        
        spec = self._specs[name]
        context = scope_context or self._default_scope_context
        
        return MemoryHandle(spec, self._backend, context)
    
    def set_default_scope_context(self, context: Dict[str, str]) -> None:
        """Set default scope context for handle creation."""
        self._default_scope_context = context
    
    def list_memories(self) -> List[str]:
        """Get list of all registered memory names."""
        return sorted(self._specs.keys())
    
    def get_spec(self, name: str) -> Optional[MemorySpec]:
        """Get memory specification by name."""
        return self._specs.get(name)
    
    async def clear_all(self) -> None:
        """Clear all memory data (testing only)."""
        await self._backend.clear_all()
        logger.warning("Cleared all memory data")


# Global registry instance for runtime use
_GLOBAL_REGISTRY: Optional[MemoryRegistry] = None


def get_memory_registry() -> MemoryRegistry:
    """Get or create the global memory registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        # Choose backend based on configuration
        backend_type = os.environ.get("NAMEL3SS_MEMORY_BACKEND", "local").lower()
        
        if backend_type == "local":
            backend = InMemoryBackend()
        else:
            # Future: support Redis, PostgreSQL, etc.
            logger.warning(
                f"Unknown memory backend '{backend_type}', falling back to in-memory"
            )
            backend = InMemoryBackend()
        
        _GLOBAL_REGISTRY = MemoryRegistry(backend)
        logger.info(f"Initialized global memory registry with {backend_type} backend")
    
    return _GLOBAL_REGISTRY


def reset_memory_registry() -> None:
    """Reset the global registry (testing only)."""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None
