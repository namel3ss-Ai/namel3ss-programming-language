"""
Memory store definitions for conversational state management.

Memory provides declarative definitions for managing conversational
context, user history, and session state in AI applications.

This module provides production-grade memory definitions with validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Memory:
    """
    Declarative memory store definition for conversational state.
    
    Memory stores manage contextual information across AI interactions:
    - Session-scoped: Persists for a user session (cleared on logout/timeout)
    - User-scoped: Persists across sessions for a specific user
    - Global: Shared across all users and sessions
    
    Memory types support different data structures:
    - list: Ordered sequence (e.g., conversation history, events)
    - dict: Key-value mappings (e.g., user preferences, facts)
    - vector: Semantic embeddings for similarity search (e.g., long-term memory)
    
    Attributes:
        name: Unique identifier for this memory store
        scope: Scope of persistence ("session", "user", "global")
        kind: Data structure type ("list", "dict", "vector")
        max_items: Maximum number of items to retain (None = unlimited)
        config: Store-specific configuration (TTL, persistence backend, etc.)
        metadata: Additional metadata for introspection and management
        
    Example DSL:
        memory conversation_history {
            scope: session
            kind: list
            max_items: 50
            config: {
                ttl: 3600,  // 1 hour session timeout
                persistence: "redis"
            }
            description: "Recent conversation turns for context"
        }
        
        memory user_preferences {
            scope: user
            kind: dict
            config: {
                persistence: "postgres",
                table: "user_settings"
            }
        }
        
        memory long_term_facts {
            scope: user
            kind: vector
            max_items: 1000
            config: {
                embedding_model: "text-embedding-3-small",
                vector_db: "pinecone",
                similarity_threshold: 0.7
            }
        }
        
        memory shared_knowledge {
            scope: global
            kind: vector
            config: {
                vector_db: "weaviate",
                read_only: true
            }
        }
    
    Validation:
        Use validate_memory() from .validation to ensure configuration
        is valid before runtime initialization.
        
    Notes:
        - Session memory is automatically cleared when sessions expire
        - User memory persists indefinitely unless TTL is configured
        - Global memory should be used sparingly (shared state complexity)
        - Vector stores require embedding models and vector database configuration
        - max_items enforces LRU/FIFO eviction when limit is reached
    """
    name: str
    scope: str = "session"  # session, user, global
    kind: str = "list"  # list, dict, vector
    max_items: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["Memory"]
