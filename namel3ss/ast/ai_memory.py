"""
Memory store definitions for conversational state management.

Memory provides declarative definitions for managing conversational
context, user history, and session state in AI applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Memory:
    """
    Declarative memory store definition for conversational state.
    
    Memory stores manage contextual information across interactions:
    - Session-scoped: Persists for a user session
    - User-scoped: Persists across sessions for a user
    - Global: Shared across all users
    
    Example DSL:
        memory conversation_history {
            scope: session
            kind: list
            max_items: 50
            config: {
                ttl: 3600
            }
        }
    """
    name: str
    scope: str = "session"  # session, user, global
    kind: str = "list"  # list, dict, vector
    max_items: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["Memory"]
