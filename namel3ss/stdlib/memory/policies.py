"""Memory policy definitions for the Namel3ss standard library."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MemoryPolicy(Enum):
    """Standard memory policies for agent conversation management."""
    
    NONE = "none"
    """No conversation history is maintained. Each interaction starts fresh."""
    
    CONVERSATION_WINDOW = "conversation_window"  
    """Maintains a sliding window of recent messages. Oldest messages are discarded."""
    
    FULL_HISTORY = "full_history"
    """Maintains complete conversation history with optional limits."""
    
    SUMMARY = "summary"
    """Incremental summarization of older messages with recent window."""


@dataclass(frozen=True)
class MemoryPolicySpec:
    """Specification for a memory policy's behavior and constraints."""
    
    policy: MemoryPolicy
    description: str
    supports_max_items: bool = False
    supports_window_size: bool = False 
    supports_summarization: bool = False
    default_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_config is None:
            object.__setattr__(self, 'default_config', {})


# Standard memory policy specifications
STANDARD_MEMORY_POLICIES: Dict[MemoryPolicy, MemoryPolicySpec] = {
    MemoryPolicy.NONE: MemoryPolicySpec(
        policy=MemoryPolicy.NONE,
        description="No message history is maintained. Each interaction starts fresh.",
        default_config={}
    ),
    
    MemoryPolicy.CONVERSATION_WINDOW: MemoryPolicySpec(
        policy=MemoryPolicy.CONVERSATION_WINDOW,
        description="Maintains a sliding window of recent messages. Oldest messages are discarded.",
        supports_max_items=True,
        supports_window_size=True,
        default_config={
            "window_size": 10,
            "max_items": 20
        }
    ),
    
    MemoryPolicy.FULL_HISTORY: MemoryPolicySpec(
        policy=MemoryPolicy.FULL_HISTORY,
        description="Maintains complete conversation history with optional limits.",
        supports_max_items=True,
        default_config={
            "max_items": 1000
        }
    ),
    
    MemoryPolicy.SUMMARY: MemoryPolicySpec(
        policy=MemoryPolicy.SUMMARY,
        description="Incremental summarization of older messages with recent window.",
        supports_max_items=True,
        supports_summarization=True,
        default_config={
            "max_summary_tokens": 512,
            "summary_trigger_messages": 20,
            "summary_trigger_tokens": 4000,
            "summary_recent_window": 5,
            "max_items": 100
        }
    )
}


def get_memory_policy_spec(policy: Union[str, MemoryPolicy]) -> MemoryPolicySpec:
    """
    Get the specification for a memory policy.
    
    Args:
        policy: Memory policy name or enum value
        
    Returns:
        Memory policy specification
        
    Raises:
        ValueError: If policy is not recognized
    """
    if isinstance(policy, str):
        try:
            policy = MemoryPolicy(policy)
        except ValueError:
            valid_policies = [p.value for p in MemoryPolicy]
            raise ValueError(
                f"Unknown memory policy '{policy}'. "
                f"Valid policies: {', '.join(valid_policies)}"
            )
    
    return STANDARD_MEMORY_POLICIES[policy]


def list_memory_policies() -> List[str]:
    """List all available memory policy names."""
    return [policy.value for policy in MemoryPolicy]


def get_policy_description(policy: Union[str, MemoryPolicy]) -> str:
    """Get human-readable description of a memory policy."""
    spec = get_memory_policy_spec(policy)
    return spec.description


def get_default_config(policy: Union[str, MemoryPolicy]) -> Dict[str, Any]:
    """Get default configuration for a memory policy."""
    spec = get_memory_policy_spec(policy)
    return spec.default_config.copy()