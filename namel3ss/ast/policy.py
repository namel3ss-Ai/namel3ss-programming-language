"""Safety policy AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PolicyDefinition:
    """Safety policy definition.
    
    Defines safety and guardrail rules that can be attached to chains,
    agents, or graphs for runtime enforcement.
    
    Example DSL:
        policy safety {
            block_categories: ["self-harm", "hate", "sexual_minors"]
            redact_pii: true
            max_tokens: 512
            fallback_message: "I can't help with that."
        }
    """
    name: str
    block_categories: List[str] = field(default_factory=list)
    allow_categories: List[str] = field(default_factory=list)
    alert_only_categories: List[str] = field(default_factory=list)
    redact_pii: bool = False
    max_tokens: Optional[int] = None
    fallback_message: Optional[str] = None
    log_level: str = "full"  # full, minimal, none
    config: Dict[str, Any] = field(default_factory=dict)  # Additional config
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["PolicyDefinition"]
