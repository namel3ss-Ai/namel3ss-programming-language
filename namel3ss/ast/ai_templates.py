"""
Template definitions for reusable prompt templates.

Templates provide a way to define named, reusable prompt strings
that can be referenced throughout the N3 program.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Template:
    """
    Reusable prompt template definition.
    
    Templates allow defining named prompt strings that can be
    referenced by prompts and chains, promoting reusability.
    
    Example DSL:
        define template summarize_text {
            prompt: "Summarize the following text concisely: {{text}}"
            metadata: {
                category: "summarization"
            }
        }
    """
    name: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["Template"]
