"""
AI model definitions for provider-backed language models.

AIModel provides a declarative handle for referencing external AI models
from providers like OpenAI, Anthropic, Google, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AIModel:
    """
    Declarative handle for a provider-backed AI model.
    
    Represents a reference to an external AI model that can be used
    in prompts, chains, and other AI-driven constructs.
    
    Example DSL:
        ai model gpt4 {
            provider: openai
            model: gpt-4-turbo
            config: {
                temperature: 0.7,
                max_tokens: 2048
            }
        }
    """
    name: str
    provider: str
    model_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["AIModel"]
