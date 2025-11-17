"""AI-centric AST nodes for connectors, templates, prompts, and chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Connector:
    name: str
    connector_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class AIModel:
    """Declarative handle for a provider-backed AI model."""

    name: str
    provider: str
    model_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Template:
    name: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptField:
    """Structured schema information for prompt inputs/outputs."""

    name: str
    field_type: str = "text"
    required: bool = True
    description: Optional[str] = None
    default: Optional[Any] = None
    enum: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prompt:
    """A named, reusable prompt with typed inputs and outputs."""

    name: str
    model: str
    template: str
    input_fields: List[PromptField] = field(default_factory=list)
    output_fields: List[PromptField] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class ChainStep:
    kind: str
    target: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chain:
    name: str
    input_key: str = "input"
    steps: List[ChainStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Connector",
    "AIModel",
    "Template",
    "PromptField",
    "Prompt",
    "ChainStep",
    "Chain",
]
