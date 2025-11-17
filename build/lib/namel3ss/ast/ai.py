"""AI-centric AST nodes for connectors, templates, and chains."""

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
class Template:
    name: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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


__all__ = ["Connector", "Template", "ChainStep", "Chain"]
