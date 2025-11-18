"""AST nodes for evaluation and safety declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evaluator:
    """Evaluation or safety tool definition."""

    name: str
    kind: str
    provider: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric derived from evaluator outputs."""

    name: str
    evaluator: str
    aggregation: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Guardrail:
    """Guardrail policy referencing evaluators."""

    name: str
    evaluators: List[str] = field(default_factory=list)
    action: str = ""
    message: Optional[str] = None


__all__ = ["Evaluator", "Metric", "Guardrail"]
