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


@dataclass
class EvalMetricSpec:
    """Specification for a single evaluation metric."""

    name: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuiteDefinition:
    """
    First-class eval_suite block defining a declarative evaluation suite.
    
    Evaluates a target chain over a dataset using specified metrics.
    """

    name: str
    dataset_name: str
    target_chain_name: str
    metrics: List[EvalMetricSpec] = field(default_factory=list)
    judge_llm_name: Optional[str] = None
    rubric: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["Evaluator", "Metric", "Guardrail", "EvalMetricSpec", "EvalSuiteDefinition"]
