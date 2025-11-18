"""AST nodes for experiment declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentVariant:
    name: str
    target_type: str
    target_name: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentMetric:
    name: str
    source_kind: Optional[str] = None
    source_name: Optional[str] = None
    goal: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentComparison:
    baseline_model: Optional[str] = None
    best_of: Optional[str] = None
    challengers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    name: str
    description: Optional[str] = None
    variants: List[ExperimentVariant] = field(default_factory=list)
    metrics: List[ExperimentMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    training_jobs: List[str] = field(default_factory=list)
    tuning_jobs: List[str] = field(default_factory=list)
    eval_datasets: List[str] = field(default_factory=list)
    eval_metrics: List[str] = field(default_factory=list)
    comparison: Optional[ExperimentComparison] = None


__all__ = [
    "Experiment",
    "ExperimentVariant",
    "ExperimentMetric",
    "ExperimentComparison",
]
