"""Insight AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .base import Expression
from .datasets import DatasetTransformStep


@dataclass
class InsightDatasetRef:
    """Represents a dataset dependency for an insight."""

    name: str
    role: Literal["source", "context", "comparison", "reference"] = "source"
    transforms: List[DatasetTransformStep] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightAudience:
    """Target audience metadata for an insight."""

    name: str
    persona: Optional[str] = None
    needs: Dict[str, Any] = field(default_factory=dict)
    channels: List[str] = field(default_factory=list)


@dataclass
class InsightDeliveryChannel:
    """Delivery channel definition for broadcasting an insight."""

    kind: Literal["dashboard", "email", "slack", "webhook", "alert", "custom"] = "dashboard"
    target: Optional[str] = None
    schedule: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightLogicStep:
    """Base class for a single instruction in an insight."""

    pass


@dataclass
class InsightAssignment(InsightLogicStep):
    name: str
    expression: Expression


@dataclass
class InsightSelect(InsightLogicStep):
    dataset: str
    condition: Optional[Expression] = None
    limit: Optional[int] = None
    order_by: Optional[List[str]] = None


@dataclass
class InsightEmit(InsightLogicStep):
    kind: Literal["narrative", "alert", "metric", "table"] = "narrative"
    content: str = ""
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightMetric:
    name: str
    value: Expression
    label: Optional[str] = None
    format: Optional[str] = None
    unit: Optional[str] = None
    baseline: Optional[Expression] = None
    target: Optional[Expression] = None
    window: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightThreshold:
    name: str
    metric: str
    operator: str = ">="
    value: Optional[Expression] = None
    level: str = "warning"
    message: Optional[str] = None
    window: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightNarrative:
    name: str
    template: str
    variant: Optional[str] = None
    style: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Insight:
    name: str
    source_dataset: str
    logic: List[InsightLogicStep] = field(default_factory=list)
    expose_as: Dict[str, Expression] = field(default_factory=dict)
    metrics: List[InsightMetric] = field(default_factory=list)
    thresholds: List[InsightThreshold] = field(default_factory=list)
    narratives: List[InsightNarrative] = field(default_factory=list)
    alert_thresholds: Dict[str, Any] = field(default_factory=dict)
    datasets: List[InsightDatasetRef] = field(default_factory=list)
    parameters: Dict[str, Expression] = field(default_factory=dict)
    audiences: List[InsightAudience] = field(default_factory=list)
    channels: List[InsightDeliveryChannel] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "InsightDatasetRef",
    "InsightAudience",
    "InsightDeliveryChannel",
    "InsightLogicStep",
    "InsightAssignment",
    "InsightSelect",
    "InsightEmit",
    "InsightMetric",
    "InsightThreshold",
    "InsightNarrative",
    "Insight",
]
