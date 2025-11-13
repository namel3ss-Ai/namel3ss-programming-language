"""Machine learning model AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .base import Expression
from .datasets import DatasetTransformStep


@dataclass
class ModelFeatureSpec:
    name: str
    dtype: Optional[str] = None
    role: Literal["feature", "target", "weight", "metadata"] = "feature"
    source: Optional[str] = None
    expression: Optional[Expression] = None
    required: bool = True
    description: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDatasetReference:
    role: Literal["train", "validation", "test", "inference", "feature_store"] = "train"
    name: str = ""
    filters: List[Expression] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelHyperParameter:
    name: str
    value: Any
    tunable: bool = False
    search_space: Optional[Dict[str, Any]] = None


@dataclass
class ModelEvaluationMetric:
    name: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    goal: Optional[str] = None
    higher_is_better: bool = True
    dataset: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMonitoringSpec:
    schedule: Optional[str] = None
    metrics: List[ModelEvaluationMetric] = field(default_factory=list)
    alerts: Dict[str, Any] = field(default_factory=dict)
    drift_thresholds: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelServingSpec:
    endpoints: List[str] = field(default_factory=list)
    realtime: Dict[str, Any] = field(default_factory=dict)
    batch: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDeploymentTarget:
    name: str
    environment: Optional[str] = None
    strategy: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTrainingSpec:
    source_type: Literal["dataset", "table", "file", "sql", "rest"] = "dataset"
    source: Optional[str] = None
    target: Optional[str] = None
    features: List[str] = field(default_factory=list)
    split: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None
    framework: Optional[str] = None
    objective: Optional[str] = None
    loss: Optional[str] = None
    optimizer: Optional[str] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    hyperparameters: List[ModelHyperParameter] = field(default_factory=list)
    datasets: List[ModelDatasetReference] = field(default_factory=list)
    transforms: List[DatasetTransformStep] = field(default_factory=list)
    callbacks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRegistryInfo:
    version: Optional[str] = None
    accuracy: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registry_name: Optional[str] = None
    owner: Optional[str] = None
    stage: Optional[str] = None
    last_updated: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    name: str
    model_type: str
    engine: Optional[str] = None
    task: Optional[str] = None
    description: Optional[str] = None
    training: ModelTrainingSpec = field(default_factory=ModelTrainingSpec)
    features_spec: List[ModelFeatureSpec] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    registry: ModelRegistryInfo = field(default_factory=ModelRegistryInfo)
    monitoring: Optional[ModelMonitoringSpec] = None
    serving: ModelServingSpec = field(default_factory=ModelServingSpec)
    deployments: List[ModelDeploymentTarget] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceTarget:
    kind: Literal["variable", "dataset", "insight", "component"] = "variable"
    name: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, Any] = field(default_factory=dict)
    preprocessing: List[DatasetTransformStep] = field(default_factory=list)


__all__ = [
    "ModelFeatureSpec",
    "ModelDatasetReference",
    "ModelHyperParameter",
    "ModelEvaluationMetric",
    "ModelMonitoringSpec",
    "ModelServingSpec",
    "ModelDeploymentTarget",
    "ModelTrainingSpec",
    "ModelRegistryInfo",
    "Model",
    "InferenceTarget",
]
