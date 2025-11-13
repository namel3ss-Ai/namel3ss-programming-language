"""Dataset-related AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .base import Expression


@dataclass
class DatasetOp:
    pass


@dataclass
class FilterOp(DatasetOp):
    condition: Expression


@dataclass
class GroupByOp(DatasetOp):
    columns: List[str]


@dataclass
class AggregateOp(DatasetOp):
    function: str
    expression: str


@dataclass
class OrderByOp(DatasetOp):
    columns: List[str]


@dataclass
class ComputedColumnOp(DatasetOp):
    name: str
    expression: Expression


@dataclass
class WindowFrame:
    mode: Literal["rolling", "expanding", "cumulative"] = "rolling"
    interval_value: Optional[int] = None
    interval_unit: Optional[str] = None


@dataclass
class WindowOp(DatasetOp):
    name: str
    function: str
    target: Optional[str] = None
    partition_by: Optional[List[str]] = None
    order_by: Optional[List[str]] = None
    frame: WindowFrame = field(default_factory=WindowFrame)


@dataclass
class JoinOp(DatasetOp):
    target_type: Literal["dataset", "table", "sql", "rest", "file"]
    target_name: str
    condition: Optional[Expression] = None
    join_type: Literal["inner", "left", "right", "full"] = "inner"


@dataclass
class DatasetTransformStep(DatasetOp):
    """Represents a named transformation within a dataset pipeline."""

    name: str
    transform_type: str
    inputs: List[str] = field(default_factory=list)
    output: Optional[str] = None
    expression: Optional[Expression] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSchemaField:
    """Schema metadata for a dataset column."""

    name: str
    dtype: str
    nullable: bool = True
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetFeature:
    """Feature engineering output tracked on the dataset."""

    name: str
    source: Optional[str] = None
    role: Literal["feature", "target", "metadata", "embedding"] = "feature"
    dtype: Optional[str] = None
    expression: Optional[Expression] = None
    description: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetTarget:
    """Represents a modelling target derived from the dataset."""

    name: str
    kind: Literal["classification", "regression", "ranking", "forecast", "custom"] = "classification"
    expression: Optional[Expression] = None
    positive_class: Optional[str] = None
    horizon: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetQualityCheck:
    """Data quality rule that should hold on the dataset."""

    name: str
    condition: Optional[Expression] = None
    metric: Optional[str] = None
    threshold: Optional[float] = None
    severity: Literal["info", "warn", "error"] = "error"
    message: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """Lightweight profile describing dataset statistics."""

    row_count: Optional[int] = None
    column_count: Optional[int] = None
    freshness: Optional[str] = None
    updated_at: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConnectorConfig:
    connector_type: str
    connector_name: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CachePolicy:
    strategy: str = "none"
    ttl_seconds: Optional[int] = None
    max_entries: Optional[int] = None


@dataclass
class PaginationPolicy:
    enabled: bool = True
    page_size: Optional[int] = None
    max_pages: Optional[int] = None


@dataclass
class StreamingPolicy:
    enabled: bool = True
    chunk_size: Optional[int] = None


@dataclass
class RefreshPolicy:
    interval_seconds: int
    mode: str = "polling"


@dataclass
class Dataset:
    name: str
    source_type: str
    source: str
    connector: Optional[DatasetConnectorConfig] = None
    operations: List[DatasetOp] = field(default_factory=list)
    transforms: List[DatasetTransformStep] = field(default_factory=list)
    schema: List[DatasetSchemaField] = field(default_factory=list)
    features: List[DatasetFeature] = field(default_factory=list)
    targets: List[DatasetTarget] = field(default_factory=list)
    quality_checks: List[DatasetQualityCheck] = field(default_factory=list)
    profile: Optional[DatasetProfile] = None
    reactive: bool = False
    refresh_policy: Optional[RefreshPolicy] = None
    cache_policy: Optional[CachePolicy] = None
    pagination: Optional[PaginationPolicy] = None
    streaming: Optional[StreamingPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


__all__ = [
    "DatasetOp",
    "FilterOp",
    "GroupByOp",
    "AggregateOp",
    "OrderByOp",
    "ComputedColumnOp",
    "WindowFrame",
    "WindowOp",
    "JoinOp",
    "DatasetTransformStep",
    "DatasetSchemaField",
    "DatasetFeature",
    "DatasetTarget",
    "DatasetQualityCheck",
    "DatasetProfile",
    "DatasetConnectorConfig",
    "CachePolicy",
    "PaginationPolicy",
    "StreamingPolicy",
    "RefreshPolicy",
    "Dataset",
]
