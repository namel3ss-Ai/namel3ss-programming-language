"""Frame-oriented AST nodes for structured data modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .base import Expression
@dataclass
class FrameSourceDef:
    """Describes how to load data for a frame."""

    kind: Literal["sql", "file"]
    connection: Optional[str] = None
    table: Optional[str] = None
    path: Optional[str] = None
    format: Optional[Literal["csv", "parquet"]] = None


@dataclass
class FrameExpression(Expression):
    """Marker base class for frame operation expressions."""

    pass


@dataclass
class FrameRef(FrameExpression):
    name: str


@dataclass
class FrameFilter(FrameExpression):
    source: FrameExpression
    predicate: Expression


@dataclass
class FrameSelect(FrameExpression):
    source: FrameExpression
    columns: List[str]


@dataclass
class FrameOrderBy(FrameExpression):
    source: FrameExpression
    columns: List[str]
    descending: bool = False


@dataclass
class FrameGroupBy(FrameExpression):
    source: FrameExpression
    columns: List[str]


@dataclass
class FrameSummarise(FrameExpression):
    source: FrameExpression
    aggregations: Dict[str, Expression]


@dataclass
class FrameJoin(FrameExpression):
    left: FrameExpression
    right: str
    on: List[str]
    how: Literal["inner", "left", "right", "outer"] = "inner"


@dataclass
class FrameColumnConstraint:
    """Validation rule applied to a column within a frame."""

    name: Optional[str] = None
    expression: Optional[Expression] = None
    message: Optional[str] = None
    severity: Literal["info", "warn", "error"] = "error"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameColumn:
    """Typed column definition for a frame."""

    name: str
    dtype: str = "string"
    nullable: bool = True
    description: Optional[str] = None
    default: Optional[Any] = None
    expression: Optional[Expression] = None
    source: Optional[str] = None
    role: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validations: List[FrameColumnConstraint] = field(default_factory=list)


@dataclass
class FrameIndex:
    """Secondary index metadata for runtime/query planners."""

    name: Optional[str]
    columns: List[str] = field(default_factory=list)
    unique: bool = False
    method: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameRelationship:
    """Describes logical relationships between frames/datasets."""

    name: str
    target_frame: Optional[str] = None
    target_dataset: Optional[str] = None
    local_key: Optional[str] = None
    remote_key: Optional[str] = None
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"] = "many_to_one"
    join_type: Literal["inner", "left", "right", "full"] = "left"
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameConstraint:
    """Row-level constraint enforced on the frame."""

    name: str
    expression: Optional[Expression] = None
    message: Optional[str] = None
    severity: Literal["info", "warn", "error"] = "error"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameAccessPolicy:
    """Access control metadata for exposing frames over APIs."""

    public: bool = False
    roles: List[str] = field(default_factory=list)
    allow_anonymous: bool = False
    rate_limit_per_minute: Optional[int] = None
    cache_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Frame:
    """Top-level strongly typed frame definition."""

    name: str
    source_type: str = "dataset"
    source: Optional[str] = None
    description: Optional[str] = None
    columns: List[FrameColumn] = field(default_factory=list)
    indexes: List[FrameIndex] = field(default_factory=list)
    relationships: List[FrameRelationship] = field(default_factory=list)
    constraints: List[FrameConstraint] = field(default_factory=list)
    access: Optional[FrameAccessPolicy] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    key: List[str] = field(default_factory=list)
    splits: Dict[str, float] = field(default_factory=dict)
    source_config: Optional[FrameSourceDef] = None


__all__ = [
    "Frame",
    "FrameColumn",
    "FrameColumnConstraint",
    "FrameConstraint",
    "FrameIndex",
    "FrameRelationship",
    "FrameAccessPolicy",
    "FrameSourceDef",
    "FrameExpression",
    "FrameRef",
    "FrameFilter",
    "FrameSelect",
    "FrameOrderBy",
    "FrameGroupBy",
    "FrameSummarise",
    "FrameJoin",
]
