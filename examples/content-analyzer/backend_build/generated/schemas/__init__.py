"""Pydantic schemas for the generated FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RuntimeErrorPayload(BaseModel):
    code: str
    message: str
    scope: Optional[str] = None
    source: Optional[str] = None
    detail: Optional[str] = None
    severity: str = Field(default="error")


class TableResponse(BaseModel):
    title: str
    source: Dict[str, str]
    columns: List[str] = Field(default_factory=list)
    filter: Optional[str] = None
    sort: Optional[str] = None
    style: Dict[str, Any] = Field(default_factory=dict)
    insight: Optional[str] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    insights: Dict[str, Any] = Field(default_factory=dict)
    errors: List[RuntimeErrorPayload] = Field(default_factory=list)


class ChartSeries(BaseModel):
    label: str
    data: List[Any] = Field(default_factory=list)


class ChartResponse(BaseModel):
    heading: Optional[str] = None
    title: Optional[str] = None
    source: Dict[str, str]
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    labels: List[Any] = Field(default_factory=list)
    series: List[ChartSeries] = Field(default_factory=list)
    legend: Dict[str, Any] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    encodings: Dict[str, Any] = Field(default_factory=dict)
    insight: Optional[str] = None
    insights: Dict[str, Any] = Field(default_factory=dict)
    errors: List[RuntimeErrorPayload] = Field(default_factory=list)


class FrameColumnValidationSchema(BaseModel):
    name: Optional[str] = None
    expression: Optional[str] = None
    message: Optional[str] = None
    severity: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class FrameColumnSchema(BaseModel):
    name: str
    dtype: Optional[str] = None
    nullable: bool = True
    description: Optional[str] = None
    default: Any = None
    expression: Optional[str] = None
    expression_expr: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    validations: List[FrameColumnValidationSchema] = Field(default_factory=list)


class FrameSchemaPayload(BaseModel):
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    columns: List[FrameColumnSchema] = Field(default_factory=list)
    indexes: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    access: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class FrameResponse(BaseModel):
    name: str
    source: Dict[str, Any]
    schema: FrameSchemaPayload
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    limit: int = 0
    offset: int = 0
    order_by: Optional[str] = None
    errors: List[RuntimeErrorPayload] = Field(default_factory=list)


class FrameErrorResponse(BaseModel):
    status_code: int
    error: str
    detail: Optional[str] = None


class InsightResponse(BaseModel):
    name: str
    dataset: str
    result: Dict[str, Any]


class ActionResponse(BaseModel):
    action: str
    trigger: str
    operations: List[str] = Field(default_factory=list)
    errors: List[RuntimeErrorPayload] = Field(default_factory=list)


class FormResponse(BaseModel):
    title: str
    fields: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    operations: List[str] = Field(default_factory=list)
    errors: List[RuntimeErrorPayload] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    model: str
    version: Optional[str] = None
    framework: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    explanations: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentVariant(BaseModel):
    name: Optional[str] = None
    target_type: str
    target_name: Optional[str] = None
    score: Optional[float] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class ExperimentMetric(BaseModel):
    name: Optional[str] = None
    value: Optional[float] = None
    source_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    name: Optional[str] = None
    variant: Optional[ExperimentVariant] = None
    metrics: List[ExperimentMetric] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrudResourceMeta(BaseModel):
    slug: str
    label: Optional[str] = None
    source_type: str
    primary_key: str
    tenant_column: Optional[str] = None
    operations: List[str] = Field(default_factory=list)
    default_limit: int = 0
    max_limit: int = 0
    read_only: bool = False


class CrudCatalogResponse(BaseModel):
    resources: List[CrudResourceMeta] = Field(default_factory=list)


class CrudBaseResponse(BaseModel):
    resource: str
    label: Optional[str] = None
    status: str = Field(default="ok")
    errors: List[RuntimeErrorPayload] = Field(default_factory=list)


class CrudListResponse(CrudBaseResponse):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    limit: int = 0
    offset: int = 0
    total: Optional[int] = None


class CrudItemResponse(CrudBaseResponse):
    item: Optional[Dict[str, Any]] = None


class CrudDeleteResponse(CrudBaseResponse):
    deleted: bool = False