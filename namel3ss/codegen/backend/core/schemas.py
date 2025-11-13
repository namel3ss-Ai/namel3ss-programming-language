"""Render the generated Pydantic schemas."""

from __future__ import annotations

import textwrap

__all__ = ["_render_schemas_module"]


def _render_schemas_module() -> str:
    template = '''
"""Pydantic schemas for the generated FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class InsightResponse(BaseModel):
    name: str
    dataset: str
    result: Dict[str, Any]


class ActionResponse(BaseModel):
    action: str
    trigger: str
    operations: List[str] = Field(default_factory=list)


class FormResponse(BaseModel):
    title: str
    fields: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    operations: List[str] = Field(default_factory=list)


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

'''
    return textwrap.dedent(template).strip()
