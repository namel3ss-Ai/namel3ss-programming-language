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
    rows: List[Dict[str, Any]] = Field(default_factory=list)

class ChartSeries(BaseModel):
    label: str
    data: List[Any] = Field(default_factory=list)

class ChartResponse(BaseModel):
    title: str
    source: Dict[str, str]
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    labels: List[Any] = Field(default_factory=list)
    series: List[ChartSeries] = Field(default_factory=list)

class ActionResponse(BaseModel):
    action: str
    trigger: str
    operations: List[str] = Field(default_factory=list)

class FormResponse(BaseModel):
    title: str
    fields: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    operations: List[str] = Field(default_factory=list)
