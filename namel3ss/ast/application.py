"""Application-level AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Theme
from .ai import Chain, Connector, Template
from .experiments import Experiment
from .datasets import Dataset, RefreshPolicy
from .insights import Insight
from .models import Model
from .pages import PageStatement, VariableAssignment
from .crud import CrudResource


@dataclass
class Page:
    name: str
    route: str
    statements: List[PageStatement] = field(default_factory=list)
    reactive: bool = False
    refresh_policy: Optional[RefreshPolicy] = None
    layout: Dict[str, Any] = field(default_factory=dict)


@dataclass
class App:
    name: str
    database: Optional[str] = None
    theme: Theme = field(default_factory=Theme)
    variables: List[VariableAssignment] = field(default_factory=list)
    datasets: List[Dataset] = field(default_factory=list)
    pages: List[Page] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    models: List[Model] = field(default_factory=list)
    connectors: List[Connector] = field(default_factory=list)
    templates: List[Template] = field(default_factory=list)
    chains: List[Chain] = field(default_factory=list)
    experiments: List[Experiment] = field(default_factory=list)
    crud_resources: List[CrudResource] = field(default_factory=list)


__all__ = [
    "Page",
    "App",
]
