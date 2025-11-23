"""Page and UI statement AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union, TYPE_CHECKING, Set

from .base import Expression, LogStatement

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .models import InferenceTarget


@dataclass
class Statement:
    """Base class for page statements."""

    pass


@dataclass
class ShowText(Statement):
    text: str
    styles: Dict[str, str] = field(default_factory=dict)


@dataclass
class LayoutSpec:
    width: Optional[int] = None
    height: Optional[int] = None
    variant: Optional[str] = None
    order: Optional[int] = None
    area: Optional[str] = None
    breakpoint: Optional[str] = None
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutMeta:
    """Layout metadata for pages and components."""
    # New fields matching backend encoder expectations
    direction: Optional[str] = None  # "row" | "column"
    spacing: Optional[str] = None  # "small" | "medium" | "large"
    # Legacy fields (kept for backward compatibility)
    width: Optional[int] = None
    height: Optional[int] = None
    variant: Optional[str] = None
    align: Optional[str] = None
    emphasis: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShowTable(Statement):
    title: str
    source_type: str
    source: str
    columns: Optional[List[str]] = None
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    layout: Optional[LayoutMeta] = None
    insight: Optional[str] = None
    dynamic_columns: Optional[Dict[str, Any]] = None


@dataclass
class ShowChart(Statement):
    heading: str
    source_type: str
    source: str
    chart_type: str = "bar"
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    layout: Optional[LayoutMeta] = None
    insight: Optional[str] = None
    encodings: Dict[str, Any] = field(default_factory=dict)
    style: Optional[Dict[str, Any]] = field(default_factory=dict)
    title: Optional[str] = None
    legend: Optional[Dict[str, Any]] = None


@dataclass
class FormField:
    name: str
    field_type: str = "text"


@dataclass
class ActionOperation:
    """Base class for action operations."""

    pass


@dataclass
class UpdateOperation(ActionOperation):
    table: str
    set_expression: str
    where_expression: Optional[str] = None


@dataclass
class ToastOperation(ActionOperation):
    message: str


@dataclass
class GoToPageOperation(ActionOperation):
    page_name: str


@dataclass
class CallPythonOperation(ActionOperation):
    module: str
    method: str
    arguments: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class AskConnectorOperation(ActionOperation):
    connector_name: str
    arguments: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class RunChainOperation(ActionOperation):
    chain_name: str
    inputs: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class RunPromptOperation(ActionOperation):
    prompt_name: str
    arguments: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class ShowForm(Statement):
    title: str
    fields: List[FormField] = field(default_factory=list)
    on_submit_ops: List['ActionOperationType'] = field(default_factory=list)
    styles: Dict[str, str] = field(default_factory=dict)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    effects: Set[str] = field(default_factory=set)


@dataclass
class Action(Statement):
    name: str
    trigger: str
    operations: List['ActionOperationType'] = field(default_factory=list)
    declared_effect: Optional[str] = None
    effects: Set[str] = field(default_factory=set)


@dataclass
class VariableAssignment(Statement):
    name: str
    value: Expression


@dataclass
class IfBlock(Statement):
    condition: Expression
    body: List['PageStatement'] = field(default_factory=list)
    elifs: List['ElifBlock'] = field(default_factory=list)
    else_body: Optional[List['PageStatement']] = None


@dataclass
class ForLoop(Statement):
    loop_var: str
    source_kind: Literal["dataset", "table", "frame"]
    source_name: str
    body: List['PageStatement'] = field(default_factory=list)


@dataclass
class ElifBlock:
    condition: Expression
    body: List['PageStatement'] = field(default_factory=list)


@dataclass
class WhileLoop(Statement):
    condition: Expression
    body: List['PageStatement'] = field(default_factory=list)


@dataclass
class BreakStatement(Statement):
    pass


@dataclass
class ContinueStatement(Statement):
    pass


def _default_inference_target() -> 'InferenceTarget':
    from .models import InferenceTarget

    return InferenceTarget()


@dataclass
class PredictStatement(Statement):
    model_name: str
    input_kind: Literal["dataset", "table", "variables", "payload"] = "dataset"
    input_ref: Optional[str] = None
    assign: 'InferenceTarget' = field(default_factory=_default_inference_target)
    parameters: Dict[str, Any] = field(default_factory=dict)


ActionOperationType = Union[
    UpdateOperation,
    ToastOperation,
    GoToPageOperation,
    CallPythonOperation,
    AskConnectorOperation,
    RunChainOperation,
    RunPromptOperation,
]
PageStatement = Union[
    ShowText,
    ShowTable,
    ShowChart,
    ShowForm,
    Action,
    IfBlock,
    ForLoop,
    WhileLoop,
    VariableAssignment,
    PredictStatement,
    BreakStatement,
    ContinueStatement,
    LogStatement,
]


__all__ = [
    "Statement",
    "ShowText",
    "LayoutSpec",
    "LayoutMeta",
    "ShowTable",
    "ShowChart",
    "FormField",
    "ShowForm",
    "Action",
    "VariableAssignment",
    "IfBlock",
    "ForLoop",
    "WhileLoop",
    "PredictStatement",
    "ElifBlock",
    "BreakStatement",
    "ContinueStatement",
    "UpdateOperation",
    "ToastOperation",
    "GoToPageOperation",
    "CallPythonOperation",
    "AskConnectorOperation",
    "RunChainOperation",
    "RunPromptOperation",
    "ActionOperation",
    "ActionOperationType",
    "PageStatement",
    "LogStatement",
]
