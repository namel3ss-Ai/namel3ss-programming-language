"""Core AST node definitions shared across the Namel3ss parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union


@dataclass
class Theme:
    """Represents global styling variables for an app."""

    values: Dict[str, str] = field(default_factory=dict)


@dataclass
class Expression:
    """Base class for all expression types."""

    pass


@dataclass
class Literal(Expression):
    value: Any


@dataclass
class NameRef(Expression):
    name: str


@dataclass
class AttributeRef(Expression):
    base: str
    attr: str


@dataclass
class BinaryOp(Expression):
    left: Expression
    op: str
    right: Expression


@dataclass
class UnaryOp(Expression):
    op: str
    operand: Expression


@dataclass
class CallExpression(Expression):
    function: Union[NameRef, AttributeRef]
    arguments: List[Expression] = field(default_factory=list)


@dataclass
class ContextValue(Expression):
    scope: Literal["ctx", "env"]
    path: List[str] = field(default_factory=list)
    default: Optional[Any] = None


__all__ = [
    "Theme",
    "Expression",
    "Literal",
    "NameRef",
    "AttributeRef",
    "BinaryOp",
    "UnaryOp",
    "CallExpression",
    "ContextValue",
]
