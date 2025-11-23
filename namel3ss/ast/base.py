"""Core AST node definitions shared across the Namel3ss parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .source_location import SourceLocation


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


class LogLevel(Enum):
    """Logging levels for log statements."""
    
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class LogStatement(Expression):
    """Represents a logging statement in the DSL.
    
    Supports various forms:
    - log "message" (defaults to info level)
    - log info "message"
    - log warn "interpolated {{variable}}"
    """
    
    level: LogLevel
    message: Expression  # Can be Literal or interpolated expression
    source_location: Optional["SourceLocation"] = None


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
    "LogLevel",
    "LogStatement",
]
