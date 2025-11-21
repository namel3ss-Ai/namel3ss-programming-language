"""Extended expression AST for symbolic programming in Namel3ss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .base import Expression

__all__ = [
    "LiteralExpr",
    "VarExpr",
    "BinaryOp",
    "AttributeExpr",
    "CallExpr",
    "LambdaExpr",
    "IfExpr",
    "LetExpr",
    "ListExpr",
    "DictExpr",
    "TupleExpr",
    "IndexExpr",
    "SliceExpr",
    "MatchExpr",
    "MatchCase",
    "Pattern",
    "LiteralPattern",
    "VarPattern",
    "ListPattern",
    "DictPattern",
    "TuplePattern",
    "ConstructorPattern",
    "WildcardPattern",
    "FunctionDef",
    "Parameter",
    "RuleDef",
    "RuleHead",
    "RuleBody",
    "RuleClause",
    "QueryExpr",
    "UnifyExpr",
]


# ==================== Core Expression Nodes ====================


@dataclass
class LiteralExpr(Expression):
    """Literal value: number, string, bool, null."""
    value: Any


@dataclass
class VarExpr(Expression):
    """Variable reference: x"""
    name: str


@dataclass
class BinaryOp(Expression):
    """Binary operation: left op right"""
    op: str
    left: Expression
    right: Expression


@dataclass
class AttributeExpr(Expression):
    """Attribute access: base.attr"""
    base: Expression
    attr: str


@dataclass
class CallExpr(Expression):
    """Function call: func(arg1, arg2, ...)"""
    func: Expression
    args: List[Expression] = field(default_factory=list)
    kwargs: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class LambdaExpr(Expression):
    """Anonymous function: fn(x, y) => expr or fn(x, y) { body }"""
    params: List[Parameter] = field(default_factory=list)
    body: Expression = field(default=None)  # type: ignore


@dataclass
class IfExpr(Expression):
    """Conditional expression: if cond then_expr else else_expr"""
    condition: Expression = field(default=None)  # type: ignore
    then_expr: Expression = field(default=None)  # type: ignore
    else_expr: Optional[Expression] = None


@dataclass
class LetExpr(Expression):
    """Local binding: let x = value in body"""
    bindings: List[tuple[str, Expression]] = field(default_factory=list)
    body: Expression = field(default=None)  # type: ignore


# ==================== Collection Expressions ====================


@dataclass
class ListExpr(Expression):
    """List literal: [expr1, expr2, ...]"""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class DictExpr(Expression):
    """Dictionary literal: {key1: val1, key2: val2}"""
    pairs: List[tuple[Expression, Expression]] = field(default_factory=list)


@dataclass
class TupleExpr(Expression):
    """Tuple literal: (expr1, expr2, ...)"""
    elements: List[Expression] = field(default_factory=list)


@dataclass
class IndexExpr(Expression):
    """Indexing: base[index]"""
    base: Expression = field(default=None)  # type: ignore
    index: Expression = field(default=None)  # type: ignore


@dataclass
class SliceExpr(Expression):
    """Slicing: base[start:end:step]"""
    base: Expression = field(default=None)  # type: ignore
    start: Optional[Expression] = None
    end: Optional[Expression] = None
    step: Optional[Expression] = None


# ==================== Pattern Matching ====================


@dataclass
class Pattern:
    """Base class for patterns."""
    pass


@dataclass
class LiteralPattern(Pattern):
    """Match literal value."""
    value: Any


@dataclass
class VarPattern(Pattern):
    """Bind to variable."""
    name: str


@dataclass
class ListPattern(Pattern):
    """Match list structure."""
    elements: List[Pattern] = field(default_factory=list)
    rest_var: Optional[str] = None  # For [x, y, ...rest]


@dataclass
class DictPattern(Pattern):
    """Match dictionary structure."""
    pairs: List[tuple[str, Pattern]] = field(default_factory=list)
    rest_var: Optional[str] = None  # For {x, y, ...rest}


@dataclass
class TuplePattern(Pattern):
    """Match tuple structure."""
    elements: List[Pattern] = field(default_factory=list)


@dataclass
class ConstructorPattern(Pattern):
    """Match constructor: Constructor(pat1, pat2)"""
    name: str
    args: List[Pattern] = field(default_factory=list)


@dataclass
class WildcardPattern(Pattern):
    """Match anything: _"""
    pass


@dataclass
class MatchCase:
    """Single case in match expression."""
    pattern: Pattern = field(default=None)  # type: ignore
    guard: Optional[Expression] = None
    body: Expression = field(default=None)  # type: ignore


@dataclass
class MatchExpr(Expression):
    """Pattern matching: match expr { case pat1 => body1, case pat2 => body2, ... }"""
    expr: Expression = field(default=None)  # type: ignore
    cases: List[MatchCase] = field(default_factory=list)


# ==================== Functions ====================


@dataclass
class Parameter:
    """Function parameter."""
    name: str
    default: Optional[Expression] = None
    type_hint: Optional[str] = None


@dataclass
class FunctionDef(Expression):
    """Named function definition: fn name(params) { body }"""
    name: str
    params: List[Parameter] = field(default_factory=list)
    body: Expression = field(default=None)  # type: ignore
    return_type: Optional[str] = None
    doc: Optional[str] = None


# ==================== Rule Engine ====================


@dataclass
class RuleHead:
    """Head of a rule: predicate(arg1, arg2, ...)"""
    predicate: str
    args: List[Expression] = field(default_factory=list)


@dataclass
class RuleClause:
    """Single clause in rule body: predicate(args) or expression"""
    predicate: Optional[str] = None
    args: List[Expression] = field(default_factory=list)
    expr: Optional[Expression] = None  # For computed conditions
    negated: bool = False  # For negation-as-failure


@dataclass
class RuleBody:
    """Body of a rule: conjunction of clauses."""
    clauses: List[RuleClause] = field(default_factory=list)


@dataclass
class RuleDef(Expression):
    """Rule definition: rule head :- body1, body2, ..."""
    head: RuleHead = field(default=None)  # type: ignore
    body: Optional[RuleBody] = None  # None for facts


@dataclass
class QueryExpr(Expression):
    """Query rules: query predicate(args)"""
    predicate: str
    args: List[Expression] = field(default_factory=list)
    limit: Optional[int] = None  # Limit number of solutions


@dataclass
class UnifyExpr(Expression):
    """Explicit unification: left ~ right"""
    left: Expression = field(default=None)  # type: ignore
    right: Expression = field(default=None)  # type: ignore
