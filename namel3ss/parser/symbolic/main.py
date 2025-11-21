"""Composition class for symbolic expression parser."""

from __future__ import annotations

from typing import List, Optional

from ..base import ParserBase
from .tokenizer import TokenizerMixin
from .tokens import TokenOperationsMixin
from .functions import FunctionParserMixin
from .expressions import ExpressionParserMixin
from .patterns import PatternParserMixin
from .logic import LogicParserMixin
from .parser import MainParserMixin


class SymbolicExpressionParser(
    TokenizerMixin,
    TokenOperationsMixin,
    FunctionParserMixin,
    ExpressionParserMixin,
    PatternParserMixin,
    LogicParserMixin,
    MainParserMixin,
    ParserBase
):
    """
    Parser for symbolic expressions, functional programming, and logic constructs.
    
    This parser extends N3's expression capabilities with advanced functional
    and symbolic programming features, enabling sophisticated data manipulation,
    pattern matching, and declarative logic programming.
    
    Supported Constructs:
        Functions:
            - Named: fn add(x, y) => x + y
            - Lambda: fn(x) => x * 2
            - With blocks: fn process(data) { transform(data) }
            - Type hints: fn calc(x: int, y: int) => x + y
            - Defaults: fn greet(name = "World") => "Hello " + name
        
        Pattern Matching:
            match value {
                case [x, ...rest] => process(x, rest),
                case {key: value} => handle(value),
                case Constructor(a, b) if a > 0 => compute(a, b),
                case _ => default_action()
            }
        
        Let Bindings:
            let x = 10, y = 20 in x + y
        
        Conditional Expressions:
            if condition then value1 else value2
        
        Logic Programming:
            - Facts: rule parent(alice, bob).
            - Rules: rule ancestor(X, Y) :- parent(X, Y).
            - Queries: query ancestor(X, carol) limit 10
        
        Data Structures:
            - Lists: [1, 2, 3]
            - Tuples: (a, b, c)
            - Dicts: {key: value, key2: value2}
        
        Advanced Operations:
            - Indexing: list[0], dict["key"]
            - Slicing: list[1:5], list[:10]
            - Unification: pattern ~ value
            - Function calls with kwargs: func(a, b, opt=value)
    
    Pattern Types:
        - Wildcard: _ (matches anything)
        - Variable: x (binds to value)
        - Literal: 42, "text", True
        - List: [a, b, c] or [head, ...tail]
        - Dict: {key: value} or {a: x, ...rest}
        - Tuple: (a, b, c)
        - Constructor: Some(value), Node(left, right)
    
    Architecture:
        The parser is composed of specialized mixins:
        - TokenizerMixin: Tokenization with regex
        - TokenOperationsMixin: Token operations (peek, consume, expect)
        - FunctionParserMixin: Function definitions and lambdas
        - ExpressionParserMixin: Conditional and let expressions
        - PatternParserMixin: Pattern matching with destructuring
        - LogicParserMixin: Logic programming rules and queries
        - MainParserMixin: Main expression parser integrating all constructs
    """
    
    def __init__(self, code: str):
        """Initialize parser with source code."""
        super().__init__(code)
        # Token-based parsing state
        self.tokens: List[str] = self._tokenize(code)
        self.token_pos: int = 0
