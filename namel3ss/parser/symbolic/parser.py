"""Main expression parser integrating all symbolic constructs."""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast.expressions import (
        ListExpr, DictExpr, TupleExpr, LiteralExpr, VarExpr,
        SliceExpr, IndexExpr, CallExpr, UnifyExpr
    )
    from ...ast.base import Expression


class MainParserMixin:
    """Mixin for main expression parsing entry point."""
    
    def parse_extended_expression(self) -> "Expression":
        """
        Parse any extended expression including all symbolic constructs.
        
        This is the main entry point for parsing symbolic expressions.
        It dispatches to specialized parsers based on detected keywords
        and constructs.
        
        Supported:
            - Control flow: if/then/else
            - Bindings: let/in
            - Pattern matching: match/case
            - Functions: fn (named or lambda)
            - Logic: query
            - Data structures: lists, dicts, tuples
            - Operations: indexing, slicing, calls, unification
        """
        from namel3ss.ast.expressions import (
            ListExpr, DictExpr, TupleExpr, LiteralExpr, VarExpr,
            SliceExpr, IndexExpr, CallExpr, UnifyExpr
        )
        
        # Check for keywords
        if self.peek_word() == "if":
            return self.parse_if_expr()
        
        if self.peek_word() == "let":
            return self.parse_let_expr()
        
        if self.peek_word() == "match":
            return self.parse_match_expr()
        
        if self.peek_word() == "fn":
            # Could be lambda or named function
            # Look ahead to distinguish
            checkpoint = self.token_pos
            self.expect("fn")
            if self.peek() == "(":
                # Lambda
                self.token_pos = checkpoint
                return self.parse_lambda()
            else:
                # Named function - but this should be at statement level
                # For now, allow it as expression
                self.token_pos = checkpoint
                return self.parse_function_def()
        
        if self.peek_word() == "query":
            return self.parse_query_expr()
        
        # List literal
        if self.try_consume("["):
            elements: List[Expression] = []
            while not self.try_consume("]"):
                if elements:
                    self.expect(",")
                elements.append(self.parse_extended_expression())
            return ListExpr(elements=elements)
        
        # Dict literal
        if self.try_consume("{"):
            pairs: List[Tuple[Expression, Expression]] = []
            while not self.try_consume("}"):
                if pairs:
                    self.expect(",")
                key = self.parse_extended_expression()
                self.expect(":")
                value = self.parse_extended_expression()
                pairs.append((key, value))
            return DictExpr(pairs=pairs)
        
        # Tuple or grouped expression
        if self.try_consume("("):
            elements: List[Expression] = []
            while not self.try_consume(")"):
                if elements:
                    self.expect(",")
                elements.append(self.parse_extended_expression())
            
            if len(elements) == 1:
                return elements[0]  # Grouped expression
            return TupleExpr(elements=elements)
        
        # Literals
        if self.peek() and self.peek() in ('"', "'"):
            return LiteralExpr(value=self.string())
        
        if self.peek() and (self.peek().isdigit() or self.peek() == "-"):
            return LiteralExpr(value=self.number())
        
        # Variables and calls
        name = self.word()
        
        # Check for special literals
        if name in ("True", "true"):
            return LiteralExpr(value=True)
        if name in ("False", "false"):
            return LiteralExpr(value=False)
        if name in ("None", "null"):
            return LiteralExpr(value=None)
        
        # Variable or function call
        expr: Expression = VarExpr(name=name)
        
        # Check for indexing, slicing, or calls
        while True:
            if self.try_consume("["):
                # Could be indexing or slicing
                if self.try_consume(":"):
                    # [:end]
                    end = self.parse_extended_expression() if not self.peek() == "]" else None
                    self.expect("]")
                    expr = SliceExpr(base=expr, start=None, end=end)
                else:
                    start = self.parse_extended_expression()
                    if self.try_consume(":"):
                        # [start:end] or [start:]
                        end = self.parse_extended_expression() if not self.peek() == "]" else None
                        self.expect("]")
                        expr = SliceExpr(base=expr, start=start, end=end)
                    else:
                        # [index]
                        self.expect("]")
                        expr = IndexExpr(base=expr, index=start)
            elif self.try_consume("("):
                # Function call
                args: List[Expression] = []
                kwargs: dict[str, Expression] = {}
                
                while not self.try_consume(")"):
                    if args or kwargs:
                        self.expect(",")
                    
                    # Check for keyword argument
                    checkpoint = self.token_pos
                    try:
                        name_part = self.word()
                        if self.try_consume("="):
                            # Keyword argument
                            kwargs[name_part] = self.parse_extended_expression()
                        else:
                            # Positional argument
                            self.token_pos = checkpoint
                            args.append(self.parse_extended_expression())
                    except:
                        self.token_pos = checkpoint
                        args.append(self.parse_extended_expression())
                
                expr = CallExpr(func=expr, args=args, kwargs=kwargs)
            elif self.try_consume("~"):
                # Unification operator
                right = self.parse_extended_expression()
                expr = UnifyExpr(left=expr, right=right)
            else:
                break
        
        return expr
