"""Expression parsing for if and let constructs."""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast.expressions import IfExpr, LetExpr
    from ...ast.base import Expression


class ExpressionParserMixin:
    """Mixin for parsing conditional and let expressions."""
    
    def parse_if_expr(self) -> "IfExpr":
        """
        Parse conditional expression.
        
        Syntax: if condition then value1 else value2
        """
        from namel3ss.ast.expressions import IfExpr
        
        self.expect("if")
        condition = self.parse_extended_expression()
        
        self.expect("then")
        then_expr = self.parse_extended_expression()
        
        else_expr = None
        if self.try_consume("else"):
            else_expr = self.parse_extended_expression()
        
        return IfExpr(condition=condition, then_expr=then_expr, else_expr=else_expr)
    
    def parse_let_expr(self) -> "LetExpr":
        """
        Parse let binding expression.
        
        Syntax: let x = value1, y = value2 in body
        """
        from namel3ss.ast.expressions import LetExpr
        
        self.expect("let")
        
        bindings: List[Tuple[str, Expression]] = []
        while True:
            var_name = self.word()
            self.expect("=")
            value = self.parse_extended_expression()
            bindings.append((var_name, value))
            
            if not self.try_consume(","):
                break
        
        self.expect("in")
        body = self.parse_extended_expression()
        
        return LetExpr(bindings=bindings, body=body)
