"""Function parsing for symbolic expressions."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast.expressions import FunctionDef, LambdaExpr, Parameter


class FunctionParserMixin:
    """Mixin for parsing function definitions and lambdas."""
    
    def parse_function_def(self) -> "FunctionDef":
        """
        Parse named function definition.
        
        Syntax: fn name(params) => expr or fn name(params) { body }
        """
        from namel3ss.ast.expressions import FunctionDef, Parameter
        
        self.expect("fn")
        name = self.word()
        self.expect("(")
        
        params: List[Parameter] = []
        while not self.try_consume(")"):
            if params:
                self.expect(",")
            param_name = self.word()
            default = None
            type_hint = None
            
            if self.try_consume(":"):
                type_hint = self.word()
            
            if self.try_consume("="):
                default = self.parse_extended_expression()
            
            params.append(Parameter(name=param_name, default=default, type_hint=type_hint))
        
        # Check for arrow function or block
        if self.try_consume("=>"):
            body = self.parse_extended_expression()
        else:
            self.expect("{")
            body = self.parse_extended_expression()
            self.expect("}")
        
        return FunctionDef(name=name, params=params, body=body)
    
    def parse_lambda(self) -> "LambdaExpr":
        """
        Parse anonymous lambda function.
        
        Syntax: fn(params) => expr or fn(params) { body }
        """
        from namel3ss.ast.expressions import LambdaExpr, Parameter
        
        self.expect("fn")
        self.expect("(")
        
        params: List[Parameter] = []
        while not self.try_consume(")"):
            if params:
                self.expect(",")
            param_name = self.word()
            default = None
            type_hint = None
            
            if self.try_consume(":"):
                type_hint = self.word()
            
            if self.try_consume("="):
                default = self.parse_extended_expression()
            
            params.append(Parameter(name=param_name, default=default, type_hint=type_hint))
        
        if self.try_consume("=>"):
            body = self.parse_extended_expression()
        else:
            self.expect("{")
            body = self.parse_extended_expression()
            self.expect("}")
        
        return LambdaExpr(params=params, body=body)
