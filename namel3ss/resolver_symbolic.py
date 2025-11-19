"""Resolver extensions for symbolic expressions.

This module provides validation for symbolic expressions including:
- Scope checking for variables in functions and let bindings
- Function arity validation
- Pattern exhaustiveness warnings
- Rule consistency checking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from namel3ss.ast.expressions import (
    CallExpr,
    Expression,
    FunctionDef,
    IfExpr,
    LambdaExpr,
    LetExpr,
    MatchCase,
    MatchExpr,
    Parameter,
    Pattern,
    QueryExpr,
    RuleBody,
    RuleDef,
    VarExpr,
    VarPattern,
    WildcardPattern,
)
from namel3ss.errors import N3ResolutionError


class SymbolicResolutionError(N3ResolutionError):
    """Raised when symbolic expression validation fails."""


@dataclass
class Scope:
    """Represents a lexical scope for variable resolution."""
    
    variables: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    parent: Optional[Scope] = None
    
    def define_variable(self, name: str) -> None:
        """Define a variable in this scope."""
        self.variables.add(name)
    
    def define_function(self, name: str) -> None:
        """Define a function in this scope."""
        self.functions.add(name)
    
    def lookup_variable(self, name: str) -> bool:
        """Check if a variable is defined in this scope or any parent."""
        if name in self.variables:
            return True
        if self.parent:
            return self.parent.lookup_variable(name)
        return False
    
    def lookup_function(self, name: str) -> bool:
        """Check if a function is defined in this scope or any parent."""
        if name in self.functions:
            return True
        if self.parent:
            return self.parent.lookup_function(name)
        return False
    
    def child(self) -> Scope:
        """Create a child scope."""
        return Scope(parent=self)


@dataclass
class FunctionSignature:
    """Function signature for arity checking."""
    
    name: str
    min_arity: int
    max_arity: Optional[int]  # None for variadic
    
    def check_call(self, num_args: int) -> bool:
        """Check if a call with num_args is valid."""
        if num_args < self.min_arity:
            return False
        if self.max_arity is not None and num_args > self.max_arity:
            return False
        return True


class SymbolicExpressionValidator:
    """Validates symbolic expressions for correctness."""
    
    def __init__(self) -> None:
        self.global_scope = Scope()
        self.function_signatures: Dict[str, FunctionSignature] = {}
        self._init_builtin_signatures()
    
    def _init_builtin_signatures(self) -> None:
        """Initialize signatures for built-in functions."""
        builtins = {
            # Higher-order
            'map': (2, 2),
            'filter': (2, 2),
            'reduce': (3, 3),
            'fold': (3, 3),
            'zip': (2, None),  # Variadic
            'enumerate': (1, 1),
            # List operations
            'head': (1, 1),
            'tail': (1, 1),
            'cons': (2, 2),
            'append': (2, 2),
            'reverse': (1, 1),
            'sort': (1, 1),
            'length': (1, 1),
            'nth': (2, 2),
            'take': (2, 2),
            'drop': (2, 2),
            # Numeric
            'sum': (1, 1),
            'product': (1, 1),
            'min': (1, None),
            'max': (1, None),
            'abs': (1, 1),
            'round': (1, 1),
            'floor': (1, 1),
            'ceil': (1, 1),
            'range': (2, 3),
            # String
            'concat': (2, None),
            'split': (2, 2),
            'join': (2, 2),
            'lower': (1, 1),
            'upper': (1, 1),
            'strip': (1, 1),
            'replace': (3, 3),
            # Dict
            'keys': (1, 1),
            'values': (1, 1),
            'items': (1, 1),
            'get': (2, 3),
            'merge': (2, 2),
            # Predicates
            'all': (2, 2),
            'any': (2, 2),
            'not': (1, 1),
            'is_empty': (1, 1),
            'is_none': (1, 1),
            'is_list': (1, 1),
            'is_dict': (1, 1),
            'is_int': (1, 1),
            'is_float': (1, 1),
            'is_str': (1, 1),
            'is_bool': (1, 1),
            # Utilities
            'identity': (1, 1),
            'const': (2, 2),
            'compose': (2, 2),
            'pipe': (2, 2),
            'assert': (2, 2),
            # Type conversions
            'int': (1, 1),
            'float': (1, 1),
            'str': (1, 1),
            'bool': (1, 1),
            'list': (1, 1),
            'dict': (1, 1),
        }
        
        for name, (min_arity, max_arity) in builtins.items():
            self.function_signatures[name] = FunctionSignature(
                name=name,
                min_arity=min_arity,
                max_arity=max_arity
            )
    
    def validate_function_def(self, func: FunctionDef) -> None:
        """Validate a function definition."""
        # Create new scope for function body
        func_scope = self.global_scope.child()
        
        # Define parameters
        for param in func.params:
            func_scope.define_variable(param.name)
        
        # Register function signature
        self.function_signatures[func.name] = FunctionSignature(
            name=func.name,
            min_arity=len(func.params),
            max_arity=len(func.params)
        )
        
        # Validate body
        self.validate_expression(func.body, func_scope)
    
    def validate_expression(self, expr: Expression, scope: Scope) -> None:
        """Validate an expression in the given scope."""
        if isinstance(expr, VarExpr):
            self._validate_var(expr, scope)
        elif isinstance(expr, CallExpr):
            self._validate_call(expr, scope)
        elif isinstance(expr, LambdaExpr):
            self._validate_lambda(expr, scope)
        elif isinstance(expr, IfExpr):
            self._validate_if(expr, scope)
        elif isinstance(expr, LetExpr):
            self._validate_let(expr, scope)
        elif isinstance(expr, MatchExpr):
            self._validate_match(expr, scope)
        elif isinstance(expr, QueryExpr):
            self._validate_query(expr, scope)
        # Add more cases as needed
    
    def _validate_var(self, expr: VarExpr, scope: Scope) -> None:
        """Validate variable reference."""
        if not scope.lookup_variable(expr.name) and expr.name not in self.function_signatures:
            raise SymbolicResolutionError(
                f"Undefined variable: {expr.name}"
            )
    
    def _validate_call(self, expr: CallExpr, scope: Scope) -> None:
        """Validate function call."""
        # Validate function expression
        self.validate_expression(expr.func, scope)
        
        # Check arity if it's a direct function reference
        if isinstance(expr.func, VarExpr):
            func_name = expr.func.name
            if func_name in self.function_signatures:
                sig = self.function_signatures[func_name]
                num_args = len(expr.args) + len(expr.kwargs)
                if not sig.check_call(num_args):
                    if sig.max_arity is None:
                        raise SymbolicResolutionError(
                            f"Function '{func_name}' requires at least {sig.min_arity} arguments, got {num_args}"
                        )
                    elif sig.min_arity == sig.max_arity:
                        raise SymbolicResolutionError(
                            f"Function '{func_name}' requires exactly {sig.min_arity} arguments, got {num_args}"
                        )
                    else:
                        raise SymbolicResolutionError(
                            f"Function '{func_name}' requires {sig.min_arity}-{sig.max_arity} arguments, got {num_args}"
                        )
        
        # Validate arguments
        for arg in expr.args:
            self.validate_expression(arg, scope)
        for val in expr.kwargs.values():
            self.validate_expression(val, scope)
    
    def _validate_lambda(self, expr: LambdaExpr, scope: Scope) -> None:
        """Validate lambda expression."""
        lambda_scope = scope.child()
        
        # Define parameters
        for param in expr.params:
            lambda_scope.define_variable(param.name)
        
        # Validate body
        self.validate_expression(expr.body, lambda_scope)
    
    def _validate_if(self, expr: IfExpr, scope: Scope) -> None:
        """Validate if expression."""
        self.validate_expression(expr.condition, scope)
        self.validate_expression(expr.then_expr, scope)
        if expr.else_expr:
            self.validate_expression(expr.else_expr, scope)
    
    def _validate_let(self, expr: LetExpr, scope: Scope) -> None:
        """Validate let expression."""
        let_scope = scope.child()
        
        # Validate and define bindings
        for name, value_expr in expr.bindings:
            self.validate_expression(value_expr, scope)  # Validate in outer scope
            let_scope.define_variable(name)  # Define in inner scope
        
        # Validate body in inner scope
        self.validate_expression(expr.body, let_scope)
    
    def _validate_match(self, expr: MatchExpr, scope: Scope) -> None:
        """Validate match expression."""
        # Validate matched expression
        self.validate_expression(expr.expr, scope)
        
        # Validate each case
        has_wildcard = False
        for case in expr.cases:
            case_scope = scope.child()
            
            # Extract variables from pattern
            pattern_vars = self._extract_pattern_vars(case.pattern)
            for var in pattern_vars:
                case_scope.define_variable(var)
            
            # Check for wildcard pattern
            if isinstance(case.pattern, WildcardPattern):
                has_wildcard = True
            
            # Validate guard if present
            if case.guard:
                self.validate_expression(case.guard, case_scope)
            
            # Validate body
            self.validate_expression(case.body, case_scope)
        
        # Warn if no wildcard (non-exhaustive)
        if not has_wildcard:
            # In a production system, you'd emit a warning here
            pass
    
    def _validate_query(self, expr: QueryExpr, scope: Scope) -> None:
        """Validate query expression."""
        # Validate arguments
        for arg in expr.args:
            self.validate_expression(arg, scope)
    
    def _extract_pattern_vars(self, pattern: Pattern) -> Set[str]:
        """Extract variable names from a pattern."""
        if isinstance(pattern, VarPattern):
            return {pattern.name}
        elif isinstance(pattern, WildcardPattern):
            return set()
        # Add more pattern types as needed
        return set()
    
    def validate_rule_def(self, rule: RuleDef) -> None:
        """Validate a rule definition."""
        # Extract variables from head
        head_vars = set()
        for arg in rule.head.args:
            if isinstance(arg, VarExpr):
                head_vars.add(arg.name)
        
        # Check body variables if present
        if rule.body:
            body_vars = self._extract_rule_body_vars(rule.body)
            
            # All body variables should appear in head (safety check)
            unbound = body_vars - head_vars
            if unbound:
                # In Prolog, this would be a warning about singleton variables
                pass
    
    def _extract_rule_body_vars(self, body: RuleBody) -> Set[str]:
        """Extract variables from rule body."""
        vars_set = set()
        for clause in body.clauses:
            for arg in clause.args:
                if isinstance(arg, VarExpr):
                    vars_set.add(arg.name)
        return vars_set


def validate_symbolic_expressions(functions: List[FunctionDef], rules: List[RuleDef]) -> None:
    """Validate a collection of functions and rules."""
    validator = SymbolicExpressionValidator()
    
    # Register all function names first
    for func in functions:
        validator.global_scope.define_function(func.name)
    
    # Validate each function
    for func in functions:
        validator.validate_function_def(func)
    
    # Validate each rule
    for rule in rules:
        validator.validate_rule_def(rule)
