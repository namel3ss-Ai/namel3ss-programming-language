"""Extended symbolic expression evaluator with functions, pattern matching, and rules."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from namel3ss.ast.expressions import (
    CallExpr,
    DictExpr,
    Expression,
    FunctionDef,
    IfExpr,
    IndexExpr,
    LambdaExpr,
    LetExpr,
    ListExpr,
    LiteralExpr,
    MatchCase,
    MatchExpr,
    QueryExpr,
    SliceExpr,
    TupleExpr,
    UnifyExpr,
    VarExpr,
)

from .builtins import BUILTIN_FUNCTIONS, get_builtin
from .pattern_matching import match_pattern
from .rule_engine import RuleDatabase, unify

__all__ = [
    "SymbolicEvaluator",
    "EvaluationError",
    "RecursionLimitError",
    "StepLimitError",
]

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Base class for evaluation errors."""
    pass


class RecursionLimitError(EvaluationError):
    """Raised when maximum recursion depth is exceeded."""
    pass


class StepLimitError(EvaluationError):
    """Raised when maximum evaluation steps exceeded."""
    pass


class EvaluationState:
    """Track evaluation state and limits."""
    
    def __init__(
        self,
        max_recursion: int = 100,
        max_steps: int = 10000
    ):
        self.max_recursion = max_recursion
        self.max_steps = max_steps
        self.step_count = 0
        self.recursion_depth = 0
        self.call_stack: List[str] = []
    
    def enter_call(self, name: str) -> None:
        """Enter a function call."""
        self.recursion_depth += 1
        self.call_stack.append(name)
        
        if self.recursion_depth > self.max_recursion:
            raise RecursionLimitError(
                f"Maximum recursion depth ({self.max_recursion}) exceeded. "
                f"Call stack: {' -> '.join(self.call_stack[-10:])}"
            )
    
    def exit_call(self) -> None:
        """Exit a function call."""
        self.recursion_depth -= 1
        if self.call_stack:
            self.call_stack.pop()
    
    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1
        
        if self.step_count > self.max_steps:
            raise StepLimitError(
                f"Maximum evaluation steps ({self.max_steps}) exceeded"
            )


class SymbolicEvaluator:
    """Evaluator for symbolic expressions with functions, patterns, and rules."""
    
    def __init__(
        self,
        env: Optional[Dict[str, Any]] = None,
        functions: Optional[Dict[str, FunctionDef]] = None,
        rule_db: Optional[RuleDatabase] = None,
        max_recursion: int = 100,
        max_steps: int = 10000
    ):
        """
        Initialize evaluator.
        
        Args:
            env: Variable environment
            functions: User-defined functions
            rule_db: Rule database for queries
            max_recursion: Maximum recursion depth
            max_steps: Maximum evaluation steps
        """
        self.env = env or {}
        self.functions = functions or {}
        self.rule_db = rule_db
        self.state = EvaluationState(max_recursion, max_steps)
    
    def eval(self, expr: Expression) -> Any:
        """Evaluate an expression."""
        self.state.step()
        
        if isinstance(expr, LiteralExpr):
            return expr.value
        
        if isinstance(expr, VarExpr):
            return self._eval_var(expr)
        
        if isinstance(expr, ListExpr):
            return self._eval_list(expr)
        
        if isinstance(expr, DictExpr):
            return self._eval_dict(expr)
        
        if isinstance(expr, TupleExpr):
            return self._eval_tuple(expr)
        
        if isinstance(expr, IndexExpr):
            return self._eval_index(expr)
        
        if isinstance(expr, SliceExpr):
            return self._eval_slice(expr)
        
        if isinstance(expr, CallExpr):
            return self._eval_call(expr)
        
        if isinstance(expr, LambdaExpr):
            return self._make_closure(expr)
        
        if isinstance(expr, IfExpr):
            return self._eval_if(expr)
        
        if isinstance(expr, LetExpr):
            return self._eval_let(expr)
        
        if isinstance(expr, MatchExpr):
            return self._eval_match(expr)
        
        if isinstance(expr, FunctionDef):
            # Store function in environment
            self.functions[expr.name] = expr
            return None
        
        if isinstance(expr, QueryExpr):
            return self._eval_query(expr)
        
        if isinstance(expr, UnifyExpr):
            return self._eval_unify(expr)
        
        raise EvaluationError(f"Unsupported expression type: {type(expr).__name__}")
    
    def _eval_var(self, expr: VarExpr) -> Any:
        """Evaluate variable reference."""
        name = expr.name
        
        # Check local environment
        if name in self.env:
            return self.env[name]
        
        # Check built-in functions
        builtin = get_builtin(name)
        if builtin is not None:
            return builtin
        
        # Check user-defined functions
        if name in self.functions:
            func_def = self.functions[name]
            return self._make_closure(LambdaExpr(params=func_def.params, body=func_def.body))
        
        raise EvaluationError(f"Undefined variable: {name}")
    
    def _eval_list(self, expr: ListExpr) -> List[Any]:
        """Evaluate list literal."""
        return [self.eval(elem) for elem in expr.elements]
    
    def _eval_dict(self, expr: DictExpr) -> Dict[Any, Any]:
        """Evaluate dict literal."""
        result = {}
        for key_expr, val_expr in expr.pairs:
            key = self.eval(key_expr)
            value = self.eval(val_expr)
            result[key] = value
        return result
    
    def _eval_tuple(self, expr: TupleExpr) -> tuple:
        """Evaluate tuple literal."""
        return tuple(self.eval(elem) for elem in expr.elements)
    
    def _eval_index(self, expr: IndexExpr) -> Any:
        """Evaluate indexing operation."""
        base = self.eval(expr.base)
        index = self.eval(expr.index)
        return base[index]
    
    def _eval_slice(self, expr: SliceExpr) -> Any:
        """Evaluate slicing operation."""
        base = self.eval(expr.base)
        start = self.eval(expr.start) if expr.start else None
        end = self.eval(expr.end) if expr.end else None
        step = self.eval(expr.step) if expr.step else None
        return base[start:end:step]
    
    def _eval_call(self, expr: CallExpr) -> Any:
        """Evaluate function call."""
        func = self.eval(expr.func)
        
        # Check if it's a closure (lambda) first
        if isinstance(func, dict) and '_closure' in func:
            args = [self.eval(arg) for arg in expr.args]
            kwargs = {k: self.eval(v) for k, v in expr.kwargs.items()}
            return self._call_closure(func, args, kwargs)
        
        if not callable(func):
            raise EvaluationError(f"Cannot call non-function: {type(func).__name__}")
        
        # Evaluate arguments
        args = [self.eval(arg) for arg in expr.args]
        kwargs = {k: self.eval(v) for k, v in expr.kwargs.items()}
        
        # Regular function call
        self.state.enter_call(str(expr.func))
        try:
            return func(*args, **kwargs)
        finally:
            self.state.exit_call()
    
    def _make_closure(self, expr: LambdaExpr) -> Dict[str, Any]:
        """Create a closure from lambda expression."""
        return {
            '_closure': True,
            'params': expr.params,
            'body': expr.body,
            'env': dict(self.env)  # Capture current environment
        }
    
    def _call_closure(self, closure: Dict[str, Any], args: List[Any], kwargs: Dict[str, Any]) -> Any:
        """Call a closure (lambda function)."""
        params = closure['params']
        body = closure['body']
        captured_env = closure['env']
        
        # Create new environment with captured variables
        new_env = dict(captured_env)
        
        # Bind parameters
        for i, param in enumerate(params):
            if i < len(args):
                new_env[param.name] = args[i]
            elif param.name in kwargs:
                new_env[param.name] = kwargs[param.name]
            elif param.default:
                new_env[param.name] = self.eval(param.default)
            else:
                raise EvaluationError(f"Missing required parameter: {param.name}")
        
        # Evaluate body in new environment
        old_env = self.env
        self.env = new_env
        
        self.state.enter_call('<lambda>')
        try:
            return self.eval(body)
        finally:
            self.state.exit_call()
            self.env = old_env
    
    def _eval_if(self, expr: IfExpr) -> Any:
        """Evaluate conditional expression."""
        condition = self.eval(expr.condition)
        
        if condition:
            return self.eval(expr.then_expr)
        elif expr.else_expr:
            return self.eval(expr.else_expr)
        else:
            return None
    
    def _eval_let(self, expr: LetExpr) -> Any:
        """Evaluate let binding."""
        # Create new environment with bindings
        new_env = dict(self.env)
        
        for var_name, val_expr in expr.bindings:
            value = self.eval(val_expr)
            new_env[var_name] = value
        
        # Evaluate body in new environment
        old_env = self.env
        self.env = new_env
        try:
            return self.eval(expr.body)
        finally:
            self.env = old_env
    
    def _eval_match(self, expr: MatchExpr) -> Any:
        """Evaluate pattern matching."""
        value = self.eval(expr.expr)
        
        for case in expr.cases:
            bindings = match_pattern(case.pattern, value)
            
            if bindings is not None:
                # Check guard if present
                if case.guard:
                    old_env = self.env
                    self.env = {**self.env, **bindings}
                    try:
                        guard_result = self.eval(case.guard)
                        if not guard_result:
                            continue
                    finally:
                        self.env = old_env
                
                # Execute case body with pattern bindings
                old_env = self.env
                self.env = {**self.env, **bindings}
                try:
                    return self.eval(case.body)
                finally:
                    self.env = old_env
        
        raise EvaluationError(f"No matching pattern for value: {value}")
    
    def _eval_query(self, expr: QueryExpr) -> List[Dict[str, Any]]:
        """Evaluate rule query."""
        if self.rule_db is None:
            raise EvaluationError("No rule database available for queries")
        
        # Evaluate arguments
        args = [self.eval(arg) for arg in expr.args]
        
        # Query rule database
        solutions = self.rule_db.query(expr.predicate, args, limit=expr.limit)
        return solutions
    
    def _eval_unify(self, expr: UnifyExpr) -> bool:
        """Evaluate unification."""
        left = self.eval(expr.left)
        right = self.eval(expr.right)
        
        result = unify(left, right)
        return result is not None


def evaluate_expression_tree(
    expr: Expression,
    env: Optional[Dict[str, Any]] = None,
    functions: Optional[Dict[str, FunctionDef]] = None,
    rule_db: Optional[RuleDatabase] = None,
    limits: Optional[Dict[str, int]] = None
) -> Any:
    """
    Convenience function to evaluate an expression tree.
    
    Args:
        expr: Expression to evaluate
        env: Variable environment
        functions: User-defined functions
        rule_db: Rule database
        limits: Dict with 'max_recursion' and 'max_steps' keys
    
    Returns:
        Evaluation result
    """
    if limits is None:
        limits = {}
    
    evaluator = SymbolicEvaluator(
        env=env,
        functions=functions,
        rule_db=rule_db,
        max_recursion=limits.get('max_recursion', 100),
        max_steps=limits.get('max_steps', 10000)
    )
    
    return evaluator.eval(expr)
