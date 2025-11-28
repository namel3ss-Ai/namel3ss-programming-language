"""
Enhanced Expression Builder with Lambda, Subscript, and Comprehension Support

This module extends the base expression builder to support modern expression features:
- Lambda expressions: fn(x) => x * 2
- Subscript/indexing: arr[0], obj['key']
- List comprehensions: [x * 2 for x in items if x > 0]
- Map/filter built-ins for functional programming

These enhancements make Namel3ss feel like a real, expressive language rather than
a toy DSL.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, List, Union

from namel3ss.ast import (
    AttributeRef,
    BinaryOp,
    CallExpression,
    ContextValue,
    Expression,
    Literal,
    NameRef,
    UnaryOp,
)
from namel3ss.ast.expressions import (
    LiteralExpr,
    VarExpr,
    BinaryOp as ExprBinaryOp,
    AttributeExpr,
    CallExpr,
    LambdaExpr,
    IfExpr,
    LetExpr,
    ListExpr,
    DictExpr,
    TupleExpr,
    IndexExpr,
    SliceExpr,
    Parameter,
)


class EnhancedExpressionBuilder(ast.NodeVisitor):
    """
    Enhanced expression builder supporting:
    - Lambda expressions
    - Subscript/indexing
    - List/dict/set comprehensions
    - Functional programming patterns
    """

    def __init__(self, raise_error: Callable[[str], None]) -> None:
        self._raise = raise_error
        self._use_new_ast = True  # Use new expression AST nodes

    def convert(self, node: ast.AST) -> Expression:
        """Convert a Python AST node to a Namel3ss expression."""
        if isinstance(node, ast.Expression):
            return self.visit(node)
        return self.visit(ast.Expression(body=node))

    def visit_Expression(self, node: ast.Expression) -> Expression:
        return self.visit(node.body)

    # ==================== Literals ====================

    def visit_Constant(self, node: ast.Constant) -> Expression:
        return LiteralExpr(node.value) if self._use_new_ast else Literal(node.value)

    def visit_NameConstant(self, node: ast.NameConstant) -> Expression:  # py<3.8
        return LiteralExpr(node.value) if self._use_new_ast else Literal(node.value)

    def visit_Num(self, node: ast.Num) -> Expression:  # py<3.8
        return LiteralExpr(node.n) if self._use_new_ast else Literal(node.n)

    def visit_Str(self, node: ast.Str) -> Expression:  # py<3.8
        return LiteralExpr(node.s) if self._use_new_ast else Literal(node.s)

    def visit_List(self, node: ast.List) -> Expression:
        if self._use_new_ast:
            elements = [self.visit(elem) for elem in node.elts]
            return ListExpr(elements)
        return Literal(self._literal_eval(node))

    def visit_Tuple(self, node: ast.Tuple) -> Expression:
        if self._use_new_ast:
            elements = [self.visit(elem) for elem in node.elts]
            return TupleExpr(elements)
        return Literal(self._literal_eval(node))

    def visit_Set(self, node: ast.Set) -> Expression:
        # Convert set to list for now
        if self._use_new_ast:
            elements = [self.visit(elem) for elem in node.elts]
            return ListExpr(elements)
        return Literal(self._literal_eval(node))

    def visit_Dict(self, node: ast.Dict) -> Expression:
        if self._use_new_ast:
            pairs = []
            for key_node, value_node in zip(node.keys, node.values):
                key = self.visit(key_node)
                value = self.visit(value_node)
                pairs.append((key, value))
            return DictExpr(pairs)
        return Literal(self._literal_eval(node))

    # ==================== Names & Attributes ====================

    def visit_Name(self, node: ast.Name) -> Expression:
        identifier = node.id
        if identifier == '_namel3ss_context':
            self._raise("Context reference placeholder cannot appear directly")
        if identifier.startswith('__') and identifier not in {'__builtins__'}:
            self._raise(f"Name '{identifier}' is not permitted in expressions")
        
        if self._use_new_ast:
            return VarExpr(name=identifier)
        return NameRef(name=identifier)

    def visit_Attribute(self, node: ast.Attribute) -> Expression:
        if self._use_new_ast:
            base = self.visit(node.value)
            return AttributeExpr(base=base, attr=node.attr)
        return self._convert_attribute(node)

    # ==================== Operators ====================

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Expression:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            op = 'not'
        elif isinstance(node.op, ast.USub):
            op = '-'
        elif isinstance(node.op, ast.UAdd):
            op = '+'
        else:
            self._raise(f"Unsupported unary operator '{type(node.op).__name__}'")
            op = '?'
        
        if self._use_new_ast:
            # Unary op not in new AST yet, use binary op placeholder
            return ExprBinaryOp(op=op, left=LiteralExpr(0), right=operand)
        return UnaryOp(op=op, operand=operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> Expression:
        if not node.values:
            self._raise("Boolean expression is empty")
        if isinstance(node.op, ast.And):
            op = 'and'
        elif isinstance(node.op, ast.Or):
            op = 'or'
        else:
            self._raise("Unsupported boolean operator")
            op = '?'
        
        current = self.visit(node.values[0])
        for value in node.values[1:]:
            right = self.visit(value)
            if self._use_new_ast:
                current = ExprBinaryOp(left=current, op=op, right=right)
            else:
                current = BinaryOp(left=current, op=op, right=right)
        return current

    def visit_BinOp(self, node: ast.BinOp) -> Expression:
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.Add):
            op = '+'
        elif isinstance(node.op, ast.Sub):
            op = '-'
        elif isinstance(node.op, ast.Mult):
            op = '*'
        elif isinstance(node.op, ast.Div):
            op = '/'
        elif isinstance(node.op, ast.Mod):
            op = '%'
        elif isinstance(node.op, ast.Pow):
            op = '**'
        elif isinstance(node.op, ast.LShift):
            op = 'like'
        elif isinstance(node.op, ast.RShift):
            op = 'ilike'
        else:
            self._raise(f"Unsupported binary operator '{type(node.op).__name__}'")
            op = '?'
        
        if self._use_new_ast:
            return ExprBinaryOp(left=left, op=op, right=right)
        return BinaryOp(left=left, op=op, right=right)

    def visit_Compare(self, node: ast.Compare) -> Expression:
        if not node.ops:
            return self.visit(node.left)
        
        left_expr = self.visit(node.left)
        result: Expression = None  # type: ignore
        current_left = left_expr
        
        for op_node, comparator in zip(node.ops, node.comparators):
            right_expr = self.visit(comparator)
            op = self._map_compare_operator(op_node)
            
            if self._use_new_ast:
                comparison = ExprBinaryOp(left=current_left, op=op, right=right_expr)
            else:
                comparison = BinaryOp(left=current_left, op=op, right=right_expr)
            
            if op == 'in':
                if not isinstance(right_expr, (Literal, LiteralExpr, ListExpr)):
                    self._raise("'in' operator requires a literal list, tuple, or set on the right-hand side")
            
            if result is None:
                result = comparison
            else:
                if self._use_new_ast:
                    result = ExprBinaryOp(left=result, op='and', right=comparison)
                else:
                    result = BinaryOp(left=result, op='and', right=comparison)
            
            current_left = right_expr
        
        return result if result is not None else left_expr

    # ==================== Function Calls ====================

    def visit_Call(self, node: ast.Call) -> Expression:
        if isinstance(node.func, ast.Name) and node.func.id == '_namel3ss_context':
            return self._convert_context_call(node)
        
        if node.keywords:
            self._raise("Function calls in expressions do not support keyword arguments")
        
        arguments: List[Expression] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                self._raise("Varargs are not supported in expressions")
            arguments.append(self.visit(arg))
        
        function = self.visit(node.func)
        
        if self._use_new_ast:
            return CallExpr(func=function, args=arguments)
        
        # Convert function to NameRef or AttributeRef for legacy AST
        if isinstance(function, (VarExpr, NameRef)):
            func_name = function.name if isinstance(function, VarExpr) else function.name
            function = NameRef(name=func_name)
        
        return CallExpression(function=function, arguments=arguments)

    # ==================== Lambda Expressions (NEW!) ====================

    def visit_Lambda(self, node: ast.Lambda) -> Expression:
        """
        Support lambda expressions: lambda x: x * 2
        Converts to: fn(x) => x * 2
        """
        params = []
        for arg in node.args.args:
            param_name = arg.arg
            # Extract type annotation if present
            type_annotation = None
            if arg.annotation:
                # Convert annotation to string for now
                type_annotation = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else None
            
            param = Parameter(name=param_name, type_annotation=type_annotation)
            params.append(param)
        
        body = self.visit(node.body)
        
        return LambdaExpr(params=params, body=body)

    # ==================== Subscript/Indexing (NEW!) ====================

    def visit_Subscript(self, node: ast.Subscript) -> Expression:
        """
        Support subscript expressions: arr[0], obj['key'], arr[1:5]
        """
        base = self.visit(node.value)
        
        # Check if it's a slice or simple index
        if isinstance(node.slice, ast.Slice):
            # Slicing: arr[start:end:step]
            start = self.visit(node.slice.lower) if node.slice.lower else None
            end = self.visit(node.slice.upper) if node.slice.upper else None
            step = self.visit(node.slice.step) if node.slice.step else None
            return SliceExpr(base=base, start=start, end=end, step=step)
        else:
            # Simple indexing: arr[index]
            index = self.visit(node.slice)
            return IndexExpr(base=base, index=index)

    # ==================== Comprehensions (NEW!) ====================

    def visit_ListComp(self, node: ast.ListComp) -> Expression:
        """
        Support list comprehensions: [x * 2 for x in items if x > 0]
        
        Converts to nested map/filter calls:
        map(filter(items, fn(x) => x > 0), fn(x) => x * 2)
        """
        # Extract the element expression
        elem_expr = self.visit(node.elt)
        
        # Process generators (for clauses)
        if not node.generators:
            self._raise("List comprehension must have at least one generator")
        
        # For now, support single generator with optional filter
        if len(node.generators) > 1:
            self._raise("Nested comprehensions not yet supported")
        
        gen = node.generators[0]
        iter_expr = self.visit(gen.iter)
        target_name = gen.target.id if isinstance(gen.target, ast.Name) else None
        
        if target_name is None:
            self._raise("Comprehension target must be a simple variable")
        
        # Build result expression
        result = iter_expr
        
        # Apply filters if present
        if gen.ifs:
            for if_clause in gen.ifs:
                filter_condition = self.visit(if_clause)
                # Wrap in lambda: fn(target) => condition
                filter_lambda = LambdaExpr(
                    params=[Parameter(name=target_name)],
                    body=filter_condition
                )
                # result = filter(result, filter_lambda)
                result = CallExpr(
                    func=VarExpr(name='filter'),
                    args=[result, filter_lambda]
                )
        
        # Apply map for element expression
        map_lambda = LambdaExpr(
            params=[Parameter(name=target_name)],
            body=elem_expr
        )
        result = CallExpr(
            func=VarExpr(name='map'),
            args=[result, map_lambda]
        )
        
        return result

    def visit_DictComp(self, node: ast.DictComp) -> Expression:
        """
        Support dict comprehensions: {k: v for k, v in items}
        
        For now, convert to map over items and collect into dict.
        """
        self._raise("Dictionary comprehensions not yet fully supported. Use map() instead.")
        # TODO: Implement dict comprehension support
        raise AssertionError

    def visit_SetComp(self, node: ast.SetComp) -> Expression:
        """Set comprehensions - treat as list comprehensions for now"""
        self._raise("Set comprehensions not yet supported. Use list comprehensions instead.")
        raise AssertionError

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Expression:
        """Generator expressions - not supported in DSL context"""
        self._raise("Generator expressions are not supported")
        raise AssertionError

    # ==================== Other Expressions ====================

    def visit_IfExp(self, node: ast.IfExp) -> Expression:
        """Ternary operator: x if condition else y"""
        condition = self.visit(node.test)
        then_expr = self.visit(node.body)
        else_expr = self.visit(node.orelse)
        
        return IfExpr(condition=condition, then_expr=then_expr, else_expr=else_expr)

    def generic_visit(self, node: ast.AST) -> Expression:
        self._raise(f"Unsupported expression element '{type(node).__name__}'")
        raise AssertionError

    # ==================== Helper Methods ====================

    def _literal_eval(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception as exc:
            self._raise(f"Unsupported literal value: {exc}")
            raise AssertionError

    def _convert_attribute(self, node: ast.Attribute) -> Expression:
        """Convert attribute access for legacy AST"""
        base = node.value
        if isinstance(base, ast.Name):
            return AttributeRef(base=base.id, attr=node.attr)
        if isinstance(base, ast.Attribute):
            parent = self._convert_attribute(base)
            prefix = self._flatten_attribute_name(parent)
            return AttributeRef(base=prefix, attr=node.attr)
        self._raise("Attribute access must start from a name or attribute chain")
        raise AssertionError

    def _flatten_attribute_name(self, expr: Expression) -> str:
        if isinstance(expr, (NameRef, VarExpr)):
            return expr.name
        if isinstance(expr, AttributeRef):
            if expr.base:
                return f"{expr.base}.{expr.attr}"
            return expr.attr
        self._raise("Invalid attribute chain")
        raise AssertionError

    def _convert_context_call(self, node: ast.Call) -> ContextValue:
        """Convert context reference: ctx('scope', 'path', 'segments')"""
        if node.keywords:
            self._raise("Context references do not support keyword arguments")
        if not node.args:
            self._raise("Context reference requires scope and path segments")
        
        scope_node = node.args[0]
        if not isinstance(scope_node, ast.Constant) or not isinstance(scope_node.value, str):
            self._raise("Context scope must be a string literal")
        scope = scope_node.value.lower()
        if scope not in {"ctx", "env"}:
            self._raise("Context scope must be 'ctx' or 'env'")
        
        path: List[str] = []
        for arg in node.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                path.append(arg.value)
                continue
            self._raise("Context path segments must be string literals")
        
        if not path:
            self._raise("Context reference must include at least one path segment")
        
        return ContextValue(scope=scope, path=path)

    def _map_compare_operator(self, op: ast.cmpop) -> str:
        """Map Python comparison operators to Namel3ss operators"""
        if isinstance(op, ast.Eq):
            return '=='
        if isinstance(op, ast.NotEq):
            return '!='
        if isinstance(op, ast.Lt):
            return '<'
        if isinstance(op, ast.LtE):
            return '<='
        if isinstance(op, ast.Gt):
            return '>'
        if isinstance(op, ast.GtE):
            return '>='
        if isinstance(op, ast.In):
            return 'in'
        if isinstance(op, ast.NotIn):
            return 'not in'
        self._raise(f"Unsupported comparison operator '{type(op).__name__}'")
        return '?'


# ==================== Public API ====================


def build_enhanced_expression(source: str, raise_error: Callable[[str], None]) -> Expression:
    """
    Build an enhanced expression from Python source code.
    
    This function parses Python expression syntax and converts it to Namel3ss
    expression AST, supporting lambdas, subscripts, and comprehensions.
    
    Args:
        source: Python expression source code
        raise_error: Error callback function
    
    Returns:
        Namel3ss Expression AST node
    
    Example:
        ```python
        # Lambda
        expr = build_enhanced_expression("lambda x: x * 2", error_callback)
        
        # Subscript
        expr = build_enhanced_expression("arr[0]", error_callback)
        
        # Comprehension
        expr = build_enhanced_expression("[x * 2 for x in items if x > 0]", error_callback)
        ```
    """
    try:
        tree = ast.parse(source, mode='eval')
        builder = EnhancedExpressionBuilder(raise_error)
        return builder.convert(tree)
    except SyntaxError as e:
        raise_error(f"Invalid expression syntax: {e}")
        raise


__all__ = [
    "EnhancedExpressionBuilder",
    "build_enhanced_expression",
]
