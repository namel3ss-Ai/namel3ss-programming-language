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
    WindowOp,
)


class _ExpressionBuilder(ast.NodeVisitor):
    """Convert Python AST nodes into Namel3ss Expression instances."""

    def __init__(self, raise_error: Callable[[str], None]) -> None:
        self._raise = raise_error

    def convert(self, node: ast.AST) -> Expression:
        if isinstance(node, ast.Expression):
            return self.visit(node)
        return self.visit(ast.Expression(body=node))  # pragma: no cover - defensive

    def visit_Expression(self, node: ast.Expression) -> Expression:  # type: ignore[override]
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Expression:  # type: ignore[override]
        return Literal(node.value)

    def visit_NameConstant(self, node: ast.NameConstant) -> Expression:  # pragma: no cover - py<3.8
        return Literal(node.value)

    def visit_Num(self, node: ast.Num) -> Expression:  # pragma: no cover - py<3.8
        return Literal(node.n)

    def visit_Str(self, node: ast.Str) -> Expression:  # pragma: no cover - py<3.8
        return Literal(node.s)

    def visit_List(self, node: ast.List) -> Expression:  # type: ignore[override]
        return Literal(self._literal_eval(node))

    def visit_Tuple(self, node: ast.Tuple) -> Expression:  # type: ignore[override]
        return Literal(self._literal_eval(node))

    def visit_Set(self, node: ast.Set) -> Expression:  # type: ignore[override]
        return Literal(self._literal_eval(node))

    def visit_Dict(self, node: ast.Dict) -> Expression:  # type: ignore[override]
        return Literal(self._literal_eval(node))

    def visit_Name(self, node: ast.Name) -> Expression:  # type: ignore[override]
        identifier = node.id
        if identifier == _CONTEXT_SENTINEL:
            self._raise("Context reference placeholder cannot appear directly")
        if identifier.startswith('__') and identifier not in {'__builtins__'}:
            self._raise(f"Name '{identifier}' is not permitted in expressions")
        return NameRef(name=identifier)

    def visit_Attribute(self, node: ast.Attribute) -> Expression:  # type: ignore[override]
        return self._convert_attribute(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Expression:  # type: ignore[override]
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return UnaryOp(op='not', operand=operand)
        if isinstance(node.op, ast.USub):
            return UnaryOp(op='-', operand=operand)
        if isinstance(node.op, ast.UAdd):
            return UnaryOp(op='+', operand=operand)
        self._raise(f"Unsupported unary operator '{type(node.op).__name__}'")
        raise AssertionError  # pragma: no cover - unreachable

    def visit_BoolOp(self, node: ast.BoolOp) -> Expression:  # type: ignore[override]
        if not node.values:
            self._raise("Boolean expression is empty")
        if isinstance(node.op, ast.And):
            op = 'and'
        elif isinstance(node.op, ast.Or):
            op = 'or'
        else:
            self._raise("Unsupported boolean operator")
        current = self.visit(node.values[0])
        for value in node.values[1:]:
            current = BinaryOp(left=current, op=op, right=self.visit(value))
        return current

    def visit_BinOp(self, node: ast.BinOp) -> Expression:  # type: ignore[override]
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
        elif isinstance(node.op, ast.LShift):
            op = 'like'
        elif isinstance(node.op, ast.RShift):
            op = 'ilike'
        else:
            self._raise(f"Unsupported binary operator '{type(node.op).__name__}'")
        return BinaryOp(left=left, op=op, right=right)

    def visit_Compare(self, node: ast.Compare) -> Expression:  # type: ignore[override]
        if not node.ops:
            return self.visit(node.left)
        left_expr = self.visit(node.left)
        result: Optional[Expression] = None
        current_left = left_expr
        for op_node, comparator in zip(node.ops, node.comparators):
            right_expr = self.visit(comparator)
            op = self._map_compare_operator(op_node)
            comparison = BinaryOp(left=current_left, op=op, right=right_expr)
            if op == 'in':
                if not isinstance(right_expr, Literal) or not isinstance(right_expr.value, (list, tuple, set)):
                    self._raise("'in' operator requires a literal list, tuple, or set on the right-hand side")
            if result is None:
                result = comparison
            else:
                result = BinaryOp(left=result, op='and', right=comparison)
            current_left = right_expr
        return result if result is not None else left_expr

    def visit_Call(self, node: ast.Call) -> Expression:  # type: ignore[override]
        if isinstance(node.func, ast.Name) and node.func.id == _CONTEXT_SENTINEL:
            return self._convert_context_call(node)
        if node.keywords:
            self._raise("Function calls in expressions do not support keyword arguments")
        arguments: List[Expression] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                self._raise("Varargs are not supported in expressions")
            arguments.append(self.visit(arg))
        function = self._convert_callable(node.func)
        return CallExpression(function=function, arguments=arguments)

    def visit_Subscript(self, node: ast.Subscript) -> Expression:  # type: ignore[override]
        self._raise("Subscript expressions are not supported")
        raise AssertionError  # pragma: no cover - unreachable

    def visit_Lambda(self, node: ast.Lambda) -> Expression:  # pragma: no cover - unsupported
        self._raise("Lambda expressions are not supported")
        raise AssertionError

    def visit_ListComp(self, node: ast.ListComp) -> Expression:  # pragma: no cover - unsupported
        self._raise("Comprehensions are not supported")
        raise AssertionError

    def visit_DictComp(self, node: ast.DictComp) -> Expression:  # pragma: no cover - unsupported
        self._raise("Comprehensions are not supported")
        raise AssertionError

    def visit_SetComp(self, node: ast.SetComp) -> Expression:  # pragma: no cover - unsupported
        self._raise("Comprehensions are not supported")
        raise AssertionError

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Expression:  # pragma: no cover - unsupported
        self._raise("Generator expressions are not supported")
        raise AssertionError

    def generic_visit(self, node: ast.AST) -> Expression:  # type: ignore[override]
        self._raise(f"Unsupported expression element '{type(node).__name__}'")
        raise AssertionError  # pragma: no cover - unreachable

    def _literal_eval(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception as exc:
            self._raise(f"Unsupported literal value: {exc}")
            raise AssertionError  # pragma: no cover - unreachable

    def _convert_attribute(self, node: ast.Attribute) -> Expression:
        base = node.value
        if isinstance(base, ast.Name):
            return AttributeRef(base=base.id, attr=node.attr)
        if isinstance(base, ast.Attribute):
            parent = self._convert_attribute(base)
            prefix = self._flatten_attribute_name(parent)
            return AttributeRef(base=prefix, attr=node.attr)
        self._raise("Attribute access must start from a name or attribute chain")
        raise AssertionError  # pragma: no cover - unreachable

    def _convert_callable(self, node: ast.AST) -> Union[NameRef, AttributeRef]:
        if isinstance(node, ast.Name):
            return NameRef(name=node.id)
        if isinstance(node, ast.Attribute):
            attr_expr = self._convert_attribute(node)
            if isinstance(attr_expr, NameRef):  # pragma: no cover - not expected
                return attr_expr
            return attr_expr
        self._raise("Unsupported function reference in call expression")
        raise AssertionError  # pragma: no cover - unreachable

    def _flatten_attribute_name(self, expr: Expression) -> str:
        if isinstance(expr, NameRef):
            return expr.name
        if isinstance(expr, AttributeRef):
            if expr.base:
                return f"{expr.base}.{expr.attr}"
            return expr.attr
        self._raise("Invalid attribute chain")
        raise AssertionError  # pragma: no cover - unreachable

    def _convert_context_call(self, node: ast.Call) -> ContextValue:
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
        self._raise(f"Unsupported comparison operator '{type(op).__name__}'")
