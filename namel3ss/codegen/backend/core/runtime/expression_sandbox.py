"""Sandboxed expression evaluation shared across runtime sections."""

from __future__ import annotations

import ast
import inspect
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "ExpressionSandboxError",
    "SandboxedExpressionEvaluator",
    "evaluate_sandboxed_expression",
    "evaluate_expression_tree",
]


_SAFE_DICT_METHODS: Set[str] = {"get", "items", "values", "keys"}
_SAFE_STR_METHODS: Set[str] = {
    "lower",
    "upper",
    "title",
    "strip",
    "lstrip",
    "rstrip",
    "startswith",
    "endswith",
    "capitalize",
    "casefold",
    "replace",
    "split",
    "join",
}
_SAFE_LIST_METHODS: Set[str] = {"count", "index"}
_DEFAULT_SAFE_SIMPLE_CALLABLES: Set[Callable[..., Any]] = {
    len,
    min,
    max,
    sum,
    abs,
    round,
    int,
    float,
    str,
    bool,
}
_DEFAULT_NAME_CONSTANTS: Dict[str, Any] = {"True": True, "False": False, "None": None}
_MISSING = object()


class ExpressionSandboxError(RuntimeError):
    """Raised when an expression violates sandbox restrictions."""


def _is_safe_callable(func: Any, extra_callables: Set[Callable[..., Any]]) -> bool:
    if func in _DEFAULT_SAFE_SIMPLE_CALLABLES or func in extra_callables:
        return True
    module_name = getattr(func, "__module__", "")
    if module_name == "math":
        return True
    owner = getattr(func, "__self__", None)
    name = getattr(func, "__name__", "")
    if isinstance(owner, dict) and name in _SAFE_DICT_METHODS:
        return True
    if isinstance(owner, str) and name in _SAFE_STR_METHODS:
        return True
    if isinstance(owner, list) and name in _SAFE_LIST_METHODS:
        return True
    if inspect.isbuiltin(func) and func in extra_callables:
        return True
    return False


class SandboxedExpressionEvaluator(ast.NodeVisitor):
    """Evaluate expressions using a restricted subset of the Python AST."""

    def __init__(
        self,
        scope: Dict[str, Any],
        *,
        extra_callables: Optional[Iterable[Callable[..., Any]]] = None,
        allow_dict_key_attributes: bool = False,
        allowed_attribute_types: Optional[Sequence[type]] = None,
        name_constants: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._scope = scope
        self._extra_callables: Set[Callable[..., Any]] = set(extra_callables or ())
        self._allow_dict_key_attributes = allow_dict_key_attributes
        if allowed_attribute_types is None:
            self._allowed_attribute_types: Tuple[type, ...] = (SimpleNamespace,)
        else:
            self._allowed_attribute_types = tuple(allowed_attribute_types)
        self._name_constants: Dict[str, Any] = dict(_DEFAULT_NAME_CONSTANTS)
        if name_constants:
            self._name_constants.update(name_constants)

    def evaluate(self, expression: str) -> Any:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - defensive parsing guard
            raise ExpressionSandboxError("Invalid expression") from exc
        return self.visit(tree)

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ExpressionSandboxError(f"Unsupported expression element '{type(node).__name__}'")
        return method(node)  # type: ignore[misc]

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        identifier = node.id
        if identifier.startswith("__"):
            raise ExpressionSandboxError(f"Name '{identifier}' is not permitted")
        if identifier in self._name_constants:
            return self._name_constants[identifier]
        if identifier in self._scope:
            return self._scope[identifier]
        raise ExpressionSandboxError(f"Unknown name '{identifier}' in expression")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = True
            for value_node in node.values:
                if not self.visit(value_node):
                    return False
            return result
        if isinstance(node.op, ast.Or):
            for value_node in node.values:
                if self.visit(value_node):
                    return True
            return False
        raise ExpressionSandboxError("Unsupported boolean operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            return not operand
        raise ExpressionSandboxError("Unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ExpressionSandboxError("Unsupported binary operator")

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for comparator, op in zip(node.comparators, node.ops):
            right = self.visit(comparator)
            if isinstance(op, ast.Eq) and not (left == right):
                return False
            if isinstance(op, ast.NotEq) and not (left != right):
                return False
            if isinstance(op, ast.Lt) and not (left < right):
                return False
            if isinstance(op, ast.LtE) and not (left <= right):
                return False
            if isinstance(op, ast.Gt) and not (left > right):
                return False
            if isinstance(op, ast.GtE) and not (left >= right):
                return False
            if isinstance(op, ast.In) and not (left in right):
                return False
            if isinstance(op, ast.NotIn) and not (left not in right):
                return False
            if isinstance(op, ast.Is) and not (left is right):
                return False
            if isinstance(op, ast.IsNot) and not (left is not right):
                return False
            left = right
        return True

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        if not _is_safe_callable(func, self._extra_callables):
            raise ExpressionSandboxError("Call to unsupported function")
        args = [self.visit(arg) for arg in node.args]
        kwargs: Dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ExpressionSandboxError("Argument unpacking is not permitted")
            kwargs[kw.arg] = self.visit(kw.value)
        return func(*args, **kwargs)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        value = self.visit(node.value)
        attr = node.attr
        if attr.startswith("__"):
            raise ExpressionSandboxError(f"Attribute '{attr}' is not permitted")
        if inspect.ismodule(value) and getattr(value, "__name__", "") == "math":
            if not hasattr(value, attr):
                raise ExpressionSandboxError(f"Unknown math attribute '{attr}'")
            return getattr(value, attr)
        if isinstance(value, dict):
            if attr in _SAFE_DICT_METHODS:
                return getattr(value, attr)
            if self._allow_dict_key_attributes and attr in value:
                return value[attr]
        if isinstance(value, str) and attr in _SAFE_STR_METHODS:
            return getattr(value, attr)
        if isinstance(value, list) and attr in _SAFE_LIST_METHODS:
            return getattr(value, attr)
        for allowed_type in self._allowed_attribute_types:
            if isinstance(value, allowed_type) and hasattr(value, attr):
                resolved = getattr(value, attr)
                if inspect.ismethod(resolved) and not _is_safe_callable(resolved, self._extra_callables):
                    raise ExpressionSandboxError("Call to unsupported method")
                return resolved
        raise ExpressionSandboxError(f"Access to attribute '{attr}' is not permitted")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        slice_node = node.slice
        if isinstance(slice_node, ast.Slice):
            lower = self.visit(slice_node.lower) if slice_node.lower is not None else None
            upper = self.visit(slice_node.upper) if slice_node.upper is not None else None
            step = self.visit(slice_node.step) if slice_node.step is not None else None
            return value[slice(lower, upper, step)]
        index = self.visit(slice_node)
        return value[index]

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(element) for element in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(element) for element in node.elts)

    def visit_Set(self, node: ast.Set) -> Any:
        return {self.visit(element) for element in node.elts}

    def visit_Dict(self, node: ast.Dict) -> Any:
        if any(key is None for key in node.keys):
            raise ExpressionSandboxError("Dict unpacking is not permitted")
        return {self.visit(key): self.visit(value) for key, value in zip(node.keys, node.values)}

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - defensive
        raise ExpressionSandboxError(f"Unsupported expression element '{type(node).__name__}'")


def evaluate_sandboxed_expression(
    expression: str,
    scope: Dict[str, Any],
    *,
    extra_callables: Optional[Iterable[Callable[..., Any]]] = None,
    allow_dict_key_attributes: bool = False,
    allowed_attribute_types: Optional[Sequence[type]] = None,
    name_constants: Optional[Dict[str, Any]] = None,
) -> Any:
    evaluator = SandboxedExpressionEvaluator(
        scope,
        extra_callables=extra_callables,
        allow_dict_key_attributes=allow_dict_key_attributes,
        allowed_attribute_types=allowed_attribute_types,
        name_constants=name_constants,
    )
    return evaluator.evaluate(expression)


def evaluate_expression_tree(
    expression: Any,
    scope: Dict[str, Any],
    context: Dict[str, Any],
    *,
    extra_callables: Optional[Iterable[Callable[..., Any]]] = None,
    allow_dict_key_attributes: bool = False,
    allowed_attribute_types: Optional[Sequence[type]] = None,
    name_constants: Optional[Dict[str, Any]] = None,
) -> Any:
    """Evaluate a structured expression previously produced for runtime execution."""

    if expression is None:
        return None
    if isinstance(expression, dict) and expression.get("type"):
        evaluator = _ExpressionSpecEvaluator(
            scope,
            context,
            extra_callables=extra_callables,
            allow_dict_key_attributes=allow_dict_key_attributes,
            allowed_attribute_types=allowed_attribute_types,
            name_constants=name_constants,
        )
        return evaluator.evaluate(expression)
    if isinstance(expression, (list, tuple)):
        evaluator = _ExpressionSpecEvaluator(
            scope,
            context,
            extra_callables=extra_callables,
            allow_dict_key_attributes=allow_dict_key_attributes,
            allowed_attribute_types=allowed_attribute_types,
            name_constants=name_constants,
        )
        return [evaluator.evaluate(item) for item in expression]
    if isinstance(expression, str):
        return evaluate_sandboxed_expression(
            expression,
            scope,
            extra_callables=extra_callables,
            allow_dict_key_attributes=allow_dict_key_attributes,
            allowed_attribute_types=allowed_attribute_types,
            name_constants=name_constants,
        )
    raise ExpressionSandboxError("Unsupported expression representation")


class _ExpressionSpecEvaluator:
    """Evaluate structured expression dictionaries emitted by code generation."""

    def __init__(
        self,
        scope: Dict[str, Any],
        context: Dict[str, Any],
        *,
        extra_callables: Optional[Iterable[Callable[..., Any]]] = None,
        allow_dict_key_attributes: bool = False,
        allowed_attribute_types: Optional[Sequence[type]] = None,
        name_constants: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._scope = scope
        self._context = context
        self._extra_callables: Set[Callable[..., Any]] = set(extra_callables or ())
        self._allow_dict_key_attributes = allow_dict_key_attributes
        if allowed_attribute_types is None:
            self._allowed_attribute_types: Tuple[type, ...] = (SimpleNamespace,)
        else:
            self._allowed_attribute_types = tuple(allowed_attribute_types)
        self._name_constants: Dict[str, Any] = dict(_DEFAULT_NAME_CONSTANTS)
        if name_constants:
            self._name_constants.update(name_constants)

    def evaluate(self, spec: Any) -> Any:
        if spec is None:
            return None
        if isinstance(spec, dict):
            spec_type = spec.get("type")
            if spec_type == "literal":
                return spec.get("value")
            if spec_type == "name":
                return self._resolve_name(spec.get("name"))
            if spec_type == "attribute":
                return self._resolve_attribute(spec.get("path"))
            if spec_type == "context":
                return self._resolve_context(spec)
            if spec_type == "binary":
                return self._evaluate_binary(spec)
            if spec_type == "unary":
                return self._evaluate_unary(spec)
            if spec_type == "call":
                return self._evaluate_call(spec)
            if spec_type == "list":
                return [self.evaluate(item) for item in spec.get("values", [])]
            if spec_type == "dict":
                return {
                    key: self.evaluate(value)
                    for key, value in (spec.get("values") or {}).items()
                }
            raise ExpressionSandboxError(f"Unsupported expression node '{spec_type}'")
        if isinstance(spec, list):
            return [self.evaluate(item) for item in spec]
        if isinstance(spec, str):
            return self._resolve_name(spec)
        return spec

    def _resolve_name(self, name: Optional[str]) -> Any:
        if not name:
            raise ExpressionSandboxError("Name reference is missing identifier")
        if name in self._name_constants:
            return self._name_constants[name]
        if name.startswith("__"):
            raise ExpressionSandboxError(f"Name '{name}' is not permitted")
        if name in self._scope:
            return self._scope[name]
        raise ExpressionSandboxError(f"Unknown name '{name}' in expression")

    def _resolve_attribute(self, path: Optional[List[str]]) -> Any:
        if not path:
            raise ExpressionSandboxError("Attribute path cannot be empty")
        iterator = iter(path)
        base_name = next(iterator)
        value = self._resolve_name(base_name)
        for attr in iterator:
            value = self._access_attribute(value, attr)
        return value

    def _access_attribute(self, value: Any, attr: str) -> Any:
        if isinstance(value, dict):
            if attr in value:
                return value[attr]
            if attr in _SAFE_DICT_METHODS:
                return getattr(value, attr)
            if self._allow_dict_key_attributes and attr in value:
                return value[attr]
        if isinstance(value, str) and attr in _SAFE_STR_METHODS:
            return getattr(value, attr)
        if isinstance(value, list) and attr in _SAFE_LIST_METHODS:
            return getattr(value, attr)
        for allowed in self._allowed_attribute_types:
            if isinstance(value, allowed) and hasattr(value, attr):
                resolved = getattr(value, attr)
                if inspect.ismethod(resolved) and not _is_safe_callable(resolved, self._extra_callables):
                    raise ExpressionSandboxError("Call to unsupported method")
                return resolved
        if hasattr(value, attr) and _is_safe_callable(getattr(value, attr), self._extra_callables):
            return getattr(value, attr)
        raise ExpressionSandboxError(f"Access to attribute '{attr}' is not permitted")

    def _resolve_context(self, spec: Dict[str, Any]) -> Any:
        scope = spec.get("scope") or "ctx"
        path = spec.get("path") or []
        default = spec.get("default")
        if not path:
            if default is not None:
                return default
            raise ExpressionSandboxError("Context path cannot be empty")
        if scope == "env":
            resolver = getattr(self._context, "get_env", None)
            if callable(resolver):
                value = resolver(path[0])
            elif isinstance(self._context, dict):
                mapping: Any = self._context.get("env") if isinstance(self._context.get("env"), dict) else self._context
                value = mapping.get(path[0]) if isinstance(mapping, dict) else None
            else:
                value = None
        else:
            resolver = getattr(self._context, "get_ctx", None)
            if callable(resolver):
                value = resolver(path)
            else:
                if isinstance(self._context, dict):
                    current: Any = self._context.get("ctx", self._context)
                else:
                    current = self._context
                value = current
                for segment in path:
                    if isinstance(value, dict):
                        value = value.get(segment)
                    else:
                        value = getattr(value, segment, None)
                    if value is None:
                        break
        if value is None:
            return default
        return value

    def _evaluate_binary(self, spec: Dict[str, Any]) -> Any:
        op = (spec.get("op") or "").lower()
        if op == "and":
            left = self.evaluate(spec.get("left"))
            if not left:
                return left
            return self.evaluate(spec.get("right"))
        if op == "or":
            left = self.evaluate(spec.get("left"))
            if left:
                return left
            return self.evaluate(spec.get("right"))

        left_value = self.evaluate(spec.get("left"))
        right_value = self.evaluate(spec.get("right"))

        if op in {"==", "="}:
            return left_value == right_value
        if op in {"!=", "<>"}:
            return left_value != right_value
        if op == "<":
            return left_value < right_value
        if op == "<=":
            return left_value <= right_value
        if op == ">":
            return left_value > right_value
        if op == ">=":
            return left_value >= right_value
        if op == "+":
            return left_value + right_value
        if op == "-":
            return left_value - right_value
        if op == "*":
            return left_value * right_value
        if op == "/":
            return left_value / right_value
        if op == "%":
            return left_value % right_value
        if op == "in":
            try:
                return left_value in right_value  # type: ignore[operator]
            except TypeError:
                return False
        if op == "like":
            return self._match_like(left_value, right_value, case_insensitive=False)
        if op == "ilike":
            return self._match_like(left_value, right_value, case_insensitive=True)
        raise ExpressionSandboxError(f"Unsupported binary operator '{op}'")

    def _evaluate_unary(self, spec: Dict[str, Any]) -> Any:
        op = (spec.get("op") or "").lower()
        operand = self.evaluate(spec.get("operand"))
        if op == "not":
            return not operand
        if op == "-":
            return -operand
        if op == "+":
            return +operand
        raise ExpressionSandboxError(f"Unsupported unary operator '{op}'")

    def _evaluate_call(self, spec: Dict[str, Any]) -> Any:
        function = self.evaluate(spec.get("function"))
        if not callable(function):
            raise ExpressionSandboxError("Call target is not callable")
        if not _is_safe_callable(function, self._extra_callables):
            raise ExpressionSandboxError("Call to unsupported function")
        args = [self.evaluate(arg) for arg in spec.get("arguments", [])]
        return function(*args)

    def _match_like(self, value: Any, pattern: Any, *, case_insensitive: bool) -> bool:
        if pattern is None:
            return False
        pattern_text = str(pattern)
        value_text = "" if value is None else str(value)
        if case_insensitive:
            pattern_text = pattern_text.lower()
            value_text = value_text.lower()
        regex = re.escape(pattern_text)
        regex = regex.replace(r"\%", ".*").replace(r"\_", ".")
        return re.fullmatch(regex, value_text, flags=re.DOTALL) is not None
