"""Expression encoding utilities for backend state translation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ....ast import (
        AttributeRef,
        BinaryOp,
        CallExpression,
        ContextValue,
        Expression,
        FrameExpression,
        Literal,
        NameRef,
        UnaryOp,
    )
    from ....frames import FrameExpressionAnalyzer


_FRAME_ANALYZER: Optional["FrameExpressionAnalyzer"] = None
_TEMPLATE_PATTERN = re.compile(r"\{([^{}]+)\}")


def set_frame_analyzer(analyzer: Optional["FrameExpressionAnalyzer"]) -> None:
    """Set the global frame analyzer for expression encoding."""
    global _FRAME_ANALYZER
    _FRAME_ANALYZER = analyzer


def _expression_to_source(expression: Optional["Expression"]) -> Optional[str]:
    """Convert an expression AST node to its source code representation."""
    from ....ast import (
        AttributeRef,
        BinaryOp,
        CallExpression,
        ContextValue,
        Literal,
        NameRef,
        UnaryOp,
        InlinePythonBlock,
        InlineReactBlock,
    )
    
    if expression is None:
        return None
    if isinstance(expression, Literal):
        return repr(expression.value)
    if isinstance(expression, NameRef):
        return expression.name
    if isinstance(expression, AttributeRef):
        return f"{expression.base}.{expression.attr}"
    if isinstance(expression, BinaryOp):
        left = _expression_to_source(expression.left) or ""
        right = _expression_to_source(expression.right) or ""
        return f"{left} {expression.op} {right}".strip()
    if isinstance(expression, UnaryOp):
        operand = _expression_to_source(expression.operand) or ""
        return f"{expression.op}{operand}".strip()
    if isinstance(expression, CallExpression):
        func = _expression_to_source(expression.function) or "call"
        args = ", ".join(
            arg for arg in [_expression_to_source(arg) for arg in expression.arguments] if arg is not None
        )
        return f"{func}({args})"
    if isinstance(expression, ContextValue):
        return None
    if isinstance(expression, InlinePythonBlock):
        # Return inline Python as reference to generated function
        return f"inline_python_{id(expression)}"
    if isinstance(expression, InlineReactBlock):
        # Return inline React as reference to generated component
        return f"inline_react_{id(expression)}"
    return str(expression)


def _expression_to_runtime(expression: Optional["Expression"]) -> Optional[Dict[str, Any]]:
    """Convert an expression AST node to runtime representation."""
    from ....ast import (
        AttributeRef,
        BinaryOp,
        CallExpression,
        ContextValue,
        Literal,
        NameRef,
        UnaryOp,
        InlinePythonBlock,
        InlineReactBlock,
    )
    
    if expression is None:
        return None
    if isinstance(expression, Literal):
        return {"type": "literal", "value": expression.value}
    if isinstance(expression, NameRef):
        return {"type": "name", "name": expression.name}
    if isinstance(expression, AttributeRef):
        path: List[str] = []
        if expression.base:
            path.extend(segment for segment in expression.base.split(".") if segment)
        if expression.attr:
            path.append(expression.attr)
        return {"type": "attribute", "path": path}
    if isinstance(expression, ContextValue):
        return {
            "type": "context",
            "scope": expression.scope,
            "path": list(expression.path),
            "default": expression.default,
        }
    if isinstance(expression, BinaryOp):
        return {
            "type": "binary",
            "op": expression.op,
            "left": _expression_to_runtime(expression.left),
            "right": _expression_to_runtime(expression.right),
        }
    if isinstance(expression, UnaryOp):
        return {
            "type": "unary",
            "op": expression.op,
            "operand": _expression_to_runtime(expression.operand),
        }
    if isinstance(expression, CallExpression):
        return {
            "type": "call",
            "function": _expression_to_runtime(expression.function),
            "arguments": [
                _expression_to_runtime(arg) for arg in expression.arguments
            ],
        }
    if isinstance(expression, InlinePythonBlock):
        return {
            "type": "inline_python",
            "code": expression.code,
            "block_id": id(expression),
            "bindings": expression.bindings,
        }
    if isinstance(expression, InlineReactBlock):
        return {
            "type": "inline_react",
            "code": expression.code,
            "block_id": id(expression),
            "component_name": expression.component_name,
            "props": expression.props,
        }
    return {"type": "literal", "value": _expression_to_source(expression)}


def _encode_frame_expression_value(expression: "FrameExpression", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a frame expression using the frame analyzer."""
    from ....frames import FrameTypeError
    
    if _FRAME_ANALYZER is None:
        raise FrameTypeError("Frame expressions require frame analyzer context during encoding")
    plan = _FRAME_ANALYZER.analyze(expression)
    return plan.to_payload(_expression_to_runtime, _expression_to_source)


def _encode_value(value: Any, env_keys: Set[str]) -> Any:
    """Encode any value to backend-friendly representation."""
    from ....ast import (
        AttributeRef,
        BinaryOp,
        CallExpression,
        ContextValue,
        FrameExpression,
        Literal,
        NameRef,
        UnaryOp,
    )
    
    if isinstance(value, ContextValue):
        marker = {
            "__context__": {
                "scope": value.scope,
                "path": list(value.path),
            }
        }
        if value.default is not None:
            marker["__context__"]["default"] = value.default
        if value.scope == "env" and value.path:
            env_keys.add(value.path[0])
        return marker
    if isinstance(value, Literal):
        return value.value
    if isinstance(value, FrameExpression):
        pipeline_payload = _encode_frame_expression_value(value, env_keys)
        return {"__frame_pipeline__": pipeline_payload}
    if isinstance(value, (NameRef, AttributeRef, BinaryOp, UnaryOp, CallExpression)):
        return _expression_to_source(value)
    if isinstance(value, list):
        return [_encode_value(item, env_keys) for item in value]
    if isinstance(value, dict):
        return {key: _encode_value(val, env_keys) for key, val in value.items()}
    if hasattr(value, "__dict__"):
        return _encode_value(value.__dict__, env_keys)
    return value


def _collect_template_markers(value: Any, env_keys: Set[str]) -> None:
    """Collect environment variable markers from template strings."""
    if isinstance(value, str):
        for match in _TEMPLATE_PATTERN.finditer(value):
            token = match.group(1).strip()
            if not token:
                continue
            if token.startswith("$"):
                env_keys.add(token[1:])
                continue


def _encode_metadata_dict(raw_metadata: Any, env_keys: Set[str]) -> Dict[str, Any]:
    """Encode metadata ensuring it's a dictionary."""
    metadata_value = _encode_value(raw_metadata, env_keys)
    if not isinstance(metadata_value, dict):
        metadata_value = {"value": metadata_value} if metadata_value is not None else {}
    return metadata_value
