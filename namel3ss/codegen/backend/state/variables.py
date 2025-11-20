"""Variable assignment encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, Set, TYPE_CHECKING

from .expressions import _encode_frame_expression_value, _encode_value, _expression_to_runtime, _expression_to_source

if TYPE_CHECKING:
    from ....ast import FrameExpression, VariableAssignment


def _encode_variable(variable: "VariableAssignment", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a variable assignment for backend state."""
    from ....ast import FrameExpression
    
    if isinstance(variable.value, FrameExpression):
        pipeline_payload = _encode_frame_expression_value(variable.value, env_keys)
        return {
            "name": variable.name,
            "value": {"__frame_pipeline__": pipeline_payload},
            "value_source": pipeline_payload.get("root"),
            "value_expr": None,
            "frame_pipeline": pipeline_payload,
        }
    return {
        "name": variable.name,
        "value": _encode_value(variable.value, env_keys),
        "value_source": _expression_to_source(variable.value),
        "value_expr": _expression_to_runtime(variable.value),
    }
