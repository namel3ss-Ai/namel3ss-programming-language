"""Action operation encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, Set, TYPE_CHECKING

from .expressions import _encode_value

if TYPE_CHECKING:
    from ....ast import ActionOperation as ActionOperationType, Prompt


def _encode_action_operation(
    operation: "ActionOperationType",
    env_keys: Set[str],
    prompt_lookup: Dict[str, "Prompt"],
) -> Dict[str, Any]:
    """Encode an action operation for backend state."""
    from ....ast import (
        ActionOperation,
        AskConnectorOperation,
        CallPythonOperation,
        GoToPageOperation,
        RunChainOperation,
        RunPromptOperation,
        ToastOperation,
        UpdateOperation,
    )
    from .utils import _validate_prompt_arguments
    
    if isinstance(operation, UpdateOperation):
        return {
            "type": "update",
            "table": operation.table,
            "set_expression": operation.set_expression,
            "where_expression": operation.where_expression,
        }
    if isinstance(operation, ToastOperation):
        return {"type": "toast", "message": operation.message}
    if isinstance(operation, GoToPageOperation):
        return {
            "type": "navigate",
            "page_name": operation.page_name,
        }
    if isinstance(operation, CallPythonOperation):
        return {
            "type": "python_call",
            "module": operation.module,
            "method": operation.method,
            "arguments": {
                key: _encode_value(value, env_keys) for key, value in operation.arguments.items()
            },
        }
    if isinstance(operation, AskConnectorOperation):
        return {
            "type": "connector_call",
            "name": operation.connector_name,
            "arguments": {
                key: _encode_value(value, env_keys) for key, value in operation.arguments.items()
            },
        }
    if isinstance(operation, RunChainOperation):
        return {
            "type": "chain_run",
            "name": operation.chain_name,
            "inputs": {
                key: _encode_value(value, env_keys) for key, value in operation.inputs.items()
            },
        }
    if isinstance(operation, RunPromptOperation):
        _validate_prompt_arguments(prompt_lookup, operation.prompt_name, operation.arguments)
        return {
            "type": "prompt_call",
            "prompt": operation.prompt_name,
            "arguments": {
                key: _encode_value(value, env_keys) for key, value in operation.arguments.items()
            },
        }
    if isinstance(operation, ActionOperation):
        return {"type": type(operation).__name__}
    return {"type": "operation"}
