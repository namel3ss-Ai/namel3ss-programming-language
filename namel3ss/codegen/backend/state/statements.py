"""Statement and component encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .actions import _encode_action_operation
from .expressions import _encode_value, _expression_to_runtime, _expression_to_source
from .pages import _encode_layout_spec

if TYPE_CHECKING:
    from ....ast import PageStatement, Prompt
    from .classes import PageComponent


def _encode_statement(
    statement: "PageStatement",
    env_keys: Set[str],
    prompt_lookup: Dict[str, "Prompt"],
) -> Optional["PageComponent"]:
    """Encode a page statement into a PageComponent."""
    from ....ast import (
        Action,
        BreakStatement,
        Chart,
        ContinueStatement,
        Display,
        ForLoop,
        Form,
        IfBlock,
        PredictStatement,
        VariableAssignment,
        WhileLoop,
    )
    from .classes import PageComponent
    
    if isinstance(statement, Display):
        payload = {
            "source_kind": statement.source_kind,
            "source_name": statement.source_name,
            "select_fields": list(statement.select_fields or []),
            "styles": dict(statement.styles or {}),
        }
        return PageComponent(type="display", payload=payload)
    
    if isinstance(statement, Chart):
        payload = {
            "kind": statement.kind,
            "source_kind": statement.source_kind,
            "source_name": statement.source_name,
            "x_axis": statement.x_axis,
            "y_axis": statement.y_axis,
            "color": statement.color,
            "filters": _encode_value(statement.filters, env_keys),
            "styles": dict(statement.styles or {}),
        }
        return PageComponent(type="chart", payload=payload)
    
    if isinstance(statement, Form):
        payload = {
            "name": statement.name,
            "source_kind": statement.source_kind,
            "source_name": statement.source_name,
            "fields": list(statement.fields or []),
            "layout": _encode_layout_spec(statement.layout),
            "operations": [_encode_action_operation(op, env_keys, prompt_lookup) for op in statement.on_submit_ops],
            "styles": dict(statement.styles),
        }
        return PageComponent(type="form", payload=payload)
    
    if isinstance(statement, Action):
        payload = {
            "name": statement.name,
            "trigger": statement.trigger,
            "operations": [_encode_action_operation(op, env_keys, prompt_lookup) for op in statement.operations],
        }
        return PageComponent(type="action", payload=payload)
    
    if isinstance(statement, VariableAssignment):
        payload = {
            "name": statement.name,
            "value": _encode_value(statement.value, env_keys),
            "value_source": _expression_to_source(statement.value),
            "value_expr": _expression_to_runtime(statement.value),
        }
        return PageComponent(type="variable", payload=payload)
    
    if isinstance(statement, IfBlock):
        body_payload: List[Dict[str, Any]] = []
        for stmt in statement.body:
            encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
            if encoded:
                body_payload.append(encoded)
        elif_payload: List[Dict[str, Any]] = []
        for branch in statement.elifs:
            branch_body: List[Dict[str, Any]] = []
            for stmt in branch.body:
                encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
                if encoded:
                    branch_body.append(encoded)
            elif_payload.append({
                "condition": _expression_to_runtime(branch.condition),
                "body": branch_body,
            })
        else_payload: List[Dict[str, Any]] = []
        for stmt in statement.else_body or []:
            encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
            if encoded:
                else_payload.append(encoded)
        payload = {
            "condition": _expression_to_runtime(statement.condition),
            "body": body_payload,
            "elifs": elif_payload,
            "else_body": else_payload,
        }
        return PageComponent(type="if", payload=payload)
    
    if isinstance(statement, ForLoop):
        loop_body: List[Dict[str, Any]] = []
        for stmt in statement.body:
            encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
            if encoded:
                loop_body.append(encoded)
        payload = {
            "loop_var": statement.loop_var,
            "source_kind": statement.source_kind,
            "source_name": statement.source_name,
            "body": loop_body,
        }
        return PageComponent(type="for_loop", payload=payload)
    
    if isinstance(statement, WhileLoop):
        loop_body: List[Dict[str, Any]] = []
        for stmt in statement.body:
            encoded = _encode_statement_dict(stmt, env_keys, prompt_lookup)
            if encoded:
                loop_body.append(encoded)
        payload = {
            "condition": _expression_to_runtime(statement.condition),
            "body": loop_body,
        }
        return PageComponent(type="while_loop", payload=payload)
    
    if isinstance(statement, BreakStatement):
        return PageComponent(type="break", payload={})
    
    if isinstance(statement, ContinueStatement):
        return PageComponent(type="continue", payload={})
    
    if isinstance(statement, PredictStatement):
        payload = {
            "model_name": statement.model_name,
            "input_kind": statement.input_kind,
            "input_ref": statement.input_ref,
            "assign": _encode_value(statement.assign, env_keys),
            "parameters": _encode_value(statement.parameters, env_keys),
        }
        return PageComponent(type="predict", payload=payload)
    
    return None


def _encode_statement_dict(
    statement: "PageStatement",
    env_keys: Set[str],
    prompt_lookup: Dict[str, "Prompt"],
) -> Optional[Dict[str, Any]]:
    """Encode a page statement into a serializable dictionary."""
    from .utils import _component_to_serializable
    
    component = _encode_statement(statement, env_keys, prompt_lookup)
    if component is None:
        return None
    return _component_to_serializable(component)
