"""Statement and component encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .actions import _encode_action_operation
from .expressions import _encode_value, _expression_to_runtime, _expression_to_source
from .pages import _encode_layout_meta, _encode_layout_spec

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
        ContinueStatement,
        ForLoop,
        IfBlock,
        LogStatement,
        PredictStatement,
        ShowChart,
        ShowForm,
        ShowTable,
        ShowText,
        VariableAssignment,
        WhileLoop,
    )
    from .classes import PageComponent

    if isinstance(statement, ShowText):
        payload = {
            "text": statement.text,
            "styles": dict(statement.styles or {}),
        }
        return PageComponent(type="text", payload=payload)

    if isinstance(statement, ShowTable):
        payload: Dict[str, Any] = {
            "title": statement.title,
            "source_kind": statement.source_type,
            "source_name": statement.source,
            "columns": list(statement.columns or []),
            "filter": statement.filter_by,
            "sort": statement.sort_by,
            "styles": dict(statement.style or {}),
            "insight": statement.insight,
            "dynamic_columns": dict(statement.dynamic_columns or {}),
        }
        if statement.layout:
            payload["layout"] = _encode_layout_meta(statement.layout)
        return PageComponent(type="table", payload={k: v for k, v in payload.items() if v not in (None, [], {})})

    if isinstance(statement, ShowChart):
        payload: Dict[str, Any] = {
            "heading": statement.heading,
            "source_kind": statement.source_type,
            "source_name": statement.source,
            "chart_type": statement.chart_type,
            "x": statement.x,
            "y": statement.y,
            "color": statement.color,
            "encodings": dict(statement.encodings or {}),
            "style": dict(statement.style or {}),
            "title": statement.title,
            "legend": dict(statement.legend or {}) if statement.legend else None,
            "insight": statement.insight,
        }
        if statement.layout:
            payload["layout"] = _encode_layout_meta(statement.layout)
        return PageComponent(type="chart", payload={k: v for k, v in payload.items() if v not in (None, [], {})})

    if isinstance(statement, ShowForm):
        # Serialize fields with full validation and component information
        fields_payload = []
        for field in statement.fields:
            field_data = {
                "name": field.name,
                "component": field.component or "text_input",
                "label": field.label or field.name,
                "type": field.field_type,  # Backward compatibility
            }
            if field.placeholder:
                field_data["placeholder"] = field.placeholder
            if field.help_text:
                field_data["help_text"] = field.help_text
            if field.required:
                field_data["required"] = field.required
            if field.default:
                field_data["default"] = str(field.default)
            
            # Validation
            validation = {}
            if field.min_length is not None:
                validation["min_length"] = field.min_length
            if field.max_length is not None:
                validation["max_length"] = field.max_length
            if field.pattern:
                validation["pattern"] = field.pattern
            if field.min_value is not None:
                validation["min_value"] = field.min_value
            if field.max_value is not None:
                validation["max_value"] = field.max_value
            if field.step is not None:
                validation["step"] = field.step
            if validation:
                field_data["validation"] = validation
            
            # Options for select/multiselect/radio
            if field.options_binding:
                field_data["options_binding"] = field.options_binding
            if field.options:
                field_data["options"] = field.options
            
            # Conditional rendering
            if field.disabled:
                field_data["disabled"] = str(field.disabled)
            if field.visible:
                field_data["visible"] = str(field.visible)
            
            # File upload
            if field.component == "file_upload":
                if field.accept:
                    field_data["accept"] = field.accept
                if field.max_file_size:
                    field_data["max_file_size"] = field.max_file_size
                if field.upload_endpoint:
                    field_data["upload_endpoint"] = field.upload_endpoint
                if field.multiple:
                    field_data["multiple"] = field.multiple
            
            fields_payload.append(field_data)
        
        operations = [_encode_action_operation(op, env_keys, prompt_lookup) for op in statement.on_submit_ops]
        payload: Dict[str, Any] = {
            "title": statement.title,
            "fields": fields_payload,
            "operations": operations,
            "layout_mode": statement.layout_mode or "vertical",
            "submit_action": statement.submit_action,
            "validation_mode": statement.validation_mode or "on_blur",
            "submit_button_text": statement.submit_button_text or "Submit",
            "reset_button": statement.reset_button,
            "success_message": statement.success_message,
            "error_message": statement.error_message,
            "styles": dict(statement.styles or {}),
            "layout": _encode_layout_spec(statement.layout) if statement.layout else {},
            "effects": sorted(statement.effects) if statement.effects else [],
        }
        return PageComponent(type="form", payload={k: v for k, v in payload.items() if v not in (None, [], {}, False)})
    
    if isinstance(statement, Action):
        operations = [_encode_action_operation(op, env_keys, prompt_lookup) for op in statement.operations]
        payload = {
            "name": statement.name,
            "trigger": statement.trigger,
            "operations": operations,
            "effects": sorted(statement.effects) if statement.effects else [],
            "declared_effect": statement.declared_effect,
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
    
    if isinstance(statement, LogStatement):
        payload = {
            "level": str(statement.level).lower(),  # Convert LogLevel enum to lowercase string
            "message": statement.message,
            "source_location": statement.source_location,
        }
        return PageComponent(type="log", payload=payload)
    
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
