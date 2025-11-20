"""Tool and plugin system for LLM workflows."""

from textwrap import dedent

# Tools content extracted from lines 372-550 of original llm.py
TOOLS = dedent(
    '''
def _tool_schema(plugin: Any, attr: str) -> Optional[Dict[str, Any]]:
    schema = getattr(plugin, attr, None)
    return dict(schema) if isinstance(schema, dict) else None


def _matches_schema_type(value: Any, expected: str) -> bool:
    normalized = expected.strip().lower()
    type_info = _SCHEMA_TYPE_MAPPING.get(normalized)
    if type_info is None:
        return True
    return isinstance(value, type_info)


def _validate_tool_payload(plugin: Any, payload: Dict[str, Any], chain_name: str, step_label: str) -> None:
    schema = _tool_schema(plugin, "input_schema")
    if not schema:
        return
    for field, rule in schema.items():
        details = rule if isinstance(rule, dict) else {"type": str(rule)}
        required = bool(details.get("required"))
        expected_type = str(details.get("type") or "").strip()
        if required and field not in payload:
            raise ValueError(f"Tool step '{step_label}' in chain '{chain_name}' is missing required field '{field}'.")
        if expected_type and field in payload and not _matches_schema_type(payload[field], expected_type):
            raise TypeError(
                f"Field '{field}' in tool step '{step_label}' does not match expected type '{expected_type}'."
            )


def _await_plugin_result(result: Any) -> Any:
    if inspect.isawaitable(result):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop and running_loop.is_running():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(result)
            finally:
                loop.close()
        return asyncio.run(result)
    return result


def _call_tool_plugin(plugin: Any, context: Dict[str, Any], payload: Dict[str, Any]) -> Any:
    result = plugin.call(context, payload)
    return _await_plugin_result(result)


_VIOLATION_LABELS = {"violation", "unsafe", "toxic", "blocked"}


def _evaluation_result_is_violation(result: Any) -> bool:
    if isinstance(result, dict):
        if bool(result.get("violation")):
            return True
        label = result.get("label")
        if isinstance(label, str) and label.strip().lower() in _VIOLATION_LABELS:
            return True
    return False


def _run_step_evaluations(
    config: Dict[str, Any],
    entry: Dict[str, Any],
    output: Any,
    context: Dict[str, Any],
    chain_name: str,
    step_label: str,
) -> Dict[str, Any]:
    evaluator_names = config.get("evaluators") or []
    if not evaluator_names:
        return {}
    plugins = context.get("evaluators") if isinstance(context.get("evaluators"), dict) else {}
    evaluation_results: Dict[str, Any] = {}
    for evaluator_name in evaluator_names:
        plugin = plugins.get(evaluator_name)
        if plugin is None:
            message = f"Evaluator '{evaluator_name}' is not configured."
            _record_runtime_error(
                context,
                code="evaluation_missing",
                message=message,
                scope=f"chain:{chain_name}",
                detail=f"step:{step_label}",
            )
            raise RuntimeError(message)
        payload = {
            "output": output,
            "step": step_label,
            "chain": chain_name,
        }
        try:
            evaluation_results[evaluator_name] = _await_plugin_result(plugin.call(context, payload))
        except Exception as exc:
            message = f"Evaluator '{evaluator_name}' failed: {exc}"
            _record_runtime_error(
                context,
                code="evaluation_failed",
                message=message,
                scope=f"chain:{chain_name}",
                detail=f"step:{step_label}",
            )
            raise
    entry["evaluation"] = evaluation_results
    return evaluation_results


def _apply_guardrail(
    guardrail_name: str,
    evaluation_results: Dict[str, Any],
    output: Any,
    entry: Dict[str, Any],
    chain_name: str,
    step_label: str,
    context: Dict[str, Any],
) -> Any:
    guardrail = GUARDRAILS.get(guardrail_name, {})
    if not guardrail:
        message = f"Guardrail '{guardrail_name}' is not defined."
        _record_runtime_error(
            context,
            code="guardrail_missing",
            message=message,
            scope=f"chain:{chain_name}",
            detail=f"step:{step_label}",
        )
        raise RuntimeError(message)
    evaluator_names = guardrail.get("evaluators") or []
    violation = False
    for evaluator_name in evaluator_names:
        result = evaluation_results.get(evaluator_name)
        if result is None:
            continue
        if _evaluation_result_is_violation(result):
            violation = True
            break
    entry["guardrail"] = {
        "name": guardrail_name,
        "action": guardrail.get("action"),
        "violated": violation,
    }
    if not violation:
        return output
    action = str(guardrail.get("action") or "block").lower()
    message = guardrail.get("message") or f"Guardrail '{guardrail_name}' blocked step '{step_label}'."
    data = {
        "guardrail": guardrail_name,
        "chain": chain_name,
        "step": step_label,
    }
    if action == "log_only":
        _record_runtime_event(
            context,
            event="guardrail_violation",
            level="warning",
            message=message,
            data=data,
        )
        return output
    _record_runtime_error(
        context,
        code="guardrail_violation",
        message=message,
        scope=f"chain:{chain_name}",
        detail=f"step:{step_label}",
    )
    blocked_payload = {
        "status": "blocked",
        "message": message,
        "guardrail": guardrail_name,
    }
    entry["output"] = blocked_payload
    entry["status"] = "error"
    return blocked_payload
'''
).strip()

__all__ = ['TOOLS']
