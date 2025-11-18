from __future__ import annotations

from textwrap import dedent

LLM_SECTION = dedent(
    '''
import asyncio
import copy
import inspect
import json
import os
import time
from typing import Any, Awaitable, Dict, List, Optional, Tuple

from namel3ss.codegen.backend.core.runtime.expression_sandbox import (
    evaluate_expression_tree as _evaluate_expression_tree,
)


def _stringify_prompt_value(name: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, separators=(",", ":"))
        except Exception:
            return str(value)
    if value is None:
        return ""
    return str(value).strip()


def _format_error_message(name: str, prompt_text: Optional[str], reason: str) -> str:
    """Compose a concise diagnostic string for logging failures."""

    title = f"llm:{name}" if name else "llm:unknown"
    cause = reason.strip() or "unspecified error"
    return f"{title} failed: {cause}"


def _default_llm_endpoint(provider: str, config: Dict[str, Any]) -> Optional[str]:
    endpoint = config.get("endpoint") or config.get("url")
    if endpoint:
        return str(endpoint)
    provider_key = provider.lower()
    base_url = str(config.get("api_base") or config.get("base_url") or "")
    if provider_key == "openai":
        api_base = base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        deployment = config.get("deployment")
        if deployment:
            version = str(config.get("api_version") or os.getenv("OPENAI_API_VERSION") or "2024-02-01")
            return f"{api_base.rstrip('/')}/deployments/{deployment}/chat/completions?api-version={version}"
        return f"{api_base.rstrip('/')}/chat/completions"
    if provider_key == "anthropic":
        api_base = base_url or "https://api.anthropic.com/v1"
        return f"{api_base.rstrip('/')}/messages"
    return None


def _build_llm_request(
    provider: str,
    model_name: str,
    prompt_text: str,
    config: Dict[str, Any],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    if not prompt_text:
        raise ValueError("Prompt text is required for LLM requests")

    provider_key = provider.lower().strip()
    method = str(config.get("method") or "post").upper()
    timeout_value = config.get("timeout", 30.0)
    try:
        timeout = max(float(timeout_value), 1.0)
    except Exception:
        timeout = 30.0

    headers = _ensure_dict(config.get("headers"))
    params = _ensure_dict(config.get("params"))
    payload = config.get("payload")
    body = dict(payload) if isinstance(payload, dict) else {}

    mode = str(config.get("mode") or ("chat" if provider_key in {"openai", "anthropic"} else "completion")).lower()
    if mode == "chat":
        messages: List[Dict[str, Any]] = []
        system_prompt = config.get("system") or config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        extra_messages = config.get("messages")
        if isinstance(extra_messages, list):
            for message in extra_messages:
                if isinstance(message, dict):
                    messages.append(dict(message))
        user_role = str(config.get("user_role") or "user")
        messages.append({"role": user_role, "content": prompt_text})
        body.setdefault("messages", messages)
        body.setdefault("model", model_name)
    else:
        prompt_field = str(config.get("prompt_field") or "prompt")
        body.setdefault("model", model_name)
        body.setdefault(prompt_field, prompt_text)

    payload_from_args = config.get("payload_from_args")
    if isinstance(payload_from_args, (list, tuple)):
        for key in payload_from_args:
            key_str = str(key)
            if key_str in args and key_str not in body:
                body[key_str] = args[key_str]

    endpoint = _default_llm_endpoint(provider_key, config)
    if not endpoint:
        raise ValueError(f"Endpoint is not configured for provider '{provider}'")

    api_key = config.get("api_key")
    if not api_key:
        env_key = config.get("api_key_env")
        if isinstance(env_key, str):
            api_key = os.getenv(env_key) or api_key

    if provider_key == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is missing")
        bearer = str(api_key)
        if not bearer.lower().startswith("bearer "):
            bearer = f"Bearer {bearer}"
        headers.setdefault("Authorization", bearer)
        headers.setdefault("Content-Type", "application/json")
    elif provider_key == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is missing")
        headers.setdefault("x-api-key", str(api_key))
        headers.setdefault("content-type", "application/json")
        headers.setdefault("anthropic-version", str(config.get("api_version") or "2023-06-01"))
    elif api_key:
        header_name = str(config.get("api_key_header") or "Authorization")
        if header_name.lower() == "authorization" and not str(api_key).lower().startswith("bearer "):
            headers.setdefault("Authorization", f"Bearer {api_key}")
        else:
            headers.setdefault(header_name, str(api_key))

    normalized_headers: Dict[str, str] = {
        str(key): str(value)
        for key, value in headers.items()
    }
    normalized_params: Dict[str, Any] = {
        str(key): value
        for key, value in params.items()
    }

    normalized_headers.setdefault("Content-Type", "application/json")

    return {
        "method": method,
        "url": str(endpoint),
        "headers": normalized_headers,
        "params": normalized_params,
        "body": body,
        "timeout": timeout,
        "provider": provider,
    }


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    sensitive_keys = {
        "api_key",
        "authorization",
        "token",
        "secret",
        "password",
        "x-api-key",
        "api-key",
    }

    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): ("***" if str(key).strip().lower() in sensitive_keys else _sanitize(val))
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        return value

    return _sanitize(dict(config)) if isinstance(config, dict) else {}


def _http_post_json(
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> tuple[int, str, Optional[Any]]:
    import json as _json
    import urllib.request
    import random

    request_headers = {
        str(key): str(value)
        for key, value in headers.items()
    }
    request_headers.setdefault("Content-Type", "application/json")

    try:
        import httpx as _httpx  # type: ignore
    except Exception:  # pragma: no cover - optional dependency guard
        _httpx = None  # type: ignore

    if _httpx is not None:
        # Get connector config from runtime settings for retry behavior
        connector_settings = RUNTIME_SETTINGS.get("connectors", {})
        retry_max_attempts = connector_settings.get("retry_max_attempts", 3)
        retry_base_delay = connector_settings.get("retry_base_delay", 0.5)
        retry_max_delay = connector_settings.get("retry_max_delay", 5.0)
        
        client_kwargs: Dict[str, Any] = {}
        try:
            client_kwargs["timeout"] = _httpx.Timeout(timeout)
        except Exception:
            client_kwargs["timeout"] = timeout
            
        last_error: Optional[Exception] = None
        for attempt in range(1, retry_max_attempts + 1):
            try:
                with _httpx.Client(**client_kwargs) as client:
                    response = client.request(
                        "POST",
                        url,
                        json=data,
                        headers=request_headers,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    status_code = int(getattr(response, "status_code", 0))
                    parsed: Optional[Any]
                    try:
                        parsed = response.json()
                    except Exception:
                        parsed = None
                    text = ""
                    raw_text_candidate = getattr(response, "text", "")
                    if isinstance(raw_text_candidate, str) and raw_text_candidate:
                        text = raw_text_candidate
                    elif parsed is not None:
                        try:
                            text = _json.dumps(parsed)
                        except Exception:
                            text = ""
                    else:
                        raw_bytes = getattr(response, "content", b"")
                        if isinstance(raw_bytes, (bytes, bytearray)):
                            text = bytes(raw_bytes).decode("utf-8", "replace")
                    return status_code, text, parsed
            except Exception as exc:
                last_error = exc
                if attempt < retry_max_attempts:
                    # Calculate exponential backoff with jitter
                    delay = min(retry_base_delay * (2 ** (attempt - 1)), retry_max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    import time
                    time.sleep(delay + jitter)
                else:
                    raise last_error from exc

    payload_bytes = _json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload_bytes, headers=request_headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        status_code = getattr(response, "status", None)
        if status_code is None:
            status_code = response.getcode()
        raw_bytes = response.read()

    text = raw_bytes.decode("utf-8", "replace")
    try:
        parsed = _json.loads(text)
    except Exception:
        parsed = None
    return int(status_code or 0), text, parsed


def _extract_llm_text(
    provider: str,
    payload: Any,
    config: Dict[str, Any],
) -> str:
    response_path = config.get("response_path") or config.get("result_path")
    if response_path:
        value = _traverse_attribute_path(payload, _as_path_segments(response_path))
        if value is not None:
            return str(value)
    provider_key = provider.lower()
    if isinstance(payload, dict):
        if provider_key == "openai":
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, list):
                            segments: List[str] = []
                            for entry in content:
                                if isinstance(entry, dict) and "text" in entry:
                                    segments.append(str(entry["text"]))
                                else:
                                    segments.append(str(entry))
                            joined = "\\n".join(segment for segment in segments if segment)
                            if joined:
                                return joined
                        if content is not None:
                            return str(content)
                    if "text" in first_choice:
                        return str(first_choice["text"])
        if provider_key == "anthropic":
            content = payload.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    if "text" in first:
                        return str(first["text"])
                    if "value" in first:
                        return str(first["value"])
            if isinstance(content, str):
                return content
        if "result" in payload and isinstance(payload["result"], str):
            return payload["result"]
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _extract_memory_names(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        names: List[str] = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
        return names
    return []


_SCHEMA_TYPE_MAPPING = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": (list, tuple),
}


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


def _execute_workflow_nodes(
    nodes: List[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    status = "partial"
    result_value: Any = None
    current_value = working
    encountered_error = False
    for node in nodes or []:
        node_type = str(node.get("type") or "step").lower()
        if node_type == "step":
            step_status, current_value, step_result, stop_on_error = _execute_workflow_step(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if step_status == "error":
                encountered_error = True
                if stop_on_error:
                    return "error", step_result, current_value
            elif step_status == "ok":
                status = "ok"
                result_value = step_result
            elif step_status == "stub" and status != "error" and result_value is None:
                result_value = step_result
        elif node_type == "if":
            branch_status, branch_result, branch_value = _execute_workflow_if(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if branch_status == "error":
                return "error", branch_result, branch_value
            if branch_status == "ok" and branch_result is not None:
                status = "ok"
                result_value = branch_result
            current_value = branch_value
        elif node_type == "for":
            loop_status, loop_result, loop_value = _execute_workflow_for(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if loop_status == "error":
                return "error", loop_result, loop_value
            if loop_status == "ok" and loop_result is not None:
                status = "ok"
                result_value = loop_result
            current_value = loop_value
        elif node_type == "while":
            while_status, while_result, while_value = _execute_workflow_while(
                node,
                context=context,
                chain_scope=chain_scope,
                args=args,
                working=current_value,
                memory_state=memory_state,
                allow_stubs=allow_stubs,
                steps_history=steps_history,
                chain_name=chain_name,
            )
            if while_status == "error":
                return "error", while_result, while_value
            if while_status == "ok" and while_result is not None:
                status = "ok"
                result_value = while_result
            current_value = while_value
        else:
            _record_runtime_error(
                context,
                code="workflow.unsupported_node",
                message=f"Workflow node type '{node_type}' is not supported",
                scope=chain_name,
                source="workflow",
            )
            return "error", {"error": f"Unsupported workflow node '{node_type}'"}, current_value
    if encountered_error and status != "error":
        status = "error"
    return status, result_value, current_value


def _execute_workflow_step(
    step: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any, bool]:
    chain_scope["counter"] = chain_scope.get("counter", 0) + 1
    step_label = step.get("name") or step.get("target") or f"step_{chain_scope['counter']}"
    kind = (step.get("kind") or "").lower()
    target = step.get("target") or ""
    options = step.get("options") or {}
    resolved_options = _resolve_placeholders(options, context)
    stop_on_error = bool(step.get("stop_on_error", True))
    read_memory = _extract_memory_names(resolved_options.pop("read_memory", None))
    write_memory = _extract_memory_names(resolved_options.pop("write_memory", None))
    memory_payload = _memory_snapshot(memory_state, read_memory) if read_memory else {}
    entry: Dict[str, Any] = {
        "step": len(steps_history) + 1,
        "kind": kind,
        "name": step_label,
        "inputs": None,
        "output": None,
        "status": "partial",
    }
    steps_history.append(entry)
    current_value = working
    step_result: Any = None
    if kind == "template":
        template = AI_TEMPLATES.get(target) or {}
        prompt = template.get("prompt", "")
        template_context = {
            "input": working,
            "vars": args,
            "payload": args,
            "memory": memory_payload,
            "steps": context.get("steps", {}),
            "loop": context.get("loop", {}),
            "locals": chain_scope.get("locals", {}),
        }
        entry["inputs"] = template_context
        try:
            rendered = _render_template_value(prompt, template_context)
            entry["output"] = rendered
            entry["status"] = "ok"
            current_value = rendered
            step_result = rendered
            if write_memory:
                _write_memory_entries(
                    memory_state,
                    write_memory,
                    rendered,
                    context=context,
                    source=f"template:{step_label}",
                )
        except Exception as exc:
            entry["output"] = {"error": str(exc)}
            entry["status"] = "error"
    elif kind == "connector":
        connector_payload = dict(args)
        connector_payload.setdefault("prompt", working)
        if memory_payload:
            connector_payload.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = connector_payload
        response = call_llm_connector(target, connector_payload)
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("result")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
        if step_status == "ok" and write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                response.get("result"),
                context=context,
                source=f"connector:{step_label}",
            )
    elif kind == "prompt":
        prompt_payload = working if isinstance(working, dict) else working
        if isinstance(prompt_payload, dict):
            payload_data = dict(prompt_payload)
        else:
            payload_data = {"value": prompt_payload}
        if memory_payload:
            payload_data.setdefault("memory", {}).update(memory_payload)
        if read_memory:
            payload_data["read_memory"] = read_memory
        if write_memory:
            payload_data["write_memory"] = write_memory
        response = run_prompt(target, payload_data, context=context, memory=memory_payload)
        entry["inputs"] = response.get("inputs")
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("output")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
    elif kind == "tool":
        tools_registry = context.get("tools") if isinstance(context.get("tools"), dict) else {}
        payload = dict(resolved_options)
        if working is not None and "input" not in payload:
            payload["input"] = working
        if memory_payload:
            payload.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = payload
        plugin = tools_registry.get(target)
        if plugin is None:
            message = f"Tool '{target}' is not configured"
            entry["output"] = {"error": message}
            entry["status"] = "error"
            step_result = entry["output"]
            _record_runtime_error(
                context,
                code="tool_not_found",
                message=message,
                scope=f"chain:{chain_name}",
                detail=f"step:{step_label}",
            )
        else:
            try:
                _validate_tool_payload(plugin, payload, chain_name, step_label)
                tool_result = _call_tool_plugin(plugin, context, payload)
                entry["output"] = tool_result
                entry["status"] = "ok"
                current_value = tool_result
                step_result = tool_result
                if write_memory:
                    _write_memory_entries(
                        memory_state,
                        write_memory,
                        tool_result,
                        context=context,
                        source=f"tool:{step_label}",
                    )
            except Exception as exc:
                message = str(exc)
                entry["output"] = {"error": message}
                entry["status"] = "error"
                step_result = entry["output"]
                _record_runtime_error(
                    context,
                    code="tool_step_error",
                    message=f"Tool step '{step_label}' failed in chain '{chain_name}'",
                    scope=f"chain:{chain_name}",
                    detail=message,
                )
    elif kind == "python":
        module_name = resolved_options.get("module") or target or ""
        method_name = resolved_options.get("method") or "predict"
        python_args = dict(args)
        provided_args = resolved_options.get("arguments")
        if isinstance(provided_args, dict):
            python_args.update(provided_args)
        if memory_payload:
            python_args.setdefault("memory", {}).update(memory_payload)
        entry["inputs"] = python_args
        response = call_python_model(module_name, method_name, python_args)
        entry["output"] = response
        entry["status"] = response.get("status", "partial")
        step_status = response.get("status")
        if step_status == "ok":
            current_value = response.get("result")
            step_result = current_value
        elif step_status == "error":
            step_result = response
        elif step_status == "stub":
            step_result = response
        if step_status == "ok" and write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                response.get("result"),
                context=context,
                source=f"python:{step_label}",
            )
    else:
        entry["output"] = {"error": f"Unsupported step kind '{kind}'"}
        entry["status"] = "error"
        step_result = entry["output"]
    evaluation_cfg = step.get("evaluation")
    if evaluation_cfg:
        try:
            evaluation_results = _run_step_evaluations(
                evaluation_cfg,
                entry,
                current_value,
                context,
                chain_name,
                step_label,
            )
        except Exception as exc:
            entry["output"] = {"error": str(exc)}
            entry["status"] = "error"
            step_result = entry["output"]
            entry["result"] = current_value
            chain_scope.setdefault("steps", {})[step_label] = entry
            return entry.get("status", "partial"), current_value, step_result, stop_on_error
        guardrail_name = evaluation_cfg.get("guardrail")
        if guardrail_name:
            try:
                current_value = _apply_guardrail(
                    guardrail_name,
                    evaluation_results,
                    current_value,
                    entry,
                    chain_name,
                    step_label,
                    context,
                )
                step_result = current_value
            except Exception as exc:
                entry["output"] = {"error": str(exc)}
                entry["status"] = "error"
                step_result = entry["output"]
                entry["result"] = current_value
                chain_scope.setdefault("steps", {})[step_label] = entry
                return entry.get("status", "partial"), current_value, step_result, stop_on_error
    entry["result"] = current_value
    chain_scope.setdefault("steps", {})[step_label] = entry
    return entry.get("status", "partial"), current_value, step_result, stop_on_error


def _execute_workflow_if(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    condition = node.get("condition")
    condition_source = node.get("condition_source")
    scope = _workflow_scope(chain_scope, args, working)
    branch_nodes = node.get("then") or []
    branch_selected = False
    if _evaluate_workflow_condition(condition, condition_source, scope, context, chain_name):
        branch_selected = True
    else:
        for branch in node.get("elif", []):
            if _evaluate_workflow_condition(
                branch.get("condition"),
                branch.get("condition_source"),
                scope,
                context,
                chain_name,
            ):
                branch_nodes = branch.get("steps") or []
                branch_selected = True
                break
        if not branch_selected:
            branch_nodes = node.get("else") or []
    if not branch_nodes:
        return "partial", None, working
    return _execute_workflow_nodes(
        branch_nodes,
        context=context,
        chain_scope=chain_scope,
        args=args,
        working=working,
        memory_state=memory_state,
        allow_stubs=allow_stubs,
        steps_history=steps_history,
        chain_name=chain_name,
    )


def _execute_workflow_for(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    iterable = _resolve_workflow_iterable(node, context, chain_scope, args, working, chain_name)
    if not iterable:
        return "partial", None, working
    loop_var = node.get("loop_var") or "item"
    max_iterations = node.get("max_iterations")
    iterations = 0
    result_value: Any = None
    current_value = working
    loop_context = context.setdefault("loop", {})
    scope_loop = chain_scope.setdefault("loop", {})
    for item in iterable:
        if max_iterations and iterations >= max_iterations:
            break
        chain_scope["locals"][loop_var] = item
        loop_context[loop_var] = item
        scope_loop[loop_var] = item
        branch_status, branch_result, branch_value = _execute_workflow_nodes(
            node.get("body") or [],
            context=context,
            chain_scope=chain_scope,
            args=args,
            working=current_value,
            memory_state=memory_state,
            allow_stubs=allow_stubs,
            steps_history=steps_history,
            chain_name=chain_name,
        )
        if branch_status == "error":
            chain_scope["locals"].pop(loop_var, None)
            loop_context.pop(loop_var, None)
            return "error", branch_result, branch_value
        if branch_result is not None:
            result_value = branch_result
        current_value = branch_value
        iterations += 1
    chain_scope["locals"].pop(loop_var, None)
    loop_context.pop(loop_var, None)
    scope_loop.pop(loop_var, None)
    return ("ok" if iterations else "partial"), result_value, current_value


def _execute_workflow_while(
    node: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    max_iterations = node.get("max_iterations") or 100
    iterations = 0
    result_value: Any = None
    current_value = working
    condition = node.get("condition")
    source = node.get("condition_source")
    while iterations < max_iterations:
        scope = _workflow_scope(chain_scope, args, current_value)
        if not _evaluate_workflow_condition(condition, source, scope, context, chain_name):
            break
        branch_status, branch_result, branch_value = _execute_workflow_nodes(
            node.get("body") or [],
            context=context,
            chain_scope=chain_scope,
            args=args,
            working=current_value,
            memory_state=memory_state,
            allow_stubs=allow_stubs,
            steps_history=steps_history,
            chain_name=chain_name,
        )
        if branch_status == "error":
            return "error", branch_result, branch_value
        if branch_result is not None:
            result_value = branch_result
        current_value = branch_value
        iterations += 1
    if iterations >= max_iterations:
        _record_runtime_error(
            context,
            code="workflow.loop_limit",
            message=f"Workflow while loop in chain '{chain_name}' exceeded iteration limit",
            scope=chain_name,
            source="workflow",
        )
        return "error", {"error": "Loop iteration limit exceeded"}, current_value
    return ("ok" if iterations else "partial"), result_value, current_value


def _evaluate_workflow_condition(
    expression: Optional[Any],
    expression_source: Optional[str],
    scope: Dict[str, Any],
    context: Dict[str, Any],
    chain_name: str,
) -> bool:
    if expression is None and not expression_source:
        return False
    try:
        if expression is None:
            return bool(_evaluate_expression_tree(expression_source or "", scope, context))
        return bool(_evaluate_expression_tree(expression, scope, context))
    except Exception as exc:
        _record_runtime_error(
            context,
            code="workflow.condition_failed",
            message=f"Workflow condition in chain '{chain_name}' failed",
            scope=chain_name,
            source="workflow",
            detail=str(exc),
        )
    return False


def _resolve_workflow_iterable(
    node: Dict[str, Any],
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    chain_name: str,
) -> List[Any]:
    source_kind = str(node.get("source_kind") or "expression").lower()
    if source_kind == "dataset":
        name = node.get("source_name")
        dataset_rows = context.get("datasets_data", {}).get(name)
        if not dataset_rows:
            _record_runtime_error(
                context,
                code="workflow.dataset_missing",
                message=f"Dataset '{name}' is not available for workflow loop",
                scope=chain_name,
                source="workflow",
            )
            return []
        return list(dataset_rows)
    expr_payload = node.get("source_expression")
    expr_source = node.get("source_expression_source")
    value = _evaluate_workflow_expression(
        expr_payload,
        expr_source,
        chain_scope,
        args,
        working,
        context,
        chain_name,
    )
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return list(value)
    try:
        return list(value)
    except Exception:
        return []


def _evaluate_workflow_expression(
    expression: Optional[Any],
    expression_source: Optional[str],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    context: Dict[str, Any],
    chain_name: str,
) -> Any:
    if expression is None and not expression_source:
        return None
    scope = _workflow_scope(chain_scope, args, working)
    try:
        if expression is None:
            expr_text = expression_source or ""
            if not expr_text:
                return None
            return _evaluate_expression_tree(expr_text, scope, context)
        return _evaluate_expression_tree(expression, scope, context)
    except Exception as exc:
        _record_runtime_error(
            context,
            code="workflow.expression_failed",
            message=f"Workflow expression in chain '{chain_name}' failed",
            scope=chain_name,
            source="workflow",
            detail=str(exc),
        )
    return None


def _workflow_scope(chain_scope: Dict[str, Any], args: Dict[str, Any], working: Any) -> Dict[str, Any]:
    scope = {
        "input": chain_scope.get("input"),
        "steps": chain_scope.get("steps", {}),
        "locals": chain_scope.get("locals", {}),
        "loop": chain_scope.get("loop", {}),
        "payload": args,
        "value": working,
    }
    scope.update(chain_scope.get("locals", {}))
    return scope


def _normalize_prompt_inputs(prompt_spec: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    schema = prompt_spec.get("input") or []
    if isinstance(payload, dict):
        return dict(payload)
    if not schema:
        return {}
    first_field = schema[0]
    return {first_field.get("name", "value"): payload}


def _coerce_prompt_field_value(field: Dict[str, Any], value: Any) -> Tuple[Any, Optional[str]]:
    name = field.get("name") or "value"
    field_type = str(field.get("type") or "text").lower()
    if value is None:
        return None, None
    try:
        if field_type in {"text", "string"}:
            return str(value), None
        if field_type in {"int", "integer"}:
            return int(value), None
        if field_type in {"float", "number"}:
            return float(value), None
        if field_type in {"boolean", "bool"}:
            if isinstance(value, bool):
                return value, None
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    return True, None
                if lowered in {"false", "0", "no", "off"}:
                    return False, None
            return bool(value), None
        if field_type in {"json", "object"}:
            if isinstance(value, (dict, list)):
                return value, None
            if isinstance(value, str):
                return json.loads(value), None
        if field_type in {"list", "array"}:
            if isinstance(value, list):
                return value, None
            return [value], None
        if field_type == "enum":
            allowed = [str(item) for item in field.get("enum", []) if item is not None]
            text_value = str(value)
            if allowed and text_value not in allowed:
                return text_value, f"Value '{text_value}' is not permitted for '{name}'"
            return text_value, None
        return value, None
    except Exception as exc:
        return value, f"Unable to coerce field '{name}': {exc}"


def _validate_prompt_inputs(
    prompt_spec: Dict[str, Any],
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    schema = prompt_spec.get("input") or []
    provided = dict(payload or {})
    values: Dict[str, Any] = {}
    errors: List[str] = []
    for field in schema:
        name = field.get("name")
        if not name:
            continue
        if name in provided:
            raw_value = provided.pop(name)
        else:
            raw_value = field.get("default")
        if raw_value is None:
            if field.get("required", True):
                errors.append(f"Missing required prompt field '{name}'")
            continue
        coerced, err = _coerce_prompt_field_value(field, raw_value)
        if err:
            errors.append(err)
            continue
        values[name] = coerced
    if provided:
        extras = ", ".join(sorted(provided.keys()))
        errors.append(f"Unsupported prompt inputs: {extras}")
    return values, errors


def _merge_prompt_request_config(
    model_spec: Dict[str, Any],
    prompt_spec: Dict[str, Any],
) -> Dict[str, Any]:
    base_config = _ensure_dict(model_spec.get("config"))
    merged = copy.deepcopy(base_config)
    prompt_params = prompt_spec.get("parameters") or {}
    if not isinstance(prompt_params, dict):
        prompt_params = {}
    payload = merged.setdefault("payload", {})
    payload_keys = {
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "seed",
    }
    for key in payload_keys:
        if key in prompt_params and prompt_params[key] is not None:
            payload.setdefault(key, prompt_params[key])
    for key in ("headers", "params"):
        value = prompt_params.get(key)
        if isinstance(value, dict):
            merged.setdefault(key, {}).update(value)
    for key in ("timeout", "mode", "user_role", "system", "deployment", "endpoint", "api_version"):
        if key in prompt_params and prompt_params[key] is not None and key not in merged:
            merged[key] = prompt_params[key]
    return merged


def _project_prompt_output(
    prompt_spec: Dict[str, Any],
    result_payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    schema = prompt_spec.get("output") or []
    if not schema:
        return {}, []
    errors: List[str] = []
    text_value = result_payload.get("text") or ""
    structured: Optional[Dict[str, Any]] = None
    if isinstance(result_payload.get("json"), dict):
        structured = result_payload["json"]
        field_names = [field.get("name") for field in schema if field.get("name")]
        if not all(name in structured for name in field_names):
            structured = None
    elif isinstance(text_value, str):
        stripped = text_value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                structured = json.loads(stripped)
            except Exception:
                structured = None
    values: Dict[str, Any] = {}
    if structured is None:
        if len(schema) == 1:
            field = schema[0]
            coerced, err = _coerce_prompt_field_value(field, text_value)
            if err:
                errors.append(err)
            else:
                values[field.get("name") or "value"] = coerced
            return values, errors
        errors.append("Prompt output is not valid JSON")
        return {}, errors
    for field in schema:
        name = field.get("name")
        if not name:
            continue
        if name not in structured:
            errors.append(f"Prompt output is missing '{name}'")
            continue
        coerced, err = _coerce_prompt_field_value(field, structured.get(name))
        if err:
            errors.append(err)
            continue
        values[name] = coerced
    return values, errors


def call_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an LLM connector and return structured response details."""

    import traceback
    import urllib.error
    import urllib.parse

    start_time = time.time()
    spec = AI_CONNECTORS.get(name, {})
    args = dict(payload or {})
    context_stub = {
        "env": {key: os.getenv(key) for key in ENV_KEYS},
        "vars": {},
        "app": APP,
    }
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context_stub)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}

    provider = str(config_resolved.get("provider") or spec.get("type") or name or "").strip()
    model_name = str(config_resolved.get("model") or "").strip()
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")

    prompt_value = args.get("prompt") or args.get("input")
    prompt_text = _stringify_prompt_value(name, prompt_value) if prompt_value is not None else ""
    redacted_config = _redact_config(config_resolved)

    def _elapsed_ms() -> float:
        return float(round((time.time() - start_time) * 1000.0, 3))

    def _stub_response(reason: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.warning(_format_error_message(name, prompt_text, reason))
        response: Dict[str, Any] = {
            "status": "stub",
            "provider": provider or "unknown",
            "model": model_name,
            "inputs": args,
            "result": {"text": "[stub: llm call failed]"},
            "config": redacted_config,
            "error": reason,
        }
        if metadata:
            response["metadata"] = metadata
        return response

    def _error_response(reason: str, metadata: Optional[Dict[str, Any]] = None, tb_text: str = "") -> Dict[str, Any]:
        logger.error(_format_error_message(name, prompt_text, reason))
        response: Dict[str, Any] = {
            "status": "error",
            "provider": provider or "unknown",
            "model": model_name,
            "inputs": args,
            "error": reason,
        }
        if metadata:
            response["metadata"] = metadata
        if tb_text:
            response["traceback"] = tb_text
        return response

    if not provider or not model_name:
        reason = "LLM provider or model is not configured"
        meta = {"elapsed_ms": _elapsed_ms()}
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta)

    try:
        request_spec = _build_llm_request(provider, model_name, prompt_text, config_resolved, args)
        method = str(request_spec.get("method") or "POST").upper()
        if method != "POST":
            raise ValueError(f"Unsupported HTTP method '{method}' for LLM connector")

        url = str(request_spec.get("url"))
        params = request_spec.get("params") or {}
        if params:
            query = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}&{query}" if "?" in url else f"{url}?{query}"

        timeout = float(request_spec.get("timeout") or 30.0)
        status_code, raw_text, parsed_json = _http_post_json(
            url,
            request_spec.get("body") or {},
            request_spec.get("headers") or {},
            timeout,
        )

        result_payload: Dict[str, Any] = {}
        if parsed_json is not None:
            result_payload["json"] = parsed_json
            extracted = _extract_llm_text(provider, parsed_json, config_resolved)
            if extracted:
                result_payload["text"] = str(extracted)
        if raw_text:
            if "text" not in result_payload:
                result_payload["text"] = raw_text
            result_payload.setdefault("raw", raw_text)

        metadata = {
            "http_status": status_code,
            "elapsed_ms": _elapsed_ms(),
        }

        return {
            "status": "ok",
            "provider": provider,
            "model": model_name,
            "inputs": args,
            "result": result_payload,
            "metadata": metadata,
        }
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        reason = f"HTTP {exc.code}: {exc.reason or 'request failed'}"
        meta = {
            "http_status": exc.code,
            "elapsed_ms": _elapsed_ms(),
            "response": error_body[:1024] if error_body else None,
        }
        if meta.get("response") is None:
            meta.pop("response", None)
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        meta = {"elapsed_ms": _elapsed_ms()}
        tb_text = ""
        try:
            tb_text = traceback.format_exc(limit=5).strip()
        except Exception:  # pragma: no cover
            tb_text = ""
        if tb_text and len(tb_text) > 3000:
            tb_text = tb_text[:3000]
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta, tb_text)


def run_prompt(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    prompt_spec = AI_PROMPTS.get(name)
    model_name = str(prompt_spec.get("model") or "") if prompt_spec else ""
    prompt_text = ""
    payload_dict = dict(payload or {}) if isinstance(payload, dict) else {}
    memory_state = _ensure_memory_state(context)
    read_memory = _extract_memory_names(payload_dict.pop("read_memory", None))
    write_memory = _extract_memory_names(payload_dict.pop("write_memory", None))
    memory_payload = dict(memory or {})
    if not memory_payload and read_memory:
        memory_payload = _memory_snapshot(memory_state, read_memory)
    normalized_inputs: Dict[str, Any] = dict(payload_dict) if isinstance(payload, dict) else {}

    def _elapsed_ms() -> float:
        return float(round((time.time() - start_time) * 1000.0, 3))

    def _stub_response(reason: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.warning(_format_error_message(name, None, reason))
        response: Dict[str, Any] = {
            "status": "stub",
            "prompt": name,
            "model": model_name,
            "inputs": normalized_inputs,
            "output": {},
            "result": {"text": "[stub: prompt call failed]"},
            "metadata": metadata or {},
            "error": reason,
        }
        return response

    def _error_response(reason: str, metadata: Optional[Dict[str, Any]] = None, tb_text: str = "") -> Dict[str, Any]:
        logger.error(_format_error_message(name, prompt_text, reason))
        response: Dict[str, Any] = {
            "status": "error",
            "prompt": name,
            "model": model_name,
            "inputs": normalized_inputs,
            "metadata": metadata or {},
            "error": reason,
        }
        if tb_text:
            response["traceback"] = tb_text
        return response

    if not prompt_spec:
        metadata = {"elapsed_ms": _elapsed_ms()}
        return _stub_response(f"Prompt '{name}' is not defined", metadata) if allow_stubs else _error_response(f"Prompt '{name}' is not defined", metadata)

    normalized_inputs = _normalize_prompt_inputs(
        prompt_spec,
        payload_dict if isinstance(payload, dict) else (payload or {}),
    )
    validated_inputs, validation_errors = _validate_prompt_inputs(prompt_spec, normalized_inputs)
    if validation_errors:
        metadata = {"elapsed_ms": _elapsed_ms()}
        reason = "; ".join(validation_errors)
        return _stub_response(reason, metadata) if allow_stubs else _error_response(reason, metadata)

    model_spec = AI_MODELS.get(model_name)
    if not model_spec:
        metadata = {"elapsed_ms": _elapsed_ms()}
        reason = f"Prompt model '{model_name}' is not defined"
        return _stub_response(reason, metadata) if allow_stubs else _error_response(reason, metadata)

    provider = str(model_spec.get("provider") or model_name or "unknown")
    model_id = model_spec.get("model")
    if not model_id:
        metadata = {"elapsed_ms": _elapsed_ms()}
        reason = f"Model id is not configured for prompt '{name}'"
        return _stub_response(reason, metadata) if allow_stubs else _error_response(reason, metadata)

    try:
        prompt_text = _render_template_value(
            prompt_spec.get("template"),
            {
                "input": validated_inputs,
                "vars": validated_inputs,
                "payload": validated_inputs,
                "memory": memory_payload,
            },
        )
    except Exception as exc:
        metadata = {"elapsed_ms": _elapsed_ms()}
        reason = f"Prompt template error: {exc}"
        return _stub_response(reason, metadata) if allow_stubs else _error_response(reason, metadata)

    try:
        request_config = _merge_prompt_request_config(model_spec, prompt_spec)
        request_spec = _build_llm_request(provider, model_id, prompt_text, request_config, validated_inputs)
        method = str(request_spec.get("method") or "POST").upper()
        if method != "POST":
            raise ValueError(f"Unsupported HTTP method '{method}' for prompt '{name}'")

        url = str(request_spec.get("url"))
        params = request_spec.get("params") or {}
        if params:
            query = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}&{query}" if "?" in url else f"{url}?{query}"

        timeout = float(request_spec.get("timeout") or 30.0)
        status_code, raw_text, parsed_json = _http_post_json(
            url,
            request_spec.get("body") or {},
            request_spec.get("headers") or {},
            timeout,
        )

        result_payload: Dict[str, Any] = {}
        if parsed_json is not None:
            result_payload["json"] = parsed_json
            extracted = _extract_llm_text(provider, parsed_json, request_config)
            if extracted:
                result_payload["text"] = str(extracted)
        if raw_text:
            if "text" not in result_payload:
                result_payload["text"] = raw_text
            result_payload.setdefault("raw", raw_text)

        metadata = {
            "http_status": status_code,
            "elapsed_ms": _elapsed_ms(),
        }
        structured_output, projection_errors = _project_prompt_output(prompt_spec, result_payload)
        response: Dict[str, Any] = {
            "status": "ok",
            "prompt": name,
            "model": model_name,
            "inputs": validated_inputs,
            "output": structured_output,
            "result": result_payload,
            "metadata": metadata,
        }
        if write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                structured_output or result_payload,
                context=context,
                source=f"prompt:{name}",
            )
        if projection_errors:
            response["warnings"] = projection_errors
        return response
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        reason = f"HTTP {exc.code}: {exc.reason or 'request failed'}"
        meta = {
            "http_status": exc.code,
            "elapsed_ms": _elapsed_ms(),
            "response": error_body[:1024] if error_body else None,
        }
        if meta.get("response") is None:
            meta.pop("response", None)
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        meta = {"elapsed_ms": _elapsed_ms()}
        tb_text = ""
        try:
            tb_text = traceback.format_exc(limit=5).strip()
        except Exception:
            tb_text = ""
        if tb_text and len(tb_text) > 3000:
            tb_text = tb_text[:3000]
        return _stub_response(reason, meta) if allow_stubs else _error_response(reason, meta, tb_text)


def run_chain(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a configured AI chain and return detailed step results."""

    start_time = time.time()
    args = dict(payload or {})
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    spec = AI_CHAINS.get(name)
    runtime_context = context if isinstance(context, dict) else build_context(None)

    if not spec:
        response: Dict[str, Any] = {
            "status": "not_found",
            "result": None,
            "steps": [],
            "inputs": args,
            "metadata": {"elapsed_ms": float(round((time.time() - start_time) * 1000.0, 3))},
        }
        if not allow_stubs:
            response["error"] = f"Chain '{name}' is not defined"
        return response

    input_key = spec.get("input_key", "input")
    working: Any = args.get(input_key, args)
    memory_state = _ensure_memory_state(runtime_context)
    chain_scope: Dict[str, Any] = {
        "input": working,
        "steps": {},
        "locals": {},
        "loop": {},
        "counter": 0,
    }
    runtime_context["steps"] = chain_scope["steps"]
    runtime_context.setdefault("chain", {})["steps"] = chain_scope["steps"]
    runtime_context["payload"] = args

    steps_history: List[Dict[str, Any]] = []
    status, result_value, working = _execute_workflow_nodes(
        spec.get("steps", []),
        context=runtime_context,
        chain_scope=chain_scope,
        args=args,
        working=working,
        memory_state=memory_state,
        allow_stubs=allow_stubs,
        steps_history=steps_history,
        chain_name=name,
    )

    elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
    if status != "error" and result_value is None:
        status = "partial"

    return {
        "status": status if steps_history else "partial",
        "result": result_value,
        "steps": steps_history,
        "inputs": args,
        "metadata": {"elapsed_ms": elapsed_ms},
    }
'''
).strip()

__all__ = ['LLM_SECTION']
