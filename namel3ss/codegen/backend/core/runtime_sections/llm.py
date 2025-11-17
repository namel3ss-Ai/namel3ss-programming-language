from __future__ import annotations

from textwrap import dedent

LLM_SECTION = dedent(
    '''
import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


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
        client_kwargs: Dict[str, Any] = {}
        try:
            client_kwargs["timeout"] = _httpx.Timeout(timeout)
        except Exception:
            client_kwargs["timeout"] = timeout
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
) -> Dict[str, Any]:
    start_time = time.time()
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    prompt_spec = AI_PROMPTS.get(name)
    model_name = str(prompt_spec.get("model") or "") if prompt_spec else ""
    prompt_text = ""
    normalized_inputs: Dict[str, Any] = dict(payload or {}) if isinstance(payload, dict) else {}

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

    normalized_inputs = _normalize_prompt_inputs(prompt_spec, payload or {})
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


def run_chain(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a configured AI chain and return detailed step results."""

    start_time = time.time()
    args = dict(payload or {})
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")
    spec = AI_CHAINS.get(name)

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
    steps_history: List[Dict[str, Any]] = []
    result_value: Any = None
    status: str = "partial"

    for index, step in enumerate(spec.get("steps", []), start=1):
        kind = (step.get("kind") or "").lower()
        target = step.get("target") or ""
        options = step.get("options") or {}
        stop_on_error = bool(step.get("stop_on_error", True))

        entry: Dict[str, Any] = {
            "step": index,
            "kind": kind,
            "name": target,
            "inputs": None,
            "output": None,
            "status": "partial",
        }
        steps_history.append(entry)

        if kind == "template":
            template = AI_TEMPLATES.get(target) or {}
            prompt = template.get("prompt", "")
            context = {
                "input": working,
                "vars": args,
                "payload": args,
            }
            entry["inputs"] = context
            try:
                rendered = _render_template_value(prompt, context)
                entry["output"] = rendered
                entry["status"] = "ok"
                working = rendered
                result_value = rendered
                status = "ok"
            except Exception as exc:  # pragma: no cover
                entry["output"] = {"error": str(exc)}
                entry["status"] = "error"
                status = "error"
                if stop_on_error:
                    break
        elif kind == "connector":
            connector_payload = dict(args)
            connector_payload.setdefault("prompt", working)
            entry["inputs"] = connector_payload
            response = call_llm_connector(target, connector_payload)
            entry["output"] = response
            entry["status"] = response.get("status", "partial")
            step_status = response.get("status")
            if step_status == "ok":
                working = response.get("result")
                result_value = working
                status = "ok"
            elif step_status == "error":
                status = "error"
                result_value = response
                if stop_on_error:
                    break
            elif step_status == "stub" and status != "error":
                result_value = result_value or response
        elif kind == "prompt":
            prompt_payload = working if isinstance(working, dict) else working
            response = run_prompt(target, prompt_payload)
            entry["inputs"] = response.get("inputs")
            entry["output"] = response
            entry["status"] = response.get("status", "partial")
            step_status = response.get("status")
            if step_status == "ok":
                working = response.get("output")
                result_value = working
                status = "ok"
            elif step_status == "error":
                status = "error"
                result_value = response
                if stop_on_error:
                    break
            elif step_status == "stub" and status != "error":
                result_value = result_value or response
        elif kind == "python":
            module_name = options.get("module") or target or ""
            method_name = options.get("method") or "predict"
            python_args = args
            provided_args = options.get("arguments")
            if isinstance(provided_args, dict):
                merged_args = dict(args)
                merged_args.update(provided_args)
                python_args = merged_args
            entry["inputs"] = python_args
            response = call_python_model(module_name, method_name, python_args)
            entry["output"] = response
            entry["status"] = response.get("status", "partial")
            step_status = response.get("status")
            if step_status == "ok":
                working = response.get("result")
                result_value = working
                status = "ok"
            elif step_status == "error":
                status = "error"
                result_value = response
                if stop_on_error:
                    break
            elif step_status == "stub" and status != "error":
                result_value = result_value or response
        else:
            entry["output"] = {"error": f"Unsupported step kind '{kind}'"}
            entry["status"] = "error"
            status = "error"
            if stop_on_error:
                break

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
