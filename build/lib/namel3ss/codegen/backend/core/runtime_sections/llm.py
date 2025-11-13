from __future__ import annotations

from textwrap import dedent

LLM_SECTION = dedent(
    '''


def _stringify_prompt_value(name: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, separators=(",", ":"))
        except Exception:
            return str(value)
    return str(value or "").strip()


def _format_stub_prompt(name: str, prompt_text: str) -> str:
    if prompt_text:
        return f"[{name}] {prompt_text}"
    return "This is a stub LLM response."


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
) -> Optional[Dict[str, Any]]:
    if not prompt_text:
        return None
    provider_key = provider.lower()
    method = str(config.get("method") or "post").upper()
    timeout_value = config.get("timeout", 15.0)
    try:
        timeout = max(float(timeout_value), 1.0)
    except Exception:
        timeout = 15.0

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
        return None

    api_key = config.get("api_key")
    if not api_key:
        env_key = config.get("api_key_env")
        if isinstance(env_key, str):
            api_key = os.getenv(env_key) or api_key

    if provider_key == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        bearer = str(api_key)
        if not bearer.lower().startswith("bearer "):
            bearer = f"Bearer {bearer}"
        headers.setdefault("Authorization", bearer)
        headers.setdefault("Content-Type", "application/json")
    elif provider_key == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        headers.setdefault("x-api-key", str(api_key))
        headers.setdefault("content-type", "application/json")
        headers.setdefault("anthropic-version", str(config.get("api_version") or "2023-06-01"))
    elif api_key:
        header_name = str(config.get("api_key_header") or "Authorization")
        if header_name.lower() == "authorization" and not str(api_key).lower().startswith("bearer "):
            headers.setdefault("Authorization", f"Bearer {api_key}")
        else:
            headers.setdefault(header_name, str(api_key))

    return {
        "method": method,
        "url": str(endpoint),
        "headers": headers,
        "params": params,
        "json": body,
        "timeout": timeout,
        "provider": provider,
    }


def _execute_llm_request(request_spec: Dict[str, Any]) -> Dict[str, Any]:
    method = str(request_spec.get("method") or "POST").upper()
    url = str(request_spec.get("url"))
    headers = _ensure_dict(request_spec.get("headers"))
    params = _ensure_dict(request_spec.get("params"))
    json_payload = request_spec.get("json")
    timeout = float(request_spec.get("timeout") or 15.0)

    with httpx.Client(timeout=timeout) as client:
        request_kwargs: Dict[str, Any] = {}
        if headers:
            request_kwargs["headers"] = headers
        if params:
            request_kwargs["params"] = params
        if method == "GET":
            if isinstance(json_payload, dict):
                combined = dict(params)
                combined.update(json_payload)
                request_kwargs["params"] = combined
            elif json_payload is not None:
                param_payload = dict(params)
                param_payload["prompt"] = str(json_payload)
                request_kwargs["params"] = param_payload
        elif isinstance(json_payload, dict):
            request_kwargs["json"] = json_payload
        response = client.request(method, url, **request_kwargs)
        response.raise_for_status()
        try:
            body = response.json()
        except Exception:
            body = {"text": response.text}

    usage = body.get("usage") if isinstance(body, dict) else None
    return {"body": body, "usage": usage}


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
                            segments = []
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


def _truncate_text(value: str, limit: int) -> str:
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)].rstrip() + "..."


def call_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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

    provider = str(config_resolved.get("provider") or spec.get("type") or "stub-provider")
    model_name = str(config_resolved.get("model") or "stub-model")

    prompt_value = args.get("prompt") or args.get("input") or ""
    prompt_text = _stringify_prompt_value(name, prompt_value)
    stub_response = _format_stub_prompt(name, prompt_text)

    try:
        request_spec = _build_llm_request(provider, model_name, prompt_text, config_resolved, args)
        if not request_spec:
            raise ValueError("LLM request is not configured")
        http_response = _execute_llm_request(request_spec)
        body = http_response.get("body")
        raw_text = _extract_llm_text(provider, body, config_resolved)
        try:
            limit = int(config_resolved.get("max_response_chars", 4000))
            limit = max(limit, 0)
        except Exception:
            limit = 4000
        final_text = _truncate_text(str(raw_text).strip() or "No response.", limit)
        return {
            "result": final_text,
            "provider": provider,
            "model": model_name,
            "inputs": args,
            "config": config_resolved,
            "status": "ok",
            "raw_response": body,
            "usage": http_response.get("usage"),
        }
    except Exception:
        logger.exception("LLM connector '%s' failed", name)
        return {
            "result": stub_response,
            "provider": provider,
            "model": model_name,
            "inputs": args,
            "config": config_resolved,
            "status": "stub",
            "error": "llm_error",
        }


def run_chain(name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = AI_CHAINS.get(name)
    args = dict(payload or {})
    if not spec:
        return {
            "result": "stub_chain_output",
            "steps": [],
            "inputs": args,
            "status": "stub",
        }
    input_key = spec.get("input_key", "input")
    working: Any = args.get(input_key)
    if working is None:
        working = args
    history: List[Dict[str, Any]] = []
    for step in spec.get("steps", []):
        history.append(copy.deepcopy(step))
        kind = step.get("kind")
        target = step.get("target")
        options = step.get("options") or {}
        if kind == "template":
            template = AI_TEMPLATES.get(target) or {}
            prompt = template.get("prompt", "")
            context = {"input": working, "vars": args, "payload": args}
            working = _render_template_value(prompt, context)
        elif kind == "connector":
            connector_payload = dict(args)
            if isinstance(working, (dict, list)):
                connector_payload.setdefault("prompt", str(working))
            else:
                connector_payload.setdefault("prompt", working)
            response = call_llm_connector(target, connector_payload)
            working = response.get("result", working)
        elif kind == "python":
            module_name = options.get("module") or target or ""
            method_name = options.get("method") or "predict"
            response = call_python_model(module_name, method_name, args)
            working = response.get("result", working)
        else:
            working = f"{kind}:{target}:{working}" if working is not None else f"{kind}:{target}"
    result_value = working if working is not None else "stub_chain_output"
    return {
        "result": result_value,
        "steps": history,
        "inputs": args,
        "status": "ok" if history else "stub",
    }

    '''
).strip()

__all__ = ['LLM_SECTION']
