"""Utility functions for LLM runtime code generation."""

from textwrap import dedent

UTILITIES = dedent(
    '''
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
'''
).strip()

__all__ = ['UTILITIES']
