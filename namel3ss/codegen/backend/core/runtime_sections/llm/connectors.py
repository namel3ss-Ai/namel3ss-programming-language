"""LLM connectors."""

from textwrap import dedent

CONNECTORS = dedent(
    '''
def call_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an LLM connector and return structured response details."""

    import traceback
    import urllib.error
    import urllib.parse

    start_time = time.time()
    
    # Try new LLM interface first
    args = dict(payload or {})
    prompt_value = args.get("prompt") or args.get("input")
    prompt_text = _stringify_prompt_value(name, prompt_value) if prompt_value is not None else ""
    
    # Attempt to use new BaseLLM interface
    spec = AI_CONNECTORS.get(name, {})
    context_stub = {
        "env": {key: os.getenv(key) for key in ENV_KEYS},
        "vars": {},
        "app": APP,
    }
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context_stub)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}
    
    llm_response = _call_llm_via_registry(name, prompt_text, args, config_resolved, start_time)
    if llm_response is not None:
        return llm_response
    
    # Fall back to old implementation
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


def _is_structured_prompt(prompt_spec: Dict[str, Any]) -> bool:
    """Check if a prompt has structured args and output_schema."""
    return bool(prompt_spec.get("args") or prompt_spec.get("output_schema"))


def _reconstruct_prompt_ast(prompt_spec: Dict[str, Any]) -> "Prompt":
    """Reconstruct a Prompt AST node from the encoded spec for structured execution."""
    if not _STRUCTURED_PROMPTS_AVAILABLE:
        raise ImportError("Structured prompts not available - missing imports")
    
    # Reconstruct args
    args_list = []
    for arg_dict in prompt_spec.get("args", []):
        arg = PromptArgument(
            name=arg_dict["name"],
            arg_type=arg_dict["type"],
            required=arg_dict.get("required", True),
            default=arg_dict.get("default"),
            description=arg_dict.get("description"),
        )
        args_list.append(arg)
    
    # Reconstruct output_schema
    output_schema = None
    schema_dict = prompt_spec.get("output_schema")
    if schema_dict:
        fields = []
        for field_dict in schema_dict.get("fields", []):
            field_type = _reconstruct_output_field_type(field_dict["field_type"])
            field = OutputField(
                name=field_dict["name"],
                field_type=field_type,
                required=field_dict.get("required", True),
                description=field_dict.get("description"),
            )
            fields.append(field)
        output_schema = OutputSchema(fields=fields)
    
    # Create Prompt AST node
    return Prompt(
        name=prompt_spec["name"],
        model=prompt_spec["model"],
        template=prompt_spec["template"],
        args=args_list,
        output_schema=output_schema,
        description=prompt_spec.get("description"),
    )


def _reconstruct_output_field_type(type_dict: Dict[str, Any]) -> "OutputFieldType":
    """Reconstruct an OutputFieldType from encoded dict."""
    element_type = None
    if type_dict.get("element_type"):
        element_type = _reconstruct_output_field_type(type_dict["element_type"])
    
    nested_fields = None
    if type_dict.get("nested_fields"):
        nested_fields = []
        for nested_dict in type_dict["nested_fields"]:
            nested_field_type = _reconstruct_output_field_type(nested_dict["field_type"])
            nested_field = OutputField(
                name=nested_dict["name"],
                field_type=nested_field_type,
                required=nested_dict.get("required", True),
                description=nested_dict.get("description"),
            )
            nested_fields.append(nested_field)
    
    return OutputFieldType(
        base_type=type_dict["base_type"],
        element_type=element_type,
        enum_values=type_dict.get("enum_values"),
        nested_fields=nested_fields,
        nullable=type_dict.get("nullable", False),
    )


def _get_llm_instance(model_name: str, context: Optional[Dict[str, Any]]) -> Optional["BaseLLM"]:
    """Get or create an instantiated LLM instance."""
    if not context:
        return None
    
    # First, try to get from llms registry (if they're already BaseLLM instances)
    llms_registry = context.get("llms", {})
    if isinstance(llms_registry, dict):
        llm_instance = llms_registry.get(model_name)
        if isinstance(llm_instance, BaseLLM):
            return llm_instance
    
    # Otherwise, get model spec and create instance
    model_spec = AI_MODELS.get(model_name)
    if not model_spec:
        return None
    
    provider = str(model_spec.get("provider", "")).lower()
    model_id = model_spec.get("model")
    
    if not model_id:
        return None
    
    # Create LLM instance based on provider
    try:
        if provider == "openai":
            from namel3ss.llm.openai_llm import OpenAILLM
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            return OpenAILLM(
                model=model_id,
                api_key=api_key,
                temperature=model_spec.get("temperature", 0.7),
                max_tokens=model_spec.get("max_tokens", 1024),
            )
        elif provider == "anthropic":
            from namel3ss.llm.anthropic_llm import AnthropicLLM
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            return AnthropicLLM(
                model=model_id,
                api_key=api_key,
                temperature=model_spec.get("temperature", 0.7),
                max_tokens=model_spec.get("max_tokens", 1024),
            )
        # Add more providers as needed
    except ImportError:
        pass
    
    return None


def _run_structured_prompt(
    prompt_spec: Dict[str, Any],
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a structured prompt with validation and return results."""
    if not _STRUCTURED_PROMPTS_AVAILABLE:
        return {
            "status": "error",
            "error": "Structured prompts not available - missing namel3ss.prompts module",
        }
    
    prompt_name = prompt_spec.get("name", "unknown")
    model_name = prompt_spec.get("model", "unknown")
    
    try:
        # Reconstruct the Prompt AST
        prompt_ast = _reconstruct_prompt_ast(prompt_spec)
        
        # Get LLM instance
        llm_instance = _get_llm_instance(model_name, context)
        if not llm_instance:
            error_msg = f"LLM instance '{model_name}' not found in context"
            logger.error(f"Structured prompt '{prompt_name}': {error_msg}")
            _record_runtime_metric(
                context,
                name="prompt_program_failures",
                value=1,
                unit="count",
                tags={"prompt": prompt_name, "reason": "llm_not_found"},
            )
            return {
                "status": "error",
                "error": error_msg,
            }
        
        # Execute structured prompt
        logger.info(f"Executing structured prompt '{prompt_name}' with model '{model_name}'")
        result = execute_structured_prompt_sync(
            prompt_def=prompt_ast,
            llm=llm_instance,
            args=args,
            retry_on_validation_error=True,
            max_retries=2,
        )
        
        # Record metrics
        _record_runtime_metric(
            context,
            name="prompt_program_latency_ms",
            value=result.latency_ms,
            unit="milliseconds",
            tags={"prompt": prompt_name, "model": model_name},
        )
        
        validation_attempts = result.metadata.get("validation_attempts", 1)
        if validation_attempts > 1:
            retry_count = validation_attempts - 1
            _record_runtime_metric(
                context,
                name="prompt_program_retries",
                value=retry_count,
                unit="count",
                tags={"prompt": prompt_name, "model": model_name},
            )
            logger.warning(f"Structured prompt '{prompt_name}' required {retry_count} retries due to validation failures")
        
        # Check for validation errors in metadata
        if result.metadata.get("validation_errors"):
            validation_errors = result.metadata["validation_errors"]
            _record_runtime_metric(
                context,
                name="prompt_program_validation_failures",
                value=len(validation_errors),
                unit="count",
                tags={"prompt": prompt_name},
            )
            # Log sanitized validation errors (first 200 chars)
            for error in validation_errors[:3]:  # Log up to 3 errors
                sanitized_error = str(error)[:200]
                logger.warning(f"Validation error in '{prompt_name}': {sanitized_error}")
        
        # Record success metric
        _record_runtime_metric(
            context,
            name="prompt_program_success",
            value=1,
            unit="count",
            tags={"prompt": prompt_name, "model": model_name},
        )
        
        logger.info(f"Structured prompt '{prompt_name}' completed successfully in {result.latency_ms}ms")
        
        # Convert to compatible response format
        return {
            "status": "ok",
            "prompt": prompt_name,
            "model": model_name,
            "inputs": args,
            "output": result.output,
            "result": {"text": json.dumps(result.output), "json": result.output},
            "metadata": {
                "elapsed_ms": result.latency_ms,
                "tokens_used": result.tokens_used or 0,
                "validation_attempts": validation_attempts,
                "structured": True,
            },
        }
    except Exception as exc:
        error_msg = f"Structured prompt execution failed: {exc}"
        logger.error(f"Structured prompt '{prompt_name}': {error_msg}")
        _record_runtime_metric(
            context,
            name="prompt_program_failures",
            value=1,
            unit="count",
            tags={"prompt": prompt_name, "reason": "execution_error"},
        )
        return {
            "status": "error",
            "error": error_msg,
            "prompt": prompt_name,
            "model": model_name,
        }


# Async version for non-blocking LLM calls
async def call_llm_connector(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Async version: Execute an LLM connector and return structured response details."""

    import traceback

    start_time = time.time()
    
    # Try new LLM interface first
    args = dict(payload or {})
    prompt_value = args.get("prompt") or args.get("input")
    prompt_text = _stringify_prompt_value(name, prompt_value) if prompt_value is not None else ""
    
    # Attempt to use new BaseLLM interface (async)
    spec = AI_CONNECTORS.get(name, {})
    context_stub = {
        "env": {key: os.getenv(key) for key in ENV_KEYS},
        "vars": {},
        "app": APP,
    }
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context_stub)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}
    
    llm_response = await _call_llm_via_registry_async(name, prompt_text, args, config_resolved, start_time)
    if llm_response is not None:
        return llm_response
    
    # Fall back to HTTP-based implementation (keep sync for now - most connectors use the registry)
    # If needed, we can make HTTP calls async using aiohttp later
    provider = str(config_resolved.get("provider") or spec.get("type") or name or "").strip()
    model_name = str(config_resolved.get("model") or "").strip()
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")

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

    # For connectors without BaseLLM registry, return error for now
    # Full HTTP async support can be added if needed
    return _error_response(f"Async connector '{name}' requires BaseLLM registry (not legacy HTTP)")
'''
).strip()

__all__ = ['CONNECTORS']
