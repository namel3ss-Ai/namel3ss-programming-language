"""Structured prompt system."""

from textwrap import dedent

STRUCTURED = dedent(
    '''
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
    
    # Check if this is a structured prompt and route to structured execution
    if _STRUCTURED_PROMPTS_AVAILABLE and _is_structured_prompt(prompt_spec):
        # Extract args from payload (skip memory-related keys)
        args_for_prompt = {k: v for k, v in payload_dict.items() if k not in ["read_memory", "write_memory"]}
        structured_result = _run_structured_prompt(prompt_spec, args_for_prompt, context)
        
        # Handle memory writes if needed
        if structured_result.get("status") == "ok" and write_memory:
            _write_memory_entries(
                memory_state,
                write_memory,
                structured_result.get("output"),
                context=context,
                source=f"prompt:{name}",
            )
        
        return structured_result

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
'''
).strip()

__all__ = ['STRUCTURED']
