"""Prompt handling."""

from textwrap import dedent

PROMPTS = dedent(
    '''
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


def _call_llm_via_registry(
    name: str,
    prompt_text: str,
    args: Dict[str, Any],
    config: Dict[str, Any],
    start_time: float,
) -> Optional[Dict[str, Any]]:
    """
    Try to call an LLM using the new BaseLLM interface from _LLM_INSTANCES.
    
    Returns response dict if successful, None if LLM not found in registry.
    """
    # Check if _LLM_INSTANCES is defined (only exists if app has LLM blocks)
    if '_LLM_INSTANCES' not in globals():
        return None
    
    # Check if LLM instance exists
    llm_instance = _LLM_INSTANCES.get(name)
    if llm_instance is None:
        return None
    
    try:
        from namel3ss.llm import ChatMessage
        
        # Determine mode (chat vs completion)
        mode = str(config.get("mode") or "chat").lower()
        
        # Build messages for chat mode
        if mode == "chat":
            messages = []
            
            # Add system message if present
            system_prompt = config.get("system") or config.get("system_prompt")
            if system_prompt:
                messages.append(ChatMessage(role="system", content=str(system_prompt)))
            
            # Add extra messages from config
            extra_messages = config.get("messages")
            if isinstance(extra_messages, list):
                for msg in extra_messages:
                    if isinstance(msg, dict):
                        role = str(msg.get("role", "user"))
                        content = str(msg.get("content", ""))
                        messages.append(ChatMessage(role=role, content=content))
            
            # Add user prompt
            user_role = str(config.get("user_role") or "user")
            messages.append(ChatMessage(role=user_role, content=prompt_text))
            
            # Call LLM with chat interface
            llm_response = llm_instance.generate_chat(messages, **args)
        else:
            # Completion mode
            llm_response = llm_instance.generate(prompt_text, **args)
        
        # Convert LLMResponse to our standard format
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        
        result_payload = {
            "text": llm_response.text,
            "raw": llm_response.raw,
        }
        
        metadata = {
            "elapsed_ms": elapsed_ms,
            "provider": llm_response.metadata.get("provider", llm_instance.get_provider_name()),
        }
        
        # Add usage info if available
        if llm_response.usage:
            metadata["usage"] = llm_response.usage
            result_payload["usage"] = llm_response.usage
        
        if llm_response.finish_reason:
            metadata["finish_reason"] = llm_response.finish_reason
        
        return {
            "status": "ok",
            "provider": llm_instance.get_provider_name(),
            "model": llm_response.model,
            "inputs": args,
            "result": result_payload,
            "metadata": metadata,
        }
        
    except Exception as exc:
        # Return error response
        import traceback
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        tb_text = ""
        try:
            tb_text = traceback.format_exc(limit=5).strip()
        except Exception:
            tb_text = ""
        if tb_text and len(tb_text) > 3000:
            tb_text = tb_text[:3000]
        
        return {
            "status": "error",
            "provider": llm_instance.get_provider_name() if llm_instance else "unknown",
            "model": getattr(llm_instance, "model", "unknown"),
            "inputs": args,
            "error": f"{type(exc).__name__}: {exc}",
            "metadata": {"elapsed_ms": elapsed_ms},
            "traceback": tb_text,
        }


async def _call_llm_via_registry_async(
    name: str,
    prompt_text: str,
    args: Dict[str, Any],
    config: Dict[str, Any],
    start_time: float,
) -> Optional[Dict[str, Any]]:
    """
    Async version: Try to call an LLM using the new BaseLLM interface from _LLM_INSTANCES.
    
    Returns response dict if successful, None if LLM not found in registry.
    """
    # Check if _LLM_INSTANCES is defined (only exists if app has LLM blocks)
    if '_LLM_INSTANCES' not in globals():
        return None
    
    # Check if LLM instance exists
    llm_instance = _LLM_INSTANCES.get(name)
    if llm_instance is None:
        return None
    
    try:
        from namel3ss.llm import ChatMessage
        
        # Determine mode (chat vs completion)
        mode = str(config.get("mode") or "chat").lower()
        
        # Build messages for chat mode
        if mode == "chat":
            messages = []
            
            # Add system message if present
            system_prompt = config.get("system") or config.get("system_prompt")
            if system_prompt:
                messages.append(ChatMessage(role="system", content=str(system_prompt)))
            
            # Add extra messages from config
            extra_messages = config.get("messages")
            if isinstance(extra_messages, list):
                for msg in extra_messages:
                    if isinstance(msg, dict):
                        role = str(msg.get("role", "user"))
                        content = str(msg.get("content", ""))
                        messages.append(ChatMessage(role=role, content=content))
            
            # Add user prompt
            user_role = str(config.get("user_role") or "user")
            messages.append(ChatMessage(role=user_role, content=prompt_text))
            
            # Call LLM with async chat interface
            llm_response = await llm_instance.agenerate_chat(messages, **args)
        else:
            # Async completion mode
            llm_response = await llm_instance.agenerate(prompt_text, **args)
        
        # Convert LLMResponse to our standard format
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        
        result_payload = {
            "text": llm_response.text,
            "raw": llm_response.raw,
        }
        
        metadata = {
            "elapsed_ms": elapsed_ms,
            "provider": llm_response.metadata.get("provider", llm_instance.get_provider_name()),
        }
        
        # Add usage info if available
        if llm_response.usage:
            metadata["usage"] = llm_response.usage
            result_payload["usage"] = llm_response.usage
        
        if llm_response.finish_reason:
            metadata["finish_reason"] = llm_response.finish_reason
        
        return {
            "status": "ok",
            "provider": llm_instance.get_provider_name(),
            "model": llm_response.model,
            "inputs": args,
            "result": result_payload,
            "metadata": metadata,
        }
        
    except Exception as exc:
        # Return error response
        import traceback
        elapsed_ms = float(round((time.time() - start_time) * 1000.0, 3))
        tb_text = ""
        try:
            tb_text = traceback.format_exc(limit=5).strip()
        except Exception:
            tb_text = ""
        if tb_text and len(tb_text) > 3000:
            tb_text = tb_text[:3000]
        
        return {
            "status": "error",
            "provider": llm_instance.get_provider_name() if llm_instance else "unknown",
            "model": getattr(llm_instance, "model", "unknown"),
            "inputs": args,
            "error": f"{type(exc).__name__}: {exc}",
            "metadata": {"elapsed_ms": elapsed_ms},
            "traceback": tb_text,
        }
'''
).strip()

__all__ = ['PROMPTS']
