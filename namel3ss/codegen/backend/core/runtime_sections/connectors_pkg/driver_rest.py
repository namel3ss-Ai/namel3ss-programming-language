"""REST API connector driver."""

from __future__ import annotations

from typing import Any, Dict

from .utilities import (
    httpx,
    _materialize_connector_value,
    _now_ms,
    _emit_connector_telemetry,
    "_is_truthy_env",
    _trim_traceback,
    _redact_secrets,
    _prune_nones,
    _coerce_bool_option,
)


    missing_env: Set[str] = set()
    resolved_options = _resolve_connector_value(raw_options, context, missing_env) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else {}

    method = str(options.get("method") or "GET").upper()
    endpoint_value = (
        options.get("endpoint")
        or options.get("url")
        or connector_obj.get("endpoint")
        or connector_obj.get("url")
    )
    endpoint = str(endpoint_value).strip() if endpoint_value else ""

    params: Dict[str, Any] = {}
    params_option = options.get("params")
    if isinstance(params_option, dict):
        params = {str(key): value for key, value in params_option.items() if value is not None}

    headers: Dict[str, Any] = {}
    headers_option = options.get("headers")
    if isinstance(headers_option, dict):
        header_entries: Dict[str, Any] = {}
        for key, value in headers_option.items():
            if value is None:
                continue
            if isinstance(value, (bytes, bytearray)):
                header_entries[str(key)] = value.decode("utf-8", "replace")
            else:
                header_entries[str(key)] = str(value)
        headers = header_entries

    def _split_context_pointer(text: str) -> Tuple[str, str]:
        expr = text.strip()
        if not expr:
            return expr, expr
        if "=" in expr:
            name, _, remainder = expr.partition("=")
            return name.strip(), remainder.strip()
        scoped_expr = expr
        if ":" in expr:
            _, _, scoped_expr = expr.partition(":")
            scoped_expr = scoped_expr.strip()
        if "." in scoped_expr:
            name = scoped_expr.rsplit(".", 1)[-1]
        else:
            name = scoped_expr
        return (name or expr, expr)

    def _resolve_context_param_pointer(pointer: Any) -> Any:
        if pointer is None:
            return None
        if isinstance(pointer, (dict, list)):
            return _resolve_connector_value(pointer, context, missing_env)
        if isinstance(pointer, str):
            text = pointer.strip()
            if not text:
                return None
            materialized = _materialize_connector_value(text, context, missing_env)
            if materialized is not None and materialized != text:
                return materialized
            parts = [segment for segment in text.split(".") if segment]
            if parts:
                fallback = _resolve_context_path(context, parts, None)
                if fallback is not None:
                    return fallback
            return materialized
        return pointer

    def _resolve_context_params(spec: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if spec is None:
            return result
        resolved_spec = _resolve_connector_value(spec, context, missing_env)
        if isinstance(resolved_spec, dict):
            for param_name, pointer in resolved_spec.items():
                value = _resolve_context_param_pointer(pointer)
                if value is not None:
                    result[str(param_name)] = value
        elif isinstance(resolved_spec, list):
            for entry in resolved_spec:
                if isinstance(entry, dict):
                    for param_name, pointer in entry.items():
                        value = _resolve_context_param_pointer(pointer)
                        if value is not None:
                            result[str(param_name)] = value
                elif isinstance(entry, str):
                    name, pointer_expr = _split_context_pointer(entry)
                    value = _resolve_context_param_pointer(pointer_expr)
                    if value is not None:
                        result[str(name)] = value
        return result

    params.update(_resolve_context_params(raw_options.get("context_params") or options.get("context_params")))
    params = {key: value for key, value in params.items() if value is not None}

    redacted_config = _prune_nones(_redact_secrets(options))
    allow_demo = _is_truthy_env("NAMEL3SS_ALLOW_STUBS") or _coerce_bool_option(options.get("demo"), False)

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    def _traverse_result_path(payload: Any, path: str) -> Any:
        if not path:
            return payload
        target = payload
        for segment in path.split("."):
            key = segment.strip()
            if not key:
                continue
            if isinstance(target, dict):
                target = target.get(key)
            elif isinstance(target, list):
                try:
                    index = int(key)
                except ValueError:
                    target = None
                else:
                    if 0 <= index < len(target):
                        target = target[index]
                    else:
                        target = None
            else:
                target = None
            if target is None:
                break
        return target

    result_path = str(options.get("result_path") or "").strip()

    timeout_value = options.get("timeout")
    if timeout_value is None:
        timeout_value = options.get("timeout_seconds")
    if timeout_value is None and options.get("timeout_ms") is not None:
        try:
            timeout_value = float(options.get("timeout_ms")) / 1000.0
        except Exception:
            timeout_value = None
    try:
        timeout_seconds = float(timeout_value) if timeout_value is not None else 10.0
    except Exception:
        timeout_seconds = 10.0

    retries_value = options.get("max_retries") if "max_retries" in options else options.get("retries")
    try:
        max_attempts = max(int(retries_value), 1) if retries_value is not None else 1
    except Exception:
        max_attempts = 1

    body = options.get("body")
    if body is None:
        body = options.get("payload")
    if body is None:
        body = options.get("data")

    inferred_body_format = "json" if isinstance(body, (dict, list)) else "raw"
    body_format_value = options.get("body_format")
    body_format = str(body_format_value or inferred_body_format).strip().lower()

    headers = {key: value for key, value in headers.items() if value is not None}

    inputs: Dict[str, Any] = {"connector": connector_obj.get("name"), "method": method}
    if params:
        inputs["params"] = sorted(params.keys())
    if headers:
        inputs["headers"] = sorted(headers.keys())

    def _finalize(
        status: str,
        *,
        result: Any = None,
        error: Optional[str] = None,
        traceback_text: Optional[str] = None,
        status_code: Optional[int] = None,
        attempts_value: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
        include_config: bool = False,
    ) -> Dict[str, Any]:
        rows_for_status = _extract_rows_from_connector_response({"result": result}) if result is not None else []
        rows_count = len(rows_for_status)
        metadata: Dict[str, Any] = {
            "elapsed_ms": _elapsed(),
            "endpoint": endpoint or None,
            "method": method,
            "status_code": status_code,
            "attempts": attempts_value,
            "result_path": result_path or None,
            "body_format": body_format or None,
        }
        if params:
            metadata["params_keys"] = sorted(params.keys())
        if headers:
            metadata["header_keys"] = sorted(headers.keys())
        if missing_env:
            metadata["missing_env"] = sorted(missing_env)
        if extra_metadata:
            metadata.update(extra_metadata)
        metadata = _prune_nones(metadata)

        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status=status,
            start_ms=start_ms,
            rows=rows_count,
            metadata={key: value for key, value in metadata.items() if key != "elapsed_ms"},
            error=error,
        )

        return {
            "status": status,
            "result": result,
            "rows": rows_for_status if rows_for_status else None,
            "error": error,
            "traceback": traceback_text,
            "config": redacted_config if include_config else None,
            "inputs": _prune_nones(dict(inputs)),
            "metadata": metadata,
        }

    if not endpoint:
        if allow_demo:
            demo_rows_raw = (
                options.get("demo_rows")
                or options.get("seed_rows")
                or options.get("sample_rows")
            )
            demo_rows = (
                _normalize_connector_rows(demo_rows_raw)
                if demo_rows_raw is not None
                else [
                    {"demo": True, "connector": connector_obj.get("name") or "rest", "sequence": index}
                    for index in range(3)
                ]
            )
            logger.info(
                "REST connector '%s' running in demo mode without endpoint",
                connector_obj.get("name"),
            )
            return _finalize(
                "demo",
                result=demo_rows,
                attempts_value=0,
                extra_metadata={"demo": True},
                include_config=True,
            )
        message = "REST connector requires an 'endpoint' option"
        if missing_env:
            message = f"{message}; missing env: {', '.join(sorted(missing_env))}"
        logger.warning("REST connector '%s' missing endpoint", connector_obj.get("name"))
        return _finalize("not_configured", error=message, attempts_value=0, include_config=True)

    if httpx is None:
        message = "REST connector requires httpx to be installed"
        logger.warning("REST connector '%s' requires httpx to be installed", connector_obj.get("name"))
        if allow_demo:
            demo_rows_raw = (
                options.get("demo_rows")
                or options.get("seed_rows")
                or options.get("sample_rows")
            )
            demo_rows = (
                _normalize_connector_rows(demo_rows_raw)
                if demo_rows_raw is not None
                else [
                    {"demo": True, "connector": connector_obj.get("name") or "rest", "sequence": index}
                    for index in range(3)
                ]
            )
            return _finalize(
                "demo",
                result=demo_rows,
                attempts_value=0,
                extra_metadata={"demo": True, "reason": "dependency_missing"},
                include_config=True,
            )
        return _finalize("dependency_missing", error=message, attempts_value=0, include_config=True)

    client_kwargs: Dict[str, Any] = {}
    if timeout_seconds is not None:
        try:
            client_kwargs["timeout"] = httpx.Timeout(timeout_seconds) if httpx is not None else timeout_seconds
        except Exception:
            client_kwargs["timeout"] = timeout_seconds
    if "verify" in options:
        client_kwargs["verify"] = _coerce_bool_option(options.get("verify"), True)
    if _coerce_bool_option(options.get("follow_redirects"), False):
        client_kwargs["follow_redirects"] = True

    request_kwargs: Dict[str, Any] = {}
    if params:
        request_kwargs["params"] = params
    if headers:
        request_kwargs["headers"] = headers
    if body is not None and method not in {"GET", "DELETE", "HEAD"}:
        if body_format == "json":
            request_kwargs["json"] = body
        elif body_format == "form":
            if isinstance(body, dict):
                request_kwargs["data"] = {str(key): "" if value is None else str(value) for key, value in body.items()}
            else:
                request_kwargs["data"] = body
        else:
            if isinstance(body, (bytes, bytearray)):
                request_kwargs["content"] = body
            else:
                request_kwargs["content"] = str(body)

    # Get connector config from runtime settings
    connector_settings = _runtime_setting("connectors", {})
    retry_max_attempts = connector_settings.get("retry_max_attempts", max_attempts)
    retry_base_delay = connector_settings.get("retry_base_delay", 0.5)
    retry_max_delay = connector_settings.get("retry_max_delay", 5.0)
    
    # Use configured retry settings if available and make_resilient_request is available
    if make_resilient_request is not None and RetryConfig is not None:
        retry_config = RetryConfig(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay,
            max_delay_seconds=retry_max_delay,
        )
        
        async def _make_request_with_retry() -> Tuple[int, Any, Optional[int]]:
            \"\"\"Make HTTP request with resilient retry wrapper.\"\"\"
            async def _request_fn(client_instance: Any) -> Any:
                response = await client_instance.request(method, endpoint, **request_kwargs)
                response.raise_for_status()
                return response
            
            async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
                response = await make_resilient_request(
                    _request_fn,
                    retry_config,
                    connector_obj.get("name") or driver_name,
                    client,
                )
                status_code = getattr(response, "status_code", None)
                raw_text = getattr(response, "text", None)
                if raw_text is None and hasattr(response, "content"):
                    content_bytes = getattr(response, "content")
                    raw_text = content_bytes.decode("utf-8", "replace") if content_bytes else ""
                if raw_text is not None and not raw_text.strip():
                    data: Any = []
                else:
                    try:
                        data = response.json()
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        logger.error(
                            "REST connector '%s' returned non-JSON payload",
                            connector_obj.get("name"),
                        )
                        raise ValueError(message) from exc
                return (retry_max_attempts, data, status_code)
        
        try:
            attempts, data, last_status = await _make_request_with_retry()
            result_data = data
            if result_path:
                result_data = _traverse_result_path(data, result_path)
            rows = _normalize_connector_rows(result_data)
            status_value = "ok" if rows else "empty"
            logger.info(
                "REST connector '%s' succeeded with status %s",
                connector_obj.get("name"),
                last_status,
            )
            return _finalize(
                status_value,
                result=result_data,
                status_code=last_status,
                attempts_value=attempts,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            traceback_text = _trim_traceback()
            last_status_code = None
            if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                last_status_code = getattr(exc.response, "status_code", None)
            return _finalize(
                "error",
                error=error_message,
                traceback_text=traceback_text,
                status_code=last_status_code,
                attempts_value=retry_max_attempts,
                include_config=True,
            )
    
    # Fallback to manual retry loop if resilient request is not available
    attempts = 0
    last_error: Optional[BaseException] = None
    last_status: Optional[int] = None

    async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
        while attempts < max_attempts:
            attempts += 1
            try:
                span_attrs = {
                    "connector": connector_obj.get("name") or driver_name,
                    "driver": driver_name,
                    "attempt": attempts,
                    "endpoint": endpoint,
                }
                with tracing_span("namel3ss.connector.rest.request", span_attrs):
                    response = await client.request(method, endpoint, **request_kwargs)
                last_status = getattr(response, "status_code", last_status)
                response.raise_for_status()
                raw_text = getattr(response, "text", None)
                if raw_text is None and hasattr(response, "content"):
                    content_bytes = getattr(response, "content")
                    raw_text = content_bytes.decode("utf-8", "replace") if content_bytes else ""
                if raw_text is not None and not raw_text.strip():
                    data: Any = []
                else:
                    try:
                        data = response.json()
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        logger.error(
                            "REST connector '%s' returned non-JSON payload",
                            connector_obj.get("name"),
                        )
                        return _finalize(
                            "error",
                            error=message,
                            status_code=last_status,
                            attempts_value=attempts,
                            include_config=True,
                        )
                result_data = data
                if result_path:
                    result_data = _traverse_result_path(data, result_path)
                rows = _normalize_connector_rows(result_data)
                status_value = "ok" if rows else "empty"
                logger.info(
                    "REST connector '%s' succeeded with status %s",
                    connector_obj.get("name"),
                    last_status,
                )
                return _finalize(
                    status_value,
                    result=result_data,
                    status_code=last_status,
                    attempts_value=attempts,
                )
            except Exception as exc:
                last_error = exc
                if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                    last_status = getattr(exc.response, "status_code", last_status)
                if httpx is not None and isinstance(exc, httpx.TimeoutException):
                    logger.warning(
                        "REST connector '%s' attempt %d/%d timed out",
                        connector_obj.get("name"),
                        attempts,
                        max_attempts,
                    )
                else:
                    logger.warning(
                        "REST connector '%s' attempt %d/%d failed: %s",
                        connector_obj.get("name"),
                        attempts,
                        max_attempts,
                        exc,
                    )
                if attempts >= max_attempts:
                    error_message = f"{type(exc).__name__}: {exc}"
                    traceback_text = _trim_traceback()
                    return _finalize(
                        "error",
                        error=error_message,
                        traceback_text=traceback_text,
                        status_code=last_status,
                        attempts_value=attempts,
                        include_config=True,
                    )

    error_message = (
        f"{type(last_error).__name__}: {last_error}"
        if last_error is not None
        else "REST connector terminated without a response"
    )
    return _finalize(
        "error",
        error=error_message,
        traceback_text=_trim_traceback(),
        status_code=last_status,
        attempts_value=attempts,
        include_config=True,
    )


async def _default_graphql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    start_ms = _now_ms()
    driver_name = "graphql"
    options = connector.get("options", {}) if connector else {}
    endpoint = options.get("endpoint") or options.get("url") or connector.get("name")


__all__ = ["_default_rest_driver"]
