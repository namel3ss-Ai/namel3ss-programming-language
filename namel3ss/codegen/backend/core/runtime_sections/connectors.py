from __future__ import annotations

from textwrap import dedent

CONNECTORS_SECTION = dedent(
    '''

import importlib
import inspect
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    httpx = None  # type: ignore

from sqlalchemy.ext.asyncio import AsyncSession


def _normalize_connector_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        rows: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.append(dict(item))
            else:
                rows.append({"value": item})
        return rows
    if isinstance(payload, dict):
        return [dict(payload)]
    return []


def _extract_rows_from_connector_response(payload: Any) -> List[Dict[str, Any]]:
    """Return normalized rows from a connector driver response."""

    if isinstance(payload, dict):
        if "rows" in payload:
            return _normalize_connector_rows(payload.get("rows"))
        if "result" in payload:
            return _normalize_connector_rows(payload.get("result"))
        if "batch" in payload:
            return _normalize_connector_rows(payload.get("batch"))
        return []
    return _normalize_connector_rows(payload)


def _is_truthy_env(name: str) -> bool:
    """Return True when the named environment variable is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _trim_traceback(limit: int = 5, max_chars: int = 3000) -> str:
    """Return a truncated traceback string for logging contexts."""

    import traceback

    try:
        formatted = traceback.format_exc(limit=limit)
    except Exception:  # pragma: no cover - defensive fallback
        return ""
    if not formatted:
        return ""
    text = formatted.strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _redact_secrets(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of *data* with secret-like keys masked."""

    secret_keys = {"api_key", "authorization", "token", "secret", "password", "x-api-key"}

    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): ("***" if str(key).strip().lower() in secret_keys else _sanitize(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        return value

    if not isinstance(data, dict):
        return {}
    return _sanitize(dict(data))


def _prune_nones(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _prune_nones(value) for key, value in payload.items() if value is not None}
    if isinstance(payload, list):
        return [_prune_nones(item) for item in payload if item is not None]
    return payload


def _coerce_bool_option(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_connector_value(value: Any, context: Dict[str, Any], missing_env: Optional[Set[str]] = None) -> Any:
    resolved = _resolve_placeholders(value, context)
    return _materialize_connector_value(resolved, context, missing_env)


def _materialize_connector_value(value: Any, context: Dict[str, Any], missing_env: Optional[Set[str]]) -> Any:
    if isinstance(value, dict):
        return {
            key: _materialize_connector_value(item, context, missing_env)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize_connector_value(item, context, missing_env) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return text
        if text.startswith("${") and text.endswith("}"):
            env_name = text[2:-1].strip()
            env_value = os.getenv(env_name)
            if env_value is None and missing_env is not None:
                missing_env.add(env_name)
            return env_value
        if text.lower().startswith("env?") and ":" not in text:
            env_name = text[4:].strip()
            return os.getenv(env_name)
        prefix, sep, remainder = text.partition(":")
        if sep:
            scope = prefix.strip().lower()
            target = remainder.strip()
            if scope == "env":
                env_value = os.getenv(target)
                if env_value is None and missing_env is not None:
                    missing_env.add(target)
                return env_value
            if scope in {"ctx", "context"}:
                parts = [segment for segment in target.split(".") if segment]
                return _resolve_context_scope("ctx", parts, context, None)
            if scope == "vars":
                parts = [segment for segment in target.split(".") if segment]
                return _resolve_context_scope("vars", parts, context, None)
            if scope == "env?":
                return os.getenv(target)
        rendered = _render_template_value(text, context)
        return rendered
    return value


def _now_ms() -> float:
    """Return the current wall-clock time in milliseconds with millisecond precision."""

    return float(round(time.time() * 1000.0, 3))


def _emit_connector_telemetry(
    context: Optional[Dict[str, Any]],
    connector: Optional[Dict[str, Any]],
    *,
    driver: str,
    status: str,
    start_ms: float,
    rows: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    name = (connector or {}).get("name") if connector else None
    duration_ms = max(_now_ms() - start_ms, 0.0)
    if 'observe_connector_status' in globals():
        try:
            observe_connector_status(name, status)
        except Exception:
            logger.debug("Failed to record connector status for %s", name or driver, exc_info=True)
    tags = {
        "connector": name or driver,
        "driver": driver,
        "status": status,
    }
    if rows is not None:
        tags["rows_present"] = "true" if rows > 0 else "false"
    payload: Dict[str, Any] = {
        "connector": name,
        "driver": driver,
        "status": status,
        "duration_ms": duration_ms,
    }
    if rows is not None:
        payload["rows"] = rows
    if metadata:
        payload.update({key: value for key, value in metadata.items() if value is not None})
    if error:
        payload["error"] = error
    level = "info"
    if status in {"error", "not_configured", "missing_config", "dependency_missing"}:
        level = "warning"
    elif status in {"demo", "cache_hit"}:
        level = "debug"
    _record_runtime_metric(
        context,
        name="connector.duration",
        value=duration_ms,
        unit="milliseconds",
        tags=tags,
        scope=name or driver,
    )
    if rows is not None:
        _record_runtime_metric(
            context,
            name="connector.rows",
            value=rows,
            unit="count",
            tags=tags,
            scope=name or driver,
        )
    _record_runtime_event(
        context,
        event="connector.execute",
        level=level,
        message=f"Connector '{name or driver}' {status}",
        data=payload,
    )


async def _default_sql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    start_ms = _now_ms()
    driver_name = "sql"
    if not connector:
        _emit_connector_telemetry(context, connector, driver=driver_name, status="missing_config", start_ms=start_ms, rows=0)
        return []
    try:
        _require_dependency("sqlalchemy", "sql")
    except ImportError as exc:
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status="error",
            start_ms=start_ms,
            rows=0,
            error=str(exc),
        )
        raise ImportError(str(exc)) from exc
    query = connector.get("options", {}).get("query")
    if not query:
        table_name = connector.get("options", {}).get("table") or connector.get("name")
        if not table_name:
            _emit_connector_telemetry(
                context,
                connector,
                driver=driver_name,
                status="not_configured",
                start_ms=start_ms,
                rows=0,
                metadata={"reason": "no_table"},
            )
            return []
        query = f"SELECT * FROM {table_name}"
    session: Optional[AsyncSession] = context.get("session")
    if session is None:
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status="no_session",
            start_ms=start_ms,
            rows=0,
        )
        return []
    try:
        result = await session.execute(text(query))
        rows = [dict(row) for row in result.mappings()]
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status="ok",
            start_ms=start_ms,
            rows=len(rows),
            metadata={"query": query},
        )
        return rows
    except Exception as exc:
        logger.exception("Default SQL driver failed for query '%s'", query)
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status="error",
            start_ms=start_ms,
            rows=0,
            metadata={"query": query},
            error=str(exc),
        )
        return []


async def _default_rest_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    start_ms = _now_ms()
    driver_name = "rest"
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
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

    attempts = 0
    last_error: Optional[BaseException] = None
    last_status: Optional[int] = None

    async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
        while attempts < max_attempts:
            attempts += 1
            try:
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
    query = options.get("query")
    if not endpoint or not query:
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status="not_configured",
            start_ms=start_ms,
            rows=0,
            metadata={"reason": "missing_endpoint_or_query"},
        )
        return []
    if httpx is None:
        logger.warning("GraphQL connector '%s' requires httpx to be installed", connector.get("name"))
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status="dependency_missing",
            start_ms=start_ms,
            rows=0,
            metadata={"endpoint": endpoint},
        )
        return []
    variables = _resolve_placeholders(options.get("variables"), context)
    headers = _resolve_placeholders(options.get("headers"), context)
    root_field = options.get("root")
    timeout_value = options.get("timeout_ms")
    try:
        timeout = float(timeout_value) / 1000.0 if timeout_value is not None else 10.0
    except Exception:
        timeout = 10.0
    retries_value = options.get("max_retries")
    try:
        retries = max(int(retries_value), 0) if retries_value is not None else 1
    except Exception:
        retries = 1
    client_kwargs: Dict[str, Any] = {}
    if httpx is not None:
        try:
            client_kwargs["timeout"] = httpx.Timeout(timeout)
        except Exception:
            client_kwargs["timeout"] = timeout
    else:
        client_kwargs["timeout"] = timeout
    attempts = 0
    payload: Dict[str, Any] = {}
    status = "error"
    last_error: Optional[Exception] = None
    async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
        try:
            while True:
                attempts += 1
                request_start = _now_ms()
                try:
                    response = await client.post(
                        endpoint,
                        json={"query": query, "variables": variables or {}},
                        headers=headers if isinstance(headers, dict) else None,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    logger.info(
                        "GraphQL connector '%s' succeeded in %.2f ms",
                        connector.get("name"),
                        _now_ms() - request_start,
                    )
                    status = "ok"
                    break
                except ((httpx.HTTPError, httpx.TimeoutException) if httpx is not None else (Exception,)) as exc:
                    logger.warning(
                        "GraphQL connector '%s' attempt %d/%d failed: %s",
                        connector.get("name"),
                        attempts,
                        retries,
                        exc,
                    )
                    last_error = exc
                    status = "retry_failed"
                    if attempts >= retries:
                        raise
                except Exception as exc:
                    logger.exception("Default GraphQL driver failed for endpoint '%s'", endpoint)
                    last_error = exc
                    status = "error"
                    raise
        except Exception as exc:
            logger.error("GraphQL connector '%s' exhausted retries", connector.get("name"))
            last_error = exc
            status = "error"
    data = payload.get("data") if isinstance(payload, dict) else None
    rows_count = 0
    rows: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        target: Any
        if root_field:
            target = data.get(root_field)
        else:
            target = next(iter(data.values())) if data else None
        rows = _normalize_connector_rows(target)
        rows_count = len(rows)
        if rows:
            status = "ok"
    metadata = {
        "endpoint": endpoint,
        "attempts": attempts,
        "root_field": root_field,
    }
    _emit_connector_telemetry(
        context,
        connector,
        driver=driver_name,
        status=status if rows else (status if status != "ok" else "empty"),
        start_ms=start_ms,
        rows=rows_count,
        metadata=metadata,
        error=str(last_error) if last_error else None,
    )
    if rows:
        return rows
    return []


async def _default_grpc_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a gRPC connector via an optional pluggable driver."""

    start_ms = _now_ms()
    driver_name = "grpc"
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    missing_env: Set[str] = set()
    resolved_options = _resolve_connector_value(raw_options, context, missing_env) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else (raw_options if isinstance(raw_options, dict) else {})
    inputs = {"connector": connector_obj.get("name")}
    redacted_config = _prune_nones(_redact_secrets(options))

    def _emit(status: str, result_payload: Any = None, metadata: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        extracted_rows = _extract_rows_from_connector_response({"result": result_payload}) if result_payload is not None else []
        rows_count = len(extracted_rows)
        metadata_payload = dict(metadata or {})
        if missing_env:
            metadata_payload.setdefault("missing_env", sorted(missing_env))
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status=status,
            start_ms=start_ms,
            rows=rows_count,
            metadata=metadata_payload,
            error=error,
        )

    allow_demo = _is_truthy_env("NAMEL3SS_ALLOW_STUBS") or _coerce_bool_option(options.get("demo"), False)

    host = str(options.get("host") or "").strip()
    service = str(options.get("service") or connector_obj.get("name") or "").strip()
    method = str(options.get("method") or "").strip()

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    if not host or not service or not method:
        message = "Missing gRPC configuration (host/service/method)"
        logger.warning("gRPC connector '%s' missing configuration", connector_obj.get("name"))
        if missing_env:
            message = f"{message}; missing env: {', '.join(sorted(missing_env))}"
        payload = {
            "status": "not_configured",
            "result": None,
            "error": message,
            "traceback": None,
            "config": redacted_config,
            "inputs": inputs,
            "metadata": _prune_nones(
                {
                    "elapsed_ms": _elapsed(),
                    "missing_env": sorted(missing_env) if missing_env else None,
                }
            ),
        }
        _emit("not_configured", metadata=payload["metadata"], error=message)
        return payload

    port_value = options.get("port")
    try:
        port = int(port_value) if port_value is not None else 443
    except Exception:
        port = 443

    tls = _coerce_bool_option(options.get("tls"), True)
    deadline_value = options.get("deadline_ms")
    try:
        deadline_ms = int(deadline_value) if deadline_value is not None else None
    except Exception:
        deadline_ms = None

    metadata_option = options.get("metadata")
    metadata_headers = metadata_option if isinstance(metadata_option, dict) else None
    payload_option = options.get("payload")
    payload_dict = payload_option if isinstance(payload_option, dict) else {}

    driver_path = options.get("driver")

    def _import_driver(path: str) -> Callable[..., Any]:
        candidate = _load_python_callable(path) if ":" in path else None
        if candidate is None and path:
            module_name, _, attr = path.rpartition(".")
            if not module_name or not attr:
                raise ImportError(f"Invalid driver path '{path}'")
            module = importlib.import_module(module_name)
            candidate = getattr(module, attr)
        if candidate is None or not callable(candidate):
            raise TypeError(f"Driver '{path}' is not callable")
        return candidate

    if isinstance(driver_path, str) and driver_path.strip():
        try:
            driver_callable = _import_driver(driver_path.strip())
        except Exception as exc:
            logger.error("Failed to import gRPC driver '%s' for connector '%s'", driver_path, connector_obj.get("name"))
            payload = {
                "status": "error",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _trim_traceback(),
                "config": redacted_config,
                "inputs": inputs,
                "metadata": _prune_nones(
                    {
                        "elapsed_ms": _elapsed(),
                        "missing_env": sorted(missing_env) if missing_env else None,
                    }
                ),
            }
            _emit("error", metadata=payload["metadata"], error=payload["error"])
            return payload

        try:
            response = driver_callable(
                host=host,
                service=service,
                method=method,
                payload=payload_dict,
                port=port,
                metadata=metadata_headers,
                tls=tls,
                deadline_ms=deadline_ms,
            )
            if inspect.isawaitable(response):
                response = await response
        except Exception as exc:
            logger.error(
                "gRPC driver '%s' raised an error for connector '%s'",
                driver_path,
                connector_obj.get("name"),
            )
            payload = {
                "status": "error",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _trim_traceback(),
                "config": redacted_config,
                "inputs": inputs,
                "metadata": _prune_nones(
                    {
                        "elapsed_ms": _elapsed(),
                        "missing_env": sorted(missing_env) if missing_env else None,
                    }
                ),
            }
            _emit("error", error=payload["error"], metadata=payload["metadata"])
            return payload

        payload = {
            "status": "ok",
            "result": response,
            "error": None,
            "traceback": None,
            "config": None,
            "inputs": inputs,
            "metadata": _prune_nones(
                {
                    "elapsed_ms": _elapsed(),
                    "endpoint": f"{host}:{port}",
                    "service": service,
                    "method": method,
                    "missing_env": sorted(missing_env) if missing_env else None,
                }
            ),
        }
        _emit("ok", result_payload=response, metadata=payload["metadata"])
        return payload

    if allow_demo:
        logger.info("gRPC connector '%s' running in demo mode", connector_obj.get("name"))
        payload = {
            "status": "demo",
            "result": [
                {
                    "service": service,
                    "method": method,
                    "note": "demo mode – no gRPC client configured",
                }
            ],
            "error": None,
            "traceback": None,
            "config": redacted_config,
            "inputs": inputs,
            "metadata": _prune_nones(
                {
                    "elapsed_ms": _elapsed(),
                    "endpoint": f"{host}:{port}",
                    "service": service,
                    "method": method,
                    "missing_env": sorted(missing_env) if missing_env else None,
                }
            ),
        }
        _emit("demo", result_payload=payload["result"], metadata=payload["metadata"])
        return payload

    message = "No gRPC driver configured. Set 'driver' to a callable implementation or enable demo mode."
    logger.warning("gRPC connector '%s' has no driver configured", connector_obj.get("name"))
    payload = {
        "status": "not_configured",
        "result": None,
        "error": message,
        "traceback": None,
        "config": redacted_config,
        "inputs": inputs,
        "metadata": _prune_nones(
            {
                "elapsed_ms": _elapsed(),
                "missing_env": sorted(missing_env) if missing_env else None,
            }
        ),
    }
    _emit("not_configured", metadata=payload["metadata"], error=message)
    return payload


async def _default_streaming_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Produce streaming batches from configured sources without fabricating data by default."""

    start_ms = _now_ms()
    driver_name = "stream"
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    resolved_options = _resolve_placeholders(raw_options, context) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else (raw_options if isinstance(raw_options, dict) else {})

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    def _emit(
        status: str,
        *,
        batch: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        rows = len(batch) if isinstance(batch, list) else 0
        _emit_connector_telemetry(
            context,
            connector,
            driver=driver_name,
            status=status,
            start_ms=start_ms,
            rows=rows,
            metadata=metadata,
            error=error,
        )

    redacted_config = _redact_secrets(options)
    allow_demo = _is_truthy_env("NAMEL3SS_ALLOW_STUBS") or bool(options.get("demo"))

    batch_size_value = options.get("batch_size", 100)
    try:
        batch_size = max(int(batch_size_value), 1)
    except Exception:
        batch_size = 100

    max_rows_value = options.get("max_rows")
    try:
        max_rows = int(max_rows_value) if max_rows_value is not None else None
    except Exception:
        max_rows = None

    connector_name = str(connector_obj.get("name") or options.get("stream") or "default")
    cursors = context.setdefault("stream_cursors", {}) if isinstance(context, dict) else {}
    cursor_state = cursors.setdefault(connector_name, {})

    seed_rows_raw = options.get("seed_rows") or options.get("rows") or options.get("sample")
    seed_rows_resolved = _resolve_placeholders(seed_rows_raw, context) if seed_rows_raw else None
    seed_rows = _normalize_connector_rows(seed_rows_resolved) if seed_rows_resolved is not None else []

    source_spec = options.get("source") if isinstance(options.get("source"), dict) else {}
    source_type = str(source_spec.get("type") or "").lower()

    def _normalize_batch(items: Iterable[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(dict(item))
            else:
                normalized.append({"value": item})
        return normalized[:batch_size]

    def _error_result(message: str, trace: Optional[str] = None, source: Optional[str] = None) -> Dict[str, Any]:
        logger.error("Streaming connector '%s' failed: %s", connector_name, message)
        metadata = {"elapsed_ms": _elapsed(), "source": source, "exhausted": False}
        _emit("error", metadata=metadata, error=message)
        return {
            "status": "error",
            "batch": None,
            "error": message,
            "traceback": trace,
            "config": redacted_config,
            "metadata": metadata,
        }

    if source_type == "python":
        driver_path = source_spec.get("driver") or source_spec.get("callable")
        if not isinstance(driver_path, str) or not driver_path.strip():
            payload = {
                "status": "not_configured",
                "batch": None,
                "error": "Python streaming source requires a 'driver' callable",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "python", "exhausted": False},
            }
            _emit("not_configured", metadata=payload["metadata"], error=payload["error"])
            return payload
        try:
            driver_callable = _load_python_callable(driver_path.strip())
            if driver_callable is None:
                module_name, _, attr = driver_path.strip().rpartition(".")
                if not module_name or not attr:
                    raise ImportError(f"Invalid driver path '{driver_path}'")
                module = importlib.import_module(module_name)
                driver_callable = getattr(module, attr)
            if not callable(driver_callable):
                raise TypeError(f"Driver '{driver_path}' is not callable")
        except Exception as exc:
            return _error_result(f"{type(exc).__name__}: {exc}", _trim_traceback(), "python")

        kwargs = source_spec.get("kwargs") if isinstance(source_spec.get("kwargs"), dict) else {}
        iterator = cursor_state.get("iterator")
        if iterator is None:
            try:
                produced = driver_callable(**kwargs) if kwargs else driver_callable()
            except Exception as exc:
                return _error_result(f"{type(exc).__name__}: {exc}", _trim_traceback(), "python")
            iterator = iter(produced)
            cursor_state["iterator"] = iterator
            cursor_state["exhausted"] = False

        batch: List[Dict[str, Any]] = []
        while len(batch) < batch_size:
            try:
                item = next(iterator)
            except StopIteration:
                cursor_state["exhausted"] = True
                break
            if item is None:
                continue
            batch.append(item if isinstance(item, dict) else {"value": item})

        metadata = {
            "elapsed_ms": _elapsed(),
            "source": "python",
            "exhausted": bool(cursor_state.get("exhausted") and not batch),
        }
        payload = {
            "status": "ok",
            "batch": batch,
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }
        _emit("ok", batch=batch, metadata=metadata)
        return payload

    if source_type == "http":
        url = source_spec.get("url")
        if not isinstance(url, str) or not url.strip():
            payload = {
                "status": "not_configured",
                "batch": None,
                "error": "HTTP streaming source requires a 'url'",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "http", "exhausted": False},
            }
            _emit("not_configured", metadata=payload["metadata"], error=payload["error"])
            return payload
        method = str(source_spec.get("method") or "GET").upper()
        headers = source_spec.get("headers") if isinstance(source_spec.get("headers"), dict) else {}
        body = source_spec.get("body") if isinstance(source_spec.get("body"), (dict, list)) else None
        timeout_value = source_spec.get("timeout", 10.0)
        try:
            timeout = float(timeout_value)
        except Exception:
            timeout = 10.0

        import json as _json
        import urllib.error
        import urllib.request

        data_bytes = None
        request_headers = {str(key): str(value) for key, value in headers.items()}
        if body is not None and method in {"POST", "PUT", "PATCH"}:
            data_bytes = _json.dumps(body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")

        request = urllib.request.Request(url, data=data_bytes, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_bytes = response.read()
        except urllib.error.URLError as exc:  # pragma: no cover - network failures
            return _error_result(f"{type(exc).__name__}: {exc}", _trim_traceback(), "http")

        text = raw_bytes.decode("utf-8", "replace") if raw_bytes else ""
        try:
            payload = _json.loads(text) if text else []
        except Exception:
            payload = text

        if isinstance(payload, list):
            batch = _normalize_batch(payload)
        elif isinstance(payload, dict):
            batch = _normalize_batch([payload])
        else:
            batch = []

        metadata = {"elapsed_ms": _elapsed(), "source": "http", "exhausted": len(batch) < batch_size}
        payload = {
            "status": "ok",
            "batch": batch,
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }
        _emit("ok", batch=batch, metadata=metadata)
        return payload

    if source_type == "file":
        path = source_spec.get("path")
        if not isinstance(path, str) or not path:
            payload = {
                "status": "not_configured",
                "batch": None,
                "error": "File streaming source requires a 'path'",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "file", "exhausted": False},
            }
            _emit("not_configured", metadata=payload["metadata"], error=payload["error"])
            return payload
        fmt = str(source_spec.get("format") or "jsonl").lower()

        import json as _json

        if fmt == "jsonl":
            offset = int(cursor_state.get("offset", 0))
            batch: List[Dict[str, Any]] = []
            exhausted = False
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    for _ in range(batch_size):
                        line = handle.readline()
                        if not line:
                            exhausted = True
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            batch.append(_json.loads(line))
                        except Exception:
                            batch.append({"value": line})
                    cursor_state["offset"] = handle.tell()
            except FileNotFoundError:
                return _error_result("File source not found", None, "file")

            metadata = {"elapsed_ms": _elapsed(), "source": "file", "exhausted": exhausted and not batch}
            payload = {
                "status": "ok",
                "batch": _normalize_batch(batch),
                "error": None,
                "traceback": None,
                "config": None,
                "metadata": metadata,
            }
            _emit("ok", batch=payload["batch"], metadata=metadata)
            return payload

        records = cursor_state.get("records")
        if records is None:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = _json.load(handle)
            except FileNotFoundError:
                return _error_result("File source not found", None, "file")
            except Exception as exc:
                return _error_result(f"{type(exc).__name__}: {exc}", _trim_traceback(), "file")
            if isinstance(payload, list):
                records = payload
            elif isinstance(payload, dict):
                records = [payload]
            else:
                records = []
            cursor_state["records"] = records

        index = int(cursor_state.get("index", 0))
        chunk = records[index:index + batch_size]
        cursor_state["index"] = index + len(chunk)
        exhausted = cursor_state["index"] >= len(records)
        metadata = {"elapsed_ms": _elapsed(), "source": "file", "exhausted": exhausted}
        payload = {
            "status": "ok",
            "batch": _normalize_batch(chunk),
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }
        _emit("ok", batch=payload["batch"], metadata=metadata)
        return payload

    if seed_rows and not source_type:
        stored_rows = cursor_state.setdefault("seed_rows", seed_rows)
        index = int(cursor_state.get("index", 0))
        slice_rows = stored_rows[index:index + batch_size]
        cursor_state["index"] = index + len(slice_rows)
        exhausted = cursor_state["index"] >= len(stored_rows)
        metadata = {"elapsed_ms": _elapsed(), "source": None, "exhausted": exhausted}
        payload = {
            "status": "ok",
            "batch": _normalize_batch(slice_rows),
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }
        _emit("ok", batch=payload["batch"], metadata=metadata)
        return payload

    if not seed_rows and not source_type:
        if allow_demo:
            start_index = int(cursor_state.get("demo_index", 0))
            end_index = start_index + batch_size
            if max_rows is not None:
                end_index = min(end_index, max_rows)
            batch = [
                {"demo": True, "sequence": seq}
                for seq in range(start_index, end_index)
            ]
            cursor_state["demo_index"] = end_index
            exhausted = max_rows is not None and end_index >= max_rows
            metadata = {"elapsed_ms": _elapsed(), "source": None, "exhausted": exhausted}
            payload = {
                "status": "demo",
                "batch": batch,
                "error": None,
                "traceback": None,
                "config": redacted_config,
                "metadata": metadata,
            }
            _emit("demo", batch=batch, metadata=metadata)
            return payload

        message = "No streaming source configured. Provide 'source' or 'seed_rows', or enable demo mode."
        logger.warning("Streaming connector '%s' has no configured source", connector_name)
        payload = {
            "status": "not_configured",
            "batch": None,
            "error": message,
            "traceback": None,
            "config": redacted_config,
            "metadata": {"elapsed_ms": _elapsed(), "source": None, "exhausted": False},
        }
        _emit("not_configured", metadata=payload["metadata"], error=message)
        return payload

    message = f"Unsupported streaming source type '{source_type or 'unknown'}'"
    logger.warning("Streaming connector '%s' has unsupported source type '%s'", connector_name, source_type)
    payload = {
        "status": "not_configured",
        "batch": None,
        "error": message,
        "traceback": None,
        "config": redacted_config,
        "metadata": {"elapsed_ms": _elapsed(), "source": source_type or None, "exhausted": False},
    }
    _emit("not_configured", metadata=payload["metadata"], error=message)
    return payload


register_connector_driver("sql", _default_sql_driver)
register_connector_driver("rest", _default_rest_driver)
register_connector_driver("graphql", _default_graphql_driver)
register_connector_driver("grpc", _default_grpc_driver)
register_connector_driver("stream", _default_streaming_driver)
register_connector_driver("streaming", _default_streaming_driver)


def _transform_take(rows: List[Dict[str, Any]], options: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    limit = options.get("limit") or options.get("count") or 10
    try:
        limit_int = int(limit)
    except Exception:
        limit_int = 10
    return rows[: max(limit_int, 0)]


def _transform_select_columns(rows: List[Dict[str, Any]], options: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = options.get("columns") or options.get("fields")
    if not columns:
        return rows
    if isinstance(columns, str):
        columns = [segment.strip() for segment in columns.split(",") if segment.strip()]
    selected: List[Dict[str, Any]] = []
    for row in rows:
        entry = {column: row.get(column) for column in columns}
        selected.append(entry)
    return selected


register_dataset_transform("take", _transform_take)
register_dataset_transform("limit", _transform_take)
register_dataset_transform("select", _transform_select_columns)

    '''
).strip()

__all__ = ['CONNECTORS_SECTION']
