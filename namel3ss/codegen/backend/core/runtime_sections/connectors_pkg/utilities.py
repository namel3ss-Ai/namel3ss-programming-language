"""Common utility functions for connector drivers."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Set

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore


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
    request_id = current_request_id()
    if request_id:
        payload["request_id"] = request_id
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
    span_attrs = {"connector": connector.get("name") if connector else None, "driver": driver_name}
    with tracing_span("namel3ss.connector.sql", span_attrs):


__all__ = [
    "_normalize_connector_rows",
    "_extract_rows_from_connector_response",
    "_is_truthy_env",
    "_trim_traceback",
    "_redact_secrets",
    "_prune_nones",
    "_coerce_bool_option",
    "_resolve_connector_value",
    "_materialize_connector_value",
    "_now_ms",
    "_emit_connector_telemetry",
]
