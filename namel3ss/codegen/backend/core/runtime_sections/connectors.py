from __future__ import annotations

from textwrap import dedent

CONNECTORS_SECTION = dedent(
    '''

import importlib
import inspect
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

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


def _now_ms() -> float:
    """Return the current wall-clock time in milliseconds with millisecond precision."""

    return float(round(time.time() * 1000.0, 3))


async def _default_sql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not connector:
        return []
    try:
        _require_dependency("sqlalchemy", "sql")
    except ImportError as exc:
        raise ImportError(str(exc)) from exc
    query = connector.get("options", {}).get("query")
    if not query:
        table_name = connector.get("options", {}).get("table") or connector.get("name")
        if not table_name:
            return []
        query = f"SELECT * FROM {table_name}"
    session: Optional[AsyncSession] = context.get("session")
    if session is None:
        return []
    try:
        result = await session.execute(text(query))
        return [dict(row) for row in result.mappings()]
    except Exception:
        logger.exception("Default SQL driver failed for query '%s'", query)
        return []


async def _default_rest_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    endpoint = connector.get("options", {}).get("endpoint") if connector else None
    if not endpoint:
        return []
    method = str(connector.get("options", {}).get("method") or "get").lower()
    payload = _resolve_placeholders(connector.get("options", {}).get("payload"), context)
    headers = _resolve_placeholders(connector.get("options", {}).get("headers"), context)
    async with _HTTPX_CLIENT_CLS() as client:
        try:
            request_method = getattr(client, method, client.get)
            response = await request_method(
                endpoint,
                json=payload if isinstance(payload, dict) else None,
                headers=headers if isinstance(headers, dict) else None,
            )
            response.raise_for_status()
            data = response.json()
            rows = _normalize_connector_rows(data)
            if rows:
                return rows
        except Exception:
            logger.exception("Default REST driver failed for endpoint '%s'", endpoint)
    return []


async def _default_graphql_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = connector.get("options", {}) if connector else {}
    endpoint = options.get("endpoint") or options.get("url") or connector.get("name")
    query = options.get("query")
    if not endpoint or not query:
        return []
    variables = _resolve_placeholders(options.get("variables"), context)
    headers = _resolve_placeholders(options.get("headers"), context)
    root_field = options.get("root")
    async with _HTTPX_CLIENT_CLS() as client:
        try:
            response = await client.post(
                endpoint,
                json={"query": query, "variables": variables or {}},
                headers=headers if isinstance(headers, dict) else None,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.exception("Default GraphQL driver failed for endpoint '%s'", endpoint)
            return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        target: Any
        if root_field:
            target = data.get(root_field)
        else:
            target = next(iter(data.values())) if data else None
        rows = _normalize_connector_rows(target)
        if rows:
            return rows
    return []


async def _default_grpc_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a gRPC connector via an optional pluggable driver."""

    start_ms = _now_ms()
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    resolved_options = _resolve_placeholders(raw_options, context) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else (raw_options if isinstance(raw_options, dict) else {})
    inputs = {"connector": connector_obj.get("name")}
    redacted_config = _redact_secrets(options)

    allow_demo = _is_truthy_env("NAMEL3SS_ALLOW_STUBS") or bool(options.get("demo"))

    host = str(options.get("host") or "").strip()
    service = str(options.get("service") or connector_obj.get("name") or "").strip()
    method = str(options.get("method") or "").strip()

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

    if not host or not service or not method:
        message = "Missing gRPC configuration (host/service/method)"
        logger.warning("gRPC connector '%s' missing configuration", connector_obj.get("name"))
        return {
            "status": "not_configured",
            "result": None,
            "error": message,
            "traceback": None,
            "config": redacted_config,
            "inputs": inputs,
            "metadata": {"elapsed_ms": _elapsed()},
        }

    port_value = options.get("port")
    try:
        port = int(port_value) if port_value is not None else 443
    except Exception:
        port = 443

    tls = bool(options.get("tls", True))
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
            return {
                "status": "error",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _trim_traceback(),
                "config": redacted_config,
                "inputs": inputs,
                "metadata": {"elapsed_ms": _elapsed()},
            }

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
            return {
                "status": "error",
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": _trim_traceback(),
                "config": redacted_config,
                "inputs": inputs,
                "metadata": {"elapsed_ms": _elapsed()},
            }

        return {
            "status": "ok",
            "result": response,
            "error": None,
            "traceback": None,
            "config": None,
            "inputs": inputs,
            "metadata": {
                "elapsed_ms": _elapsed(),
                "endpoint": f"{host}:{port}",
                "service": service,
                "method": method,
            },
        }

    if allow_demo:
        logger.info("gRPC connector '%s' running in demo mode", connector_obj.get("name"))
        return {
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
            "metadata": {
                "elapsed_ms": _elapsed(),
                "endpoint": f"{host}:{port}",
                "service": service,
                "method": method,
            },
        }

    message = "No gRPC driver configured. Set 'driver' to a callable implementation or enable demo mode."
    logger.warning("gRPC connector '%s' has no driver configured", connector_obj.get("name"))
    return {
        "status": "not_configured",
        "result": None,
        "error": message,
        "traceback": None,
        "config": redacted_config,
        "inputs": inputs,
        "metadata": {"elapsed_ms": _elapsed()},
    }


async def _default_streaming_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Produce streaming batches from configured sources without fabricating data by default."""

    start_ms = _now_ms()
    connector_obj = connector or {}
    raw_options = connector_obj.get("options") or {}
    resolved_options = _resolve_placeholders(raw_options, context) if raw_options else {}
    options = resolved_options if isinstance(resolved_options, dict) else (raw_options if isinstance(raw_options, dict) else {})

    def _elapsed() -> float:
        return max(_now_ms() - start_ms, 0.0)

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
            return {
                "status": "not_configured",
                "batch": None,
                "error": "Python streaming source requires a 'driver' callable",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "python", "exhausted": False},
            }
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
        return {
            "status": "ok",
            "batch": batch,
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }

    if source_type == "http":
        url = source_spec.get("url")
        if not isinstance(url, str) or not url.strip():
            return {
                "status": "not_configured",
                "batch": None,
                "error": "HTTP streaming source requires a 'url'",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "http", "exhausted": False},
            }
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
        return {
            "status": "ok",
            "batch": batch,
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }

    if source_type == "file":
        path = source_spec.get("path")
        if not isinstance(path, str) or not path:
            return {
                "status": "not_configured",
                "batch": None,
                "error": "File streaming source requires a 'path'",
                "traceback": None,
                "config": redacted_config,
                "metadata": {"elapsed_ms": _elapsed(), "source": "file", "exhausted": False},
            }
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
            return {
                "status": "ok",
                "batch": _normalize_batch(batch),
                "error": None,
                "traceback": None,
                "config": None,
                "metadata": metadata,
            }

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
        return {
            "status": "ok",
            "batch": _normalize_batch(chunk),
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }

    if seed_rows and not source_type:
        stored_rows = cursor_state.setdefault("seed_rows", seed_rows)
        index = int(cursor_state.get("index", 0))
        slice_rows = stored_rows[index:index + batch_size]
        cursor_state["index"] = index + len(slice_rows)
        exhausted = cursor_state["index"] >= len(stored_rows)
        metadata = {"elapsed_ms": _elapsed(), "source": None, "exhausted": exhausted}
        return {
            "status": "ok",
            "batch": _normalize_batch(slice_rows),
            "error": None,
            "traceback": None,
            "config": None,
            "metadata": metadata,
        }

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
            return {
                "status": "demo",
                "batch": batch,
                "error": None,
                "traceback": None,
                "config": redacted_config,
                "metadata": metadata,
            }

        message = "No streaming source configured. Provide 'source' or 'seed_rows', or enable demo mode."
        logger.warning("Streaming connector '%s' has no configured source", connector_name)
        return {
            "status": "not_configured",
            "batch": None,
            "error": message,
            "traceback": None,
            "config": redacted_config,
            "metadata": {"elapsed_ms": _elapsed(), "source": None, "exhausted": False},
        }

    message = f"Unsupported streaming source type '{source_type or 'unknown'}'"
    logger.warning("Streaming connector '%s' has unsupported source type '%s'", connector_name, source_type)
    return {
        "status": "not_configured",
        "batch": None,
        "error": message,
        "traceback": None,
        "config": redacted_config,
        "metadata": {"elapsed_ms": _elapsed(), "source": source_type or None, "exhausted": False},
    }


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
