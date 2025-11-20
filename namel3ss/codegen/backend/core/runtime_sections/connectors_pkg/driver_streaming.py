"""Streaming/WebSocket connector driver."""

from __future__ import annotations

from typing import Any, Dict

from .utilities import (
    _materialize_connector_value,
    _now_ms,
    _emit_connector_telemetry,
    _is_truthy_env,
    _trim_traceback,
)


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


__all__ = ["_default_streaming_driver"]
