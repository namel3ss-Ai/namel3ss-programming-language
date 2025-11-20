"""gRPC connector driver."""

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


__all__ = ["_default_grpc_driver"]
