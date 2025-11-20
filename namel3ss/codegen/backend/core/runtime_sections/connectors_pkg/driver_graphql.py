"""GraphQL connector driver."""

from __future__ import annotations

from typing import Any, Dict, List

from .utilities import (
    httpx,
    _materialize_connector_value,
    _now_ms,
    _emit_connector_telemetry,
    _is_truthy_env,
    _trim_traceback,
    _extract_rows_from_connector_response,
)


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
        
    # Get connector config from runtime settings
    connector_settings = _runtime_setting("connectors", {})
    retry_max_attempts = connector_settings.get("retry_max_attempts", retries)
    retry_base_delay = connector_settings.get("retry_base_delay", 0.5)
    retry_max_delay = connector_settings.get("retry_max_delay", 5.0)
    
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
    
    # Use resilient request if available
    if make_resilient_request is not None and RetryConfig is not None:
        retry_config = RetryConfig(
            max_attempts=retry_max_attempts,
            base_delay_seconds=retry_base_delay,
            max_delay_seconds=retry_max_delay,
        )
        
        try:
            async def _graphql_request(client_instance: Any) -> Any:
                response = await client_instance.post(
                    endpoint,
                    json={"query": query, "variables": variables or {}},
                    headers=headers if isinstance(headers, dict) else None,
                )
                response.raise_for_status()
                return response.json()
            
            async with _HTTPX_CLIENT_CLS(**client_kwargs) as client:
                payload = await make_resilient_request(
                    _graphql_request,
                    retry_config,
                    connector.get("name") or driver_name,
                    client,
                )
                attempts = retry_max_attempts
                status = "ok"
        except Exception as exc:
            logger.error("GraphQL connector '%s' failed after retries", connector.get("name"))
            last_error = exc
            status = "error"
    else:
        # Fallback to manual retry loop
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


__all__ = ["_default_graphql_driver"]
