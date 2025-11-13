from __future__ import annotations

from textwrap import dedent

CONNECTORS_SECTION = dedent(
    '''


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


async def _default_grpc_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = connector.get("options", {}) if connector else {}
    sample_rows = _resolve_placeholders(options.get("sample") or options.get("rows"), context)
    rows = _normalize_connector_rows(sample_rows)
    if rows:
        return rows
    service = options.get("service") or connector.get("name")
    method = options.get("method")
    logger.info("gRPC connector '%s' invoked without concrete driver; returning stub row", service)
    return [
        {
            "service": service,
            "method": method,
            "status": "UNIMPLEMENTED",
        }
    ]


async def _default_streaming_driver(connector: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    options = connector.get("options", {}) if connector else {}
    stream_name = connector.get("name") or options.get("stream") or "default"
    limit_value = options.get("limit") or options.get("window") or 50
    try:
        limit = max(int(limit_value), 1)
    except Exception:
        limit = 50
    seed_rows = _normalize_connector_rows(_resolve_placeholders(options.get("sample") or options.get("rows"), context))
    buffers = context.setdefault("_stream_buffers", {})
    buffer = buffers.setdefault(stream_name, [])
    if seed_rows:
        buffer.extend(seed_rows)
    if not buffer and options.get("auto_generate", True):
        generated = [{"sequence": index} for index in range(limit)]
        buffer.extend(generated)
    buffer[:] = buffer[-limit:]
    return [dict(row) for row in buffer[-limit:]]


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
