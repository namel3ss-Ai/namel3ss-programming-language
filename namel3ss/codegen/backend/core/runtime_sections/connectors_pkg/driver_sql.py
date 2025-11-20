"""SQL database connector driver."""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy.ext.asyncio import AsyncSession

from .utilities import (
    _materialize_connector_value,
    _now_ms,
    _emit_connector_telemetry,
    _is_truthy_env,
    _trim_traceback,
)


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


__all__ = ["_default_sql_driver"]
