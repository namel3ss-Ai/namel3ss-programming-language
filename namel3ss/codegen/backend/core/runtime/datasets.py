"""Shared dataset runtime helpers used by generated modules."""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("namel3ss.runtime.datasets")

__all__ = [
    "fetch_dataset_rows",
    "execute_sql",
    "load_dataset_source",
    "execute_dataset_pipeline",
    "resolve_connector",
]


async def fetch_dataset_rows(
    key: str,
    session: Any,
    context: Dict[str, Any],
    *,
    datasets: Dict[str, Any],
    resolve_connector: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    dataset_cache_settings: Callable[[Dict[str, Any], Dict[str, Any]], Tuple[str, bool, Optional[int]]],
    make_dataset_cache_key: Callable[[str, str, Dict[str, Any]], str],
    dataset_cache_index: Dict[str, Set[str]],
    cache_get: Optional[Callable[[str], Awaitable[Any]]],
    clone_rows: Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]],
    load_dataset_source: Callable[[Dict[str, Any], Dict[str, Any], Any, Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
    execute_dataset_pipeline: Callable[[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
    cache_set: Optional[Callable[[str, List[Dict[str, Any]], Optional[int]], Awaitable[None]]],
    broadcast_dataset_refresh: Optional[Callable[[Optional[str], str, List[Dict[str, Any]], Dict[str, Any], str], Awaitable[None]]],
    schedule_dataset_refresh: Optional[Callable[[str, Dict[str, Any], Any, Dict[str, Any]], Awaitable[None]]],
    record_error: Optional[Callable[..., Dict[str, Any]]] = None,
    record_event: Optional[Callable[..., Dict[str, Any]]] = None,
    record_metric: Optional[Callable[..., Dict[str, Any]]] = None,
    observe_dataset_stage: Optional[Callable[[Optional[str], str, float, int], None]] = None,
    observe_dataset_fetch: Optional[Callable[[Optional[str], str, float, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """Return dataset rows, applying caching and refresh policies when configured."""

    context.setdefault("session", session)
    dataset = datasets.get(key)
    dataset_name = key
    if isinstance(dataset, dict) and dataset.get("name"):
        dataset_name = str(dataset.get("name"))
    started = time.perf_counter()
    cache_state = "disabled"

    if not dataset:
        # Attempt to query the key as a table name directly
        logger.info("Dataset '%s' not found in registry; attempting to query as table", key)
        try:
            # Import required for SQL queries
            from sqlalchemy import text
            
            # Build a simple SELECT query for the table
            query_text = f"SELECT * FROM {key}"
            query = text(query_text)
            
            # Execute the query using the provided session
            if hasattr(session, 'execute'):
                result = await session.execute(query)
                rows = [dict(row._mapping) for row in result.fetchall()]
                
                # Log success
                duration_ms = max((time.perf_counter() - started) * 1000.0, 0.0)
                if record_event is not None:
                    record_event(
                        context,
                        event="dataset.fetch",
                        level="info",
                        message=f"Table '{key}' queried successfully as fallback",
                        data={
                            "dataset": dataset_name,
                            "key": key,
                            "status": "table_fallback",
                            "rows": len(rows),
                            "cache": cache_state,
                        },
                    )
                if record_metric is not None:
                    tags = {"dataset": dataset_name, "status": "table_fallback", "cache": cache_state}
                    record_metric(
                        context,
                        name="dataset.fetch.duration",
                        value=duration_ms,
                        unit="milliseconds",
                        tags=tags,
                        scope=dataset_name,
                    )
                    record_metric(
                        context,
                        name="dataset.fetch.rows",
                        value=len(rows),
                        unit="count",
                        tags=tags,
                        scope=dataset_name,
                    )
                return rows
        except Exception as exc:
            logger.warning("Failed to query table '%s' as fallback: %s", key, exc)
            if record_error is not None:
                record_error(
                    context,
                    code="table_fallback_failed",
                    message=f"Table '{key}' could not be queried as fallback.",
                    scope=key,
                    source="dataset",
                    detail=str(exc),
                )
        
        # If table fallback fails, log as missing and return empty
        if record_error is not None:
            record_error(
                context,
                code="dataset_missing",
                message=f"Dataset '{key}' is not defined.",
                scope=key,
                source="dataset",
                detail="The requested dataset key was not found in the runtime registry.",
            )
        if record_event is not None or record_metric is not None:
            duration_ms = max((time.perf_counter() - started) * 1000.0, 0.0)
            tags = {"dataset": dataset_name, "status": "missing", "cache": cache_state}
            if record_metric is not None:
                record_metric(
                    context,
                    name="dataset.fetch.duration",
                    value=duration_ms,
                    unit="milliseconds",
                    tags=tags,
                    scope=dataset_name,
                )
                record_metric(
                    context,
                    name="dataset.fetch.rows",
                    value=0,
                    unit="count",
                    tags=tags,
                    scope=dataset_name,
                )
            if record_event is not None:
                record_event(
                    context,
                    event="dataset.fetch",
                    level="warning",
                    message=f"Dataset '{dataset_name}' is missing",
                    data={
                        "dataset": dataset_name,
                        "key": key,
                        "status": "missing",
                        "rows": 0,
                        "cache": cache_state,
                    },
                )
        return []

    resolved_connector = resolve_connector(dataset, context)
    scope, cache_enabled, ttl = dataset_cache_settings(dataset, context)
    cache_key = make_dataset_cache_key(key, scope, context)
    cache_state = "enabled" if cache_enabled else "disabled"

    def _emit(status: str, *, rows: int = 0, extra: Optional[Dict[str, Any]] = None, level: Optional[str] = None) -> None:
        if record_event is None and record_metric is None:
            return
        duration_ms = max((time.perf_counter() - started) * 1000.0, 0.0)
        if observe_dataset_fetch is not None:
            try:
                observe_dataset_fetch(dataset_name, status, duration_ms, rows, cache_state)
            except Exception:
                logger.debug("Failed to record dataset fetch observability for %s", dataset_name, exc_info=True)
        tags = {
            "dataset": dataset_name,
            "status": status,
            "cache": cache_state,
        }
        if record_metric is not None:
            record_metric(
                context,
                name="dataset.fetch.duration",
                value=duration_ms,
                unit="milliseconds",
                tags=tags,
                scope=dataset_name,
            )
            record_metric(
                context,
                name="dataset.fetch.rows",
                value=rows,
                unit="count",
                tags=tags,
                scope=dataset_name,
            )
        if record_event is not None:
            payload = {
                "dataset": dataset_name,
                "key": key,
                "status": status,
                "rows": rows,
                "cache": cache_state,
                "scope": scope,
                "cacheKey": cache_key,
            }
            if extra:
                payload.update(extra)
            record_event(
                context,
                event="dataset.fetch",
                level=level or ("warning" if status not in {"ok", "cache_hit"} else "info"),
                message=f"Dataset '{dataset_name}' fetch {status}",
                data=payload,
            )

    def _record_stage(stage: str, duration_ms: float, rows: int) -> None:
        if observe_dataset_stage is not None:
            try:
                observe_dataset_stage(dataset_name, stage, duration_ms, rows)
            except Exception:
                logger.debug("Failed to record dataset stage observability for %s", dataset_name, exc_info=True)
        if record_event is None and record_metric is None:
            return
        base_tags = {"dataset": dataset_name, "cache": cache_state, "stage": stage}
        if record_metric is not None:
            record_metric(
                context,
                name="dataset.stage.duration",
                value=duration_ms,
                unit="milliseconds",
                tags=base_tags,
                scope=dataset_name,
            )
            record_metric(
                context,
                name="dataset.stage.rows",
                value=rows,
                unit="count",
                tags=base_tags,
                scope=dataset_name,
            )
        if record_event is not None:
            record_event(
                context,
                event="dataset.stage",
                level="debug",
                message=f"Dataset '{dataset_name}' stage '{stage}' completed",
                data={
                    "dataset": dataset_name,
                    "stage": stage,
                    "duration_ms": duration_ms,
                    "rows": rows,
                    "cache": cache_state,
                    "scope": scope,
                },
            )

    cache_hit = False
    if cache_enabled:
        dataset_cache_index.setdefault(key, set()).add(cache_key)
        if cache_get is not None:
            cached = await cache_get(cache_key)
            if isinstance(cached, list) and cached:
                rows_cached = clone_rows(cached)
                cache_hit = True
                _emit("cache_hit", rows=len(rows_cached), extra={"cacheHit": True})
                return rows_cached

    source_started = time.perf_counter()
    source_rows = await load_dataset_source(
        dataset,
        resolved_connector,
        session,
        context,
        record_error=record_error,
    )
    source_duration_ms = max((time.perf_counter() - source_started) * 1000.0, 0.0)
    _record_stage("source", source_duration_ms, len(source_rows))

    pipeline_started = time.perf_counter()
    rows = await execute_dataset_pipeline(dataset, source_rows, context)
    pipeline_duration_ms = max((time.perf_counter() - pipeline_started) * 1000.0, 0.0)
    _record_stage("pipeline", pipeline_duration_ms, len(rows))

    if cache_enabled and cache_set is not None:
        await cache_set(cache_key, rows, ttl)
    else:
        context[cache_key] = clone_rows(rows)

    slug = context.get("page") if isinstance(context.get("page"), str) else None
    if broadcast_dataset_refresh is not None:
        await broadcast_dataset_refresh(slug, key, rows, context, cache_key)
    if dataset.get("refresh_policy") and schedule_dataset_refresh is not None:
        await schedule_dataset_refresh(key, dataset, session, context)

    _emit("ok", rows=len(rows), extra={"cacheHit": cache_hit})
    return rows


try:  # Lazy optional imports for SQL execution helpers.
    from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _AsyncSession = None  # type: ignore

try:  # pragma: no cover - optional dependency for sync fallback
    from sqlalchemy.orm import Session as _SyncSession  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _SyncSession = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from starlette.concurrency import run_in_threadpool as _run_in_threadpool  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _run_in_threadpool = None  # type: ignore


async def execute_sql(
    session: Any,
    query: Any,
    *,
    async_session_type: Optional[type] = _AsyncSession,
    sync_session_type: Optional[type] = _SyncSession,
    run_in_threadpool: Optional[Callable[..., Awaitable[Any]]] = _run_in_threadpool,
) -> Any:
    """Execute an SQLAlchemy query against the provided session."""

    if session is None:
        raise RuntimeError("SQL execution requires a database session")

    if async_session_type is not None and isinstance(session, async_session_type):
        return await session.execute(query)

    if sync_session_type is not None and isinstance(session, sync_session_type):
        if run_in_threadpool is None:
            raise RuntimeError("Threadpool executor is required for sync SQL sessions")
        return await run_in_threadpool(session.execute, query)

    executor = getattr(session, "execute", None)
    if executor is None:
        raise RuntimeError("Session does not expose an execute method")
    if run_in_threadpool is None:
        raise RuntimeError("Threadpool executor is required for generic SQL sessions")
    return await run_in_threadpool(executor, query)


async def load_dataset_source(
    dataset: Dict[str, Any],
    connector: Optional[Dict[str, Any]],
    session: Any,
    context: Dict[str, Any],
    *,
    connector_drivers: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]],
    httpx_client_cls: Any,
    normalize_connector_rows: Callable[[Any], List[Dict[str, Any]]],
    extract_connector_rows: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
    execute_sql: Callable[[Any, Any], Awaitable[Any]],
    logger: Any,
    fetch_dataset_rows_fn: Callable[[str, Any, Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
    record_error: Optional[Callable[..., Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Load rows from the dataset's declared source."""

    source_type = str(dataset.get("source_type") or "table").lower()
    source_name = dataset.get("source")

    extractor = extract_connector_rows or normalize_connector_rows

    def _rows_from_driver_result(result: Any, driver_name: str, connector_name: Optional[str]) -> List[Dict[str, Any]]:
        rows = extractor(result) if extractor is not None else normalize_connector_rows(result)
        if isinstance(result, dict):
            status = str(result.get("status") or "").lower()
            error_message = result.get("error")
            severity = "warning"
            if status == "error":
                severity = "error"
            if record_error is not None and status in {"error", "not_configured", "dependency_missing"}:
                record_error(
                    context,
                    code="connector_driver_status",
                    message=(
                        f"Connector driver '{connector_name or driver_name}' returned status '{status}'."
                        if connector_name or driver_name
                        else f"Connector driver returned status '{status}'."
                    ),
                    scope=dataset.get("name"),
                    source="dataset",
                    detail=str(error_message or status),
                    severity=severity,
                )
        return rows

    if source_type == "table":
        query = text(f"SELECT * FROM {source_name}") if source_name else None
        if query is None:
            return []
        try:
            result = await execute_sql(session, query)
            return [dict(row) for row in result.mappings()]
        except Exception as exc:
            if record_error is not None:
                record_error(
                    context,
                    code="dataset_source_error",
                    message=f"Failed to load table dataset '{source_name}'.",
                    scope=dataset.get("name"),
                    source="dataset",
                    detail=str(exc),
                )
            logger.exception("Failed to load table dataset '%s'", source_name)
            return []

    if source_type == "sql":
        connector_name = connector.get("name") if connector else None
        driver = connector_drivers.get("sql")
        if driver:
            try:
                result = await driver(connector, context)
                return _rows_from_driver_result(result, "sql", connector_name)
            except Exception as exc:
                if record_error is not None:
                    record_error(
                        context,
                        code="connector_driver_error",
                        message=f"SQL connector driver '{connector_name}' failed.",
                        scope=dataset.get("name"),
                        source="dataset",
                        detail=str(exc),
                    )
                logger.exception("SQL connector driver '%s' failed", connector_name)
                return []
        query_text = connector.get("options", {}).get("query") if connector else None
        if not query_text:
            return []
        try:
            result = await execute_sql(session, text(query_text))
            return [dict(row) for row in result.mappings()]
        except Exception as exc:
            if record_error is not None:
                record_error(
                    context,
                    code="dataset_source_error",
                    message=f"Failed to execute SQL query for dataset '{dataset.get('name')}'.",
                    scope=dataset.get("name"),
                    source="dataset",
                    detail=str(exc),
                )
            logger.exception("Failed to execute SQL query for dataset '%s'", dataset.get("name"))
            return []

    if source_type == "file":
        import csv

        path = connector.get("name") if connector else source_name
        if not path:
            return []
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [dict(row) for row in reader]
        except FileNotFoundError:
            logger.warning("Dataset file '%s' not found", path)
            if record_error is not None:
                record_error(
                    context,
                    code="dataset_source_missing",
                    message=f"Dataset file '{path}' was not found.",
                    scope=dataset.get("name"),
                    source="dataset",
                    detail="The configured file path does not exist.",
                    severity="warning",
                )
        except Exception as exc:
            if record_error is not None:
                record_error(
                    context,
                    code="dataset_source_error",
                    message=f"Failed to load dataset file '{path}'.",
                    scope=dataset.get("name"),
                    source="dataset",
                    detail=str(exc),
                )
            logger.exception("Failed to load dataset file '%s'", path)
        return []

    if source_type == "rest":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = connector_drivers.get("rest")
        if driver:
            try:
                result = await driver(connector, context)
                return _rows_from_driver_result(result, "rest", connector_name)
            except Exception as exc:
                if record_error is not None:
                    record_error(
                        context,
                        code="connector_driver_error",
                        message=f"REST connector driver '{connector_name}' failed.",
                        scope=dataset.get("name"),
                        source="dataset",
                        detail=str(exc),
                    )
                logger.exception("REST connector driver '%s' failed", connector_name)
        endpoint = connector.get("options", {}).get("endpoint") if connector else None
        if not endpoint:
            return []
        async with httpx_client_cls() as client:
            try:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
                rows = normalize_connector_rows(payload)
                if rows:
                    return rows
            except Exception as exc:
                if record_error is not None:
                    record_error(
                        context,
                        code="dataset_source_error",
                        message=f"Failed to fetch REST dataset '{connector_name}'.",
                        scope=dataset.get("name"),
                        source="dataset",
                        detail=str(exc),
                    )
                logger.exception("Failed to fetch REST dataset '%s'", connector_name)
        return []

    if source_type == "graphql":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = connector_drivers.get("graphql")
        if driver:
            try:
                result = await driver(connector, context)
                return _rows_from_driver_result(result, "graphql", connector_name)
            except Exception as exc:
                if record_error is not None:
                    record_error(
                        context,
                        code="connector_driver_error",
                        message=f"GraphQL connector driver '{connector_name}' failed.",
                        scope=dataset.get("name"),
                        source="dataset",
                        detail=str(exc),
                    )
                logger.exception("GraphQL connector driver '%s' failed", connector_name)
        return []

    if source_type == "grpc":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = connector_drivers.get("grpc")
        if driver:
            try:
                result = await driver(connector, context)
                return _rows_from_driver_result(result, "grpc", connector_name)
            except Exception as exc:
                if record_error is not None:
                    record_error(
                        context,
                        code="connector_driver_error",
                        message=f"gRPC connector driver '{connector_name}' failed.",
                        scope=dataset.get("name"),
                        source="dataset",
                        detail=str(exc),
                    )
                logger.exception("gRPC connector driver '%s' failed", connector_name)
        return []

    if source_type in {"stream", "streaming"}:
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = connector_drivers.get("stream") or connector_drivers.get("streaming")
        if driver:
            try:
                result = await driver(connector, context)
                return _rows_from_driver_result(result, "stream", connector_name)
            except Exception as exc:
                if record_error is not None:
                    record_error(
                        context,
                        code="connector_driver_error",
                        message=f"Streaming connector driver '{connector_name}' failed.",
                        scope=dataset.get("name"),
                        source="dataset",
                        detail=str(exc),
                    )
                logger.exception("Streaming connector driver '%s' failed", connector_name)
        return []

    if source_type == "dataset" and source_name:
        target_name = str(source_name)
        if target_name == dataset.get("name"):
            return list(dataset.get("sample_rows", []))
        return await fetch_dataset_rows_fn(target_name, session, context)

    return list(dataset.get("sample_rows", []))


async def execute_dataset_pipeline(
    dataset: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
    *,
    clone_rows: Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]],
    apply_filter: Callable[[List[Dict[str, Any]], Optional[str], Dict[str, Any], Optional[str], Optional[Any]], List[Dict[str, Any]]],
    apply_computed_column: Callable[[List[Dict[str, Any]], str, Optional[str], Dict[str, Any], Optional[str], Optional[Any]], None],
    apply_order: Callable[[List[Dict[str, Any]], Sequence[str]], List[Dict[str, Any]]],
    apply_window_operation: Callable[[List[Dict[str, Any]], str, str, Optional[str]], None],
    apply_group_aggregate: Callable[[List[Dict[str, Any]], Sequence[str], Sequence[Tuple[str, str]], Dict[str, Any], Optional[str]], List[Dict[str, Any]]],
    apply_transforms: Callable[[List[Dict[str, Any]], Sequence[Dict[str, Any]], Dict[str, Any], Optional[str]], List[Dict[str, Any]]],
    evaluate_quality_checks: Callable[[List[Dict[str, Any]], Sequence[Dict[str, Any]], Dict[str, Any], Optional[str]], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Apply dataset operations and transforms to produce the final row set."""

    operations: List[Dict[str, Any]] = list(dataset.get("operations") or [])
    transforms: List[Dict[str, Any]] = list(dataset.get("transforms") or [])
    current_rows = clone_rows(rows)
    aggregate_ops: List[Tuple[str, str]] = []
    group_by_columns: List[str] = []
    dataset_name = dataset.get("name")

    for operation in operations:
        otype = str(operation.get("type") or "").lower()
        if otype == "filter":
            current_rows = apply_filter(
                current_rows,
                operation.get("condition"),
                context,
                dataset_name,
                operation.get("condition_expr"),
            )
        elif otype == "computed_column":
            apply_computed_column(
                current_rows,
                operation.get("name"),
                operation.get("expression"),
                context,
                dataset_name,
                operation.get("expression_expr"),
            )
        elif otype == "group_by":
            group_by_columns = list(operation.get("columns") or [])
        elif otype == "aggregate":
            aggregate_ops.append((operation.get("function"), operation.get("expression")))
        elif otype == "order_by":
            current_rows = apply_order(current_rows, operation.get("columns") or [])
        elif otype == "window":
            apply_window_operation(
                current_rows,
                operation.get("name"),
                operation.get("function"),
                operation.get("target"),
            )

    if aggregate_ops:
        current_rows = apply_group_aggregate(current_rows, group_by_columns, aggregate_ops, context, dataset_name)
    if transforms:
        current_rows = apply_transforms(current_rows, transforms, context, dataset_name)

    quality_checks: List[Dict[str, Any]] = list(dataset.get("quality_checks") or [])
    evaluation = evaluate_quality_checks(current_rows, quality_checks, context, dataset_name)
    if evaluation:
        context.setdefault("quality", {})[dataset.get("name")] = evaluation

    features = dataset.get("features") or []
    targets = dataset.get("targets") or []
    if features:
        context.setdefault("features", {})[dataset.get("name")] = features
    if targets:
        context.setdefault("targets", {})[dataset.get("name")] = targets

    return current_rows


def resolve_connector(
    dataset: Dict[str, Any],
    context: Dict[str, Any],
    *,
    deepcopy: Callable[[Dict[str, Any]], Dict[str, Any]],
    resolve_placeholders: Callable[[Any, Dict[str, Any]], Any],
) -> Dict[str, Any]:
    """Return a connector with placeholders resolved against runtime context."""

    connector = dataset.get("connector")
    if not connector:
        return {}

    resolved = deepcopy(connector)
    options = connector.get("options") if isinstance(connector, dict) else None
    resolved["options"] = resolve_placeholders(options, context)
    return resolved
