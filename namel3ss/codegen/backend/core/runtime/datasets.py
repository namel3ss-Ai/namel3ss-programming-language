"""Shared dataset runtime helpers used by generated modules."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
    broadcast_dataset_refresh: Optional[Callable[[Optional[str], str, List[Dict[str, Any]]], Awaitable[None]]],
    schedule_dataset_refresh: Optional[Callable[[str, Dict[str, Any], Any, Dict[str, Any]], Awaitable[None]]],
) -> List[Dict[str, Any]]:
    """Return dataset rows, applying caching and refresh policies when configured."""

    context.setdefault("session", session)
    dataset = datasets.get(key)
    if not dataset:
        return []

    resolved_connector = resolve_connector(dataset, context)
    scope, cache_enabled, ttl = dataset_cache_settings(dataset, context)
    cache_key = make_dataset_cache_key(key, scope, context)

    if cache_enabled:
        dataset_cache_index.setdefault(key, set()).add(cache_key)
        if cache_get is not None:
            cached = await cache_get(cache_key)
            if isinstance(cached, list) and cached:
                return clone_rows(cached)
    else:
        cached = None

    source_rows = await load_dataset_source(dataset, resolved_connector, session, context)
    rows = await execute_dataset_pipeline(dataset, source_rows, context)

    if cache_enabled and cache_set is not None:
        await cache_set(cache_key, rows, ttl)
    else:
        context[cache_key] = clone_rows(rows)

    slug = context.get("page") if isinstance(context.get("page"), str) else None
    if broadcast_dataset_refresh is not None:
        await broadcast_dataset_refresh(slug, key, rows)
    if dataset.get("refresh_policy") and schedule_dataset_refresh is not None:
        await schedule_dataset_refresh(key, dataset, session, context)

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


from sqlalchemy import text  # noqa: E402  # Local import keeps optional dependency lazy


async def load_dataset_source(
    dataset: Dict[str, Any],
    connector: Optional[Dict[str, Any]],
    session: Any,
    context: Dict[str, Any],
    *,
    connector_drivers: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]],
    httpx_client_cls: Any,
    normalize_connector_rows: Callable[[Any], List[Dict[str, Any]]],
    execute_sql: Callable[[Any, Any], Awaitable[Any]],
    logger: Any,
    fetch_dataset_rows_fn: Callable[[str, Any, Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Load rows from the dataset's declared source."""

    source_type = str(dataset.get("source_type") or "table").lower()
    source_name = dataset.get("source")

    if source_type == "table":
        query = text(f"SELECT * FROM {source_name}") if source_name else None
        if query is None:
            return []
        try:
            result = await execute_sql(session, query)
            return [dict(row) for row in result.mappings()]
        except Exception:
            logger.exception("Failed to load table dataset '%s'", source_name)
            return []

    if source_type == "sql":
        connector_name = connector.get("name") if connector else None
        driver = connector_drivers.get("sql")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("SQL connector driver '%s' failed", connector_name)
                return []
        query_text = connector.get("options", {}).get("query") if connector else None
        if not query_text:
            return []
        try:
            result = await execute_sql(session, text(query_text))
            return [dict(row) for row in result.mappings()]
        except Exception:
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
        except Exception:
            logger.exception("Failed to load dataset file '%s'", path)
        return []

    if source_type == "rest":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = connector_drivers.get("rest")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
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
            except Exception:
                logger.exception("Failed to fetch REST dataset '%s'", connector_name)
        return []

    if source_type == "graphql":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = connector_drivers.get("graphql")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("GraphQL connector driver '%s' failed", connector_name)
        return []

    if source_type == "grpc":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = connector_drivers.get("grpc")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("gRPC connector driver '%s' failed", connector_name)
        return []

    if source_type in {"stream", "streaming"}:
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = connector_drivers.get("stream") or connector_drivers.get("streaming")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
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
    apply_filter: Callable[[List[Dict[str, Any]], Optional[str], Dict[str, Any]], List[Dict[str, Any]]],
    apply_computed_column: Callable[[List[Dict[str, Any]], str, Optional[str], Dict[str, Any]], None],
    apply_order: Callable[[List[Dict[str, Any]], Sequence[str]], List[Dict[str, Any]]],
    apply_window_operation: Callable[[List[Dict[str, Any]], str, str, Optional[str]], None],
    apply_group_aggregate: Callable[[List[Dict[str, Any]], Sequence[str], Sequence[Tuple[str, str]], Dict[str, Any]], List[Dict[str, Any]]],
    apply_transforms: Callable[[List[Dict[str, Any]], Sequence[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]],
    evaluate_quality_checks: Callable[[List[Dict[str, Any]], Sequence[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Apply dataset operations and transforms to produce the final row set."""

    operations: List[Dict[str, Any]] = list(dataset.get("operations") or [])
    transforms: List[Dict[str, Any]] = list(dataset.get("transforms") or [])
    current_rows = clone_rows(rows)
    aggregate_ops: List[Tuple[str, str]] = []
    group_by_columns: List[str] = []

    for operation in operations:
        otype = str(operation.get("type") or "").lower()
        if otype == "filter":
            current_rows = apply_filter(current_rows, operation.get("condition"), context)
        elif otype == "computed_column":
            apply_computed_column(current_rows, operation.get("name"), operation.get("expression"), context)
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
        current_rows = apply_group_aggregate(current_rows, group_by_columns, aggregate_ops, context)
    if transforms:
        current_rows = apply_transforms(current_rows, transforms, context)

    quality_checks: List[Dict[str, Any]] = list(dataset.get("quality_checks") or [])
    evaluation = evaluate_quality_checks(current_rows, quality_checks, context)
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
