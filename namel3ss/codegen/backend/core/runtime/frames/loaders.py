"""Frame source loading from files and databases."""

from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List

from .constants import pa, pq, FrameSourceLoadError


def _resolve_frame_source_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    app_root = os.getenv("NAMEL3SS_APP_ROOT")
    if app_root:
        return Path(app_root) / path
    return Path.cwd() / path


def load_frame_file_source(
    source_config: Dict[str, Any],
    *,
    context: Dict[str, Any],
    resolve_placeholders: Callable[[Any, Dict[str, Any]], Any],
) -> List[Dict[str, Any]]:
    path_value = resolve_placeholders(source_config.get("path"), context) if source_config.get("path") is not None else source_config.get("path")
    if not path_value:
        raise FrameSourceLoadError("Frame file source requires a 'path' value.")
    if not isinstance(path_value, str):
        path_value = str(path_value)
    resolved_path = _resolve_frame_source_path(path_value)
    fmt = str(source_config.get("format") or "csv").lower()
    if fmt == "csv":
        return _read_csv_rows(resolved_path)
    if fmt == "parquet":
        return _read_parquet_rows(resolved_path)
    raise FrameSourceLoadError(f"Unsupported file format '{fmt}' for frame source.")


def _read_csv_rows(resolved_path: Path) -> List[Dict[str, Any]]:
    try:
        if pl is not None:
            frame = pl.read_csv(resolved_path)
            return [dict(row) for row in frame.to_dicts()]
        if pd is not None:
            df = pd.read_csv(resolved_path)
            return df.to_dict(orient="records")
        with resolved_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise FrameSourceLoadError(f"CSV file '{resolved_path}' was not found.") from exc
    except Exception as exc:  # pragma: no cover - unexpected read failure
        raise FrameSourceLoadError(f"Failed to read CSV file '{resolved_path}': {exc}") from exc


def _read_parquet_rows(resolved_path: Path) -> List[Dict[str, Any]]:
    try:
        if pl is not None:
            frame = pl.read_parquet(resolved_path)
            return [dict(row) for row in frame.to_dicts()]
        if pd is not None:
            df = pd.read_parquet(resolved_path)
            return df.to_dict(orient="records")
        if pq is not None:
            table = pq.read_table(resolved_path)
            return table.to_pylist()
        raise FrameSourceLoadError("Parquet sources require polars, pandas, or pyarrow installed.")
    except FileNotFoundError as exc:
        raise FrameSourceLoadError(f"Parquet file '{resolved_path}' was not found.") from exc
    except Exception as exc:  # pragma: no cover - unexpected read failure
        raise FrameSourceLoadError(f"Failed to read Parquet file '{resolved_path}': {exc}") from exc


async def load_frame_sql_source(
    source_config: Dict[str, Any],
    session: Optional[Any],
    *,
    context: Dict[str, Any],
    resolve_placeholders: Callable[[Any, Dict[str, Any]], Any],
) -> List[Dict[str, Any]]:
    try:
        from sqlalchemy import text as _sql_text  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise FrameSourceLoadError("SQL frame sources require SQLAlchemy installed.") from exc

    connection_value = source_config.get("connection")
    query_value = source_config.get("query")
    table_value = source_config.get("table")
    resolved_query = resolve_placeholders(query_value, context) if query_value is not None else query_value
    if resolved_query is None:
        resolved_table = resolve_placeholders(table_value, context) if table_value is not None else table_value
        if not resolved_table:
            raise FrameSourceLoadError("SQL frame sources require a 'table' or explicit 'query'.")
        resolved_query = f"SELECT * FROM {resolved_table}"
    if not isinstance(resolved_query, str):
        resolved_query = str(resolved_query)
    resolved_connection = resolve_placeholders(connection_value, context) if connection_value is not None else None
    if resolved_connection:
        if not isinstance(resolved_connection, str):
            resolved_connection = str(resolved_connection)

        def _run_query() -> List[Dict[str, Any]]:
            try:
                from sqlalchemy import create_engine  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise FrameSourceLoadError("SQL frame sources require SQLAlchemy installed.") from exc
            engine = create_engine(resolved_connection)
            try:
                with engine.connect() as conn:
                    result = conn.execute(_sql_text(resolved_query))
                    return [dict(row) for row in result.mappings()]
            finally:
                engine.dispose()

        return await asyncio.to_thread(_run_query)
    if session is None:
        raise FrameSourceLoadError("SQL frame sources require a database session when no connection is provided.")
    statement = _sql_text(resolved_query)
    result = await _execute_sql_with_session(session, statement)
    return [dict(row) for row in result.mappings()]


async def _execute_sql_with_session(session: Any, statement: Any) -> Any:
    executor = getattr(session, "execute", None)
    if executor is None:
        raise FrameSourceLoadError("Database session does not expose an execute method.")
    maybe_result = executor(statement)
    if inspect.isawaitable(maybe_result):
        return await maybe_result
    return maybe_result




__all__ = [
    "_resolve_frame_source_path",
    "load_frame_file_source",
    "_read_csv_rows",
    "_read_parquet_rows",
    "load_frame_sql_source",
    "_execute_sql_with_session",
]
