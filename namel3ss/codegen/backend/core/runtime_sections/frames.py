from __future__ import annotations

from textwrap import dedent

FRAMES_SECTION = dedent(
    '''
from namel3ss.codegen.backend.core.runtime.frames import (
    DEFAULT_FRAME_LIMIT as _DEFAULT_FRAME_LIMIT,
    MAX_FRAME_LIMIT as _MAX_FRAME_LIMIT,
    N3Frame as _N3Frame,
    FramePipelineExecutionError as _FramePipelineExecutionError,
    FrameSourceLoadError as _FrameSourceLoadError,
    build_pipeline_frame_spec as _build_pipeline_frame_spec,
    execute_frame_pipeline_plan as _execute_frame_pipeline_plan,
    load_frame_file_source as _load_frame_file_source,
    load_frame_sql_source as _load_frame_sql_source,
    project_frame_rows as _project_frame_rows,
)

_FRAME_ERROR_NOT_FOUND = "FRAME_NOT_FOUND"
_FRAME_ERROR_INVALID_PARAMS = "INVALID_QUERY_PARAMS"


async def fetch_frame_rows(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    order_by: Optional[str] = None,
    as_response: bool = False,
) -> Any:
    try:
        frame = await _resolve_frame_runtime(key, session, context, set())
        enforce_defaults = as_response or limit is not None or (offset not in {None, 0}) or order_by is not None
        if not enforce_defaults:
            return frame.rows
        window, total, effective_limit, normalized_offset, normalized_order = _materialize_frame_view(
            frame,
            limit,
            offset,
            order_by,
            require_defaults=as_response,
        )
        if not as_response:
            return window
        schema_payload = frame.schema_payload()
        errors = _collect_runtime_errors(context, scope=frame.name)
        return {
            "name": frame.name,
            "source": _frame_source(frame.spec, frame.name),
            "schema": schema_payload,
            "rows": window,
            "total": total,
            "limit": effective_limit,
            "offset": normalized_offset,
            "order_by": normalized_order,
            "errors": errors,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Frame execution failed for %s", key)
        raise _frame_error(500, "FRAME_INTERNAL_ERROR", "Unexpected error while processing frame request.")


async def fetch_frame_schema(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    frame_key = _normalize_frame_key(key)
    frame_spec = _require_frame_spec(frame_key, context)
    frame = _N3Frame(frame_key, frame_spec, [])
    return frame.schema_payload()


async def export_frame_csv(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    order_by: Optional[str] = None,
) -> bytes:
    frame = await _resolve_frame_runtime(key, session, context, set())
    window, *_ = _materialize_frame_view(
        frame,
        limit,
        offset,
        order_by,
        require_defaults=False,
    )
    column_names = [column.get("name") for column in frame.schema_payload()["columns"] if column.get("name")]
    return frame.to_csv_bytes(columns=column_names, rows=window)


async def export_frame_parquet(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    order_by: Optional[str] = None,
) -> bytes:
    frame = await _resolve_frame_runtime(key, session, context, set())
    window, *_ = _materialize_frame_view(
        frame,
        limit,
        offset,
        order_by,
        require_defaults=False,
    )
    return frame.to_parquet_bytes(rows=window)


def _frame_error(status_code: int, error: str, detail: Optional[str] = None) -> HTTPException:
    payload = {"status_code": status_code, "error": error, "detail": detail}
    return HTTPException(status_code=status_code, detail=payload)


def _normalize_frame_key(key: str) -> str:
    if not key:
        raise _frame_error(404, _FRAME_ERROR_NOT_FOUND, "Frame name must be provided.")
    return str(key)


async def _resolve_frame_runtime(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    visited: Optional[Set[str]] = None,
) -> _N3Frame:
    frame_key = _normalize_frame_key(key)
    active_visited = visited or set()
    if frame_key in active_visited:
        _record_runtime_error(
            context,
            code="frame_recursion_detected",
            message=f"Cyclic frame dependency detected for '{frame_key}'.",
            scope=frame_key,
            source="frame",
            severity="error",
        )
        return _N3Frame(frame_key, {}, [])
    active_visited.add(frame_key)
    frame_spec = _require_frame_spec(frame_key, context)
    base_rows = await _load_frame_source_rows(frame_key, frame_spec, session, context, active_visited)
    if not base_rows:
        base_rows = _frame_example_rows(frame_spec)
    shaped_rows = _project_frame_rows(
        frame_key,
        frame_spec,
        base_rows or [],
        context,
        resolve_placeholders=_resolve_placeholders,
        evaluate_expression=_evaluate_dataset_expression,
        runtime_truthy=_runtime_truthy,
        record_error=_record_runtime_error,
    )
    return _N3Frame(frame_key, frame_spec, shaped_rows or [])


def _require_frame_spec(frame_key: str, context: Dict[str, Any]) -> Dict[str, Any]:
    frame_spec = FRAMES.get(frame_key)
    if isinstance(frame_spec, dict):
        return frame_spec
    _record_runtime_error(
        context,
        code="frame_missing",
        message=f"Frame '{frame_key}' is not defined.",
        scope=frame_key,
        source="frame",
        severity="error",
    )
    raise _frame_error(404, _FRAME_ERROR_NOT_FOUND, f"Frame '{frame_key}' does not exist.")


def _frame_example_rows(frame_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples = frame_spec.get("examples")
    if not isinstance(examples, list):
        return []
    dict_rows = [row for row in examples if isinstance(row, dict)]
    if not dict_rows:
        return []
    return _clone_rows(dict_rows)


def _frame_source(frame_spec: Dict[str, Any], frame_key: str) -> Dict[str, Any]:
    return {
        "type": str(frame_spec.get("source_type") or "dataset"),
        "name": frame_spec.get("source") or frame_spec.get("name") or frame_key,
    }


def _materialize_frame_view(
    frame: _N3Frame,
    limit: Optional[int],
    offset: Optional[int],
    order_by: Optional[str],
    *,
    require_defaults: bool,
) -> Tuple[List[Dict[str, Any]], int, int, int, Optional[str]]:
    normalized_limit, normalized_offset = _normalize_pagination(limit, offset, require_defaults=require_defaults)
    order_spec, normalized_order = _normalize_order_by(order_by, frame)
    if normalized_limit is None and normalized_offset == 0 and order_spec is None:
        window = frame.rows
        total = len(window)
        effective_limit = total
    else:
        window, total = frame.window_rows(order_spec, normalized_limit, normalized_offset)
        remaining = max(total - normalized_offset, 0)
        if normalized_limit is None:
            effective_limit = remaining
        else:
            effective_limit = min(normalized_limit, remaining)
    return window, total, effective_limit, normalized_offset, normalized_order


def _normalize_pagination(
    limit: Optional[int],
    offset: Optional[int],
    *,
    require_defaults: bool,
) -> Tuple[Optional[int], int]:
    normalized_limit: Optional[int]
    if limit is None:
        normalized_limit = _DEFAULT_FRAME_LIMIT if require_defaults else None
    else:
        normalized_limit = _coerce_positive_int(limit, "limit")
    if normalized_limit is not None:
        normalized_limit = min(normalized_limit, _MAX_FRAME_LIMIT)
    normalized_offset = _coerce_non_negative_int(offset, "offset")
    return normalized_limit, normalized_offset


def _coerce_positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, f"{field.title()} must be an integer.")
    if result <= 0:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, f"{field.title()} must be positive.")
    return result


def _coerce_non_negative_int(value: Any, field: str) -> int:
    if value is None:
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, f"{field.title()} must be an integer.")
    if result < 0:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, f"{field.title()} cannot be negative.")
    return result


def _normalize_order_by(order_by: Optional[str], frame: _N3Frame) -> Tuple[Optional[Tuple[str, bool]], Optional[str]]:
    if order_by is None:
        return None, None
    candidate = str(order_by).strip()
    if not candidate:
        return None, None
    descending = False
    column_token = candidate
    if candidate.startswith("-"):
        descending = True
        column_token = candidate[1:]
    if ":" in column_token:
        column_token, direction_token = column_token.split(":", 1)
        direction_token = direction_token.strip().lower()
        if direction_token in {"desc", "descending", "down"}:
            descending = True
        elif direction_token in {"asc", "ascending", "up"}:
            descending = False
    column_name = column_token.strip()
    if not column_name:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, "order_by must reference a column name.")
    valid_columns = {column.get("name") for column in frame.spec.get("columns") or [] if column.get("name")}
    if column_name not in valid_columns:
        raise _frame_error(
            400,
            _FRAME_ERROR_INVALID_PARAMS,
            f"Column '{column_name}' is not defined for frame '{frame.name}'.",
        )
    normalized = f"{column_name}:{'desc' if descending else 'asc'}"
    return (column_name, descending), normalized


async def _load_frame_source_rows(
    frame_key: str,
    frame_spec: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    visited: Set[str],
) -> List[Dict[str, Any]]:
    source_config = frame_spec.get("source_config")
    if isinstance(source_config, dict):
        try:
            return await _load_frame_source_from_config(
                frame_key,
                source_config,
                session,
                context,
            )
        except _FrameSourceLoadError as exc:
            _record_runtime_error(
                context,
                code="frame_source_failed",
                message=str(exc),
                scope=frame_key,
                source="frame",
                detail=str(exc),
            )
            return []
    source_kind = str(frame_spec.get("source_type") or "dataset").lower()
    source_name = frame_spec.get("source") or frame_spec.get("name") or frame_key
    if source_kind == "dataset":
        try:
            return await fetch_dataset_rows(source_name, session, context)
        except Exception as exc:  # pragma: no cover - dataset fetch failure
            _record_runtime_error(
                context,
                code="frame_source_failed",
                message=f"Failed to load dataset '{source_name}' for frame '{frame_key}'.",
                scope=frame_key,
                source="frame",
                detail=str(exc),
            )
            return []
    if source_kind == "frame":
        nested_frame = await _resolve_frame_runtime(source_name, session, context, visited)
        return nested_frame.rows
    if source_kind == "table":
        tables = context.get("tables")
        if isinstance(tables, dict):
            table_value = tables.get(source_name)
            if isinstance(table_value, list):
                dict_rows = [row for row in table_value if isinstance(row, dict)]
                if dict_rows:
                    return _clone_rows(dict_rows)
            elif isinstance(table_value, dict):
                return _clone_rows([table_value])
        dataset_spec = DATASETS.get(source_name)
        if isinstance(dataset_spec, dict):
            sample_rows = dataset_spec.get("sample_rows")
            if isinstance(sample_rows, list):
                dict_rows = [row for row in sample_rows if isinstance(row, dict)]
                if dict_rows:
                    return _clone_rows(dict_rows)
        return []
    _record_runtime_error(
        context,
        code="frame_source_unknown",
        message=f"Unsupported frame source type '{source_kind}'.",
        scope=frame_key,
        source="frame",
        detail=source_kind,
    )
    return []


async def _load_frame_source_from_config(
    frame_key: str,
    source_config: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    kind = str(source_config.get("kind") or source_config.get("type") or "file").lower()
    if kind == "file":
        return _load_frame_file_source(
            source_config,
            context=context,
            resolve_placeholders=_resolve_placeholders,
        )
    if kind == "sql":
        return await _load_frame_sql_source(
            source_config,
            session,
            context=context,
            resolve_placeholders=_resolve_placeholders,
        )
    raise _FrameSourceLoadError(f"Unsupported frame source kind '{kind}' for '{frame_key}'.")


async def _evaluate_frame_pipeline(
    plan: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    visited: Set[str],
) -> _N3Frame:
    if not isinstance(plan, dict):
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, "Invalid frame pipeline payload.")
    root = plan.get("root")
    if not isinstance(root, str) or not root:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, "Frame pipeline requires a root frame.")
    base_frame = await _resolve_frame_runtime(root, session, context, set(visited))
    operations: List[Dict[str, Any]] = []
    for operation in plan.get("operations") or []:
        if not isinstance(operation, dict):
            continue
        if operation.get("op") == "join":
            join_target = operation.get("join_target")
            if not join_target:
                raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, "Join operation requires a target frame.")
            join_frame = await _resolve_frame_runtime(join_target, session, context, set(visited))
            op_payload = dict(operation)
            op_payload["join_rows"] = join_frame.rows
            op_payload["join_schema"] = join_frame.spec.get("columns")
            operations.append(op_payload)
            continue
        operations.append(operation)
    schema_payload = plan.get("schema") or {}
    try:
        rows = _execute_frame_pipeline_plan(
            root,
            schema_payload,
            base_frame.rows,
            operations,
            context=context,
            evaluate_expression=_evaluate_dataset_expression,
            runtime_truthy=_runtime_truthy,
        )
    except _FramePipelineExecutionError as exc:
        _record_runtime_error(
            context,
            code="frame_pipeline_failed",
            message=str(exc),
            scope=root,
            source="frame",
            detail=str(exc),
            severity="error",
        )
        raise
    derived_spec = _build_pipeline_frame_spec(schema_payload, base_frame.spec, f"{root}__pipeline")
    return _N3Frame(derived_spec.get("name") or root, derived_spec, rows)


async def _resolve_frame_pipeline_value(
    value: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    plan = value.get("__frame_pipeline__")
    if not isinstance(plan, dict):
        return []
    try:
        derived = await _evaluate_frame_pipeline(plan, session, context, set())
    except _FramePipelineExecutionError as exc:
        raise _frame_error(400, _FRAME_ERROR_INVALID_PARAMS, str(exc))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Frame pipeline evaluation failed")
        raise _frame_error(500, "FRAME_PIPELINE_FAILED", "Frame pipeline evaluation failed.")
    return derived.rows
'''
)

__all__ = ["FRAMES_SECTION"]
