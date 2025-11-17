from __future__ import annotations

from textwrap import dedent

FRAMES_SECTION = dedent(
    '''
from namel3ss.codegen.backend.core.runtime.frames import (
    project_frame_rows as _project_frame_rows,
)


async def fetch_frame_rows(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return await _fetch_frame_rows_internal(key, session, context, set())


async def _fetch_frame_rows_internal(
    key: str,
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    visited: Set[str],
) -> List[Dict[str, Any]]:
    if not key:
        return []
    frame_key = str(key)
    if frame_key in visited:
        _record_runtime_error(
            context,
            code="frame_recursion_detected",
            message=f"Cyclic frame dependency detected for '{frame_key}'.",
            scope=frame_key,
            source="frame",
            severity="error",
        )
        return []
    visited.add(frame_key)
    frame_spec = FRAMES.get(frame_key)
    if not isinstance(frame_spec, dict):
        _record_runtime_error(
            context,
            code="frame_missing",
            message=f"Frame '{frame_key}' is not defined.",
            scope=frame_key,
            source="frame",
            severity="error",
        )
        return []
    base_rows = await _load_frame_source_rows(frame_key, frame_spec, session, context, visited)
    if not base_rows:
        examples = frame_spec.get("examples")
        if isinstance(examples, list):
            example_rows = [row for row in examples if isinstance(row, dict)]
            if example_rows:
                base_rows = _clone_rows(example_rows)
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
    return shaped_rows


async def _load_frame_source_rows(
    frame_key: str,
    frame_spec: Dict[str, Any],
    session: Optional[AsyncSession],
    context: Dict[str, Any],
    visited: Set[str],
) -> List[Dict[str, Any]]:
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
        return await _fetch_frame_rows_internal(source_name, session, context, visited)
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
'''
)

__all__ = ["FRAMES_SECTION"]
