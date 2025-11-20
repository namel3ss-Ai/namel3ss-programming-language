"""Constants, type aliases, and type casters for frame execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

try:  # pragma: no cover - optional dependency for dataframe execution
    import polars as pl  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when polars is unavailable
    pl = None  # type: ignore

try:  # pragma: no cover - optional dependency for dataframe execution
    import pandas as pd  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when pandas is unavailable
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency for real parquet export
    import pyarrow as pa  # type: ignore[import]
    import pyarrow.parquet as pq  # type: ignore[import]
except Exception:  # pragma: no cover - fallback when pyarrow is unavailable
    pa = None  # type: ignore
    pq = None  # type: ignore


# Type aliases
RuntimeExpressionEvaluator = Callable[
    [Any, Dict[str, Any], Dict[str, Any], Optional[Sequence[Dict[str, Any]]], Optional[str]],
    Any,
]
RuntimePlaceholderResolver = Callable[[Any, Dict[str, Any]], Any]
RuntimeTruthiness = Callable[[Any], bool]
RuntimeErrorRecorder = Callable[..., Dict[str, Any]]

# Frame limits
DEFAULT_FRAME_LIMIT = 100
MAX_FRAME_LIMIT = 1000

# Type casters for column types
_TYPE_CASTERS: Dict[str, Callable[[Any], Any]] = {
    "string": lambda value: str(value) if value is not None else None,
    "text": lambda value: str(value) if value is not None else None,
    "int": lambda value: int(value) if value is not None else None,
    "integer": lambda value: int(value) if value is not None else None,
    "number": lambda value: float(value) if value is not None else None,
    "float": lambda value: float(value) if value is not None else None,
    "decimal": lambda value: float(value) if value is not None else None,
    "bool": lambda value: bool(value) if value is not None else None,
    "boolean": lambda value: bool(value) if value is not None else None,
}

# Polars type mapping
_POLARS_TYPE_MAP: Dict[str, str] = {
    "string": "Utf8",
    "text": "Utf8",
    "int": "Int64",
    "integer": "Int64",
    "number": "Float64",
    "float": "Float64",
    "decimal": "Float64",
    "bool": "Boolean",
    "boolean": "Boolean",
    "datetime": "Datetime",
    "timestamp": "Datetime",
    "date": "Date",
    "time": "Time",
}


# Exceptions
class FrameSourceLoadError(RuntimeError):
    """Raised when external frame source loading fails."""


class FramePipelineExecutionError(RuntimeError):
    """Raised when frame pipeline execution fails."""


__all__ = [
    "pl",
    "pd",
    "pa",
    "pq",
    "RuntimeExpressionEvaluator",
    "RuntimePlaceholderResolver",
    "RuntimeTruthiness",
    "RuntimeErrorRecorder",
    "DEFAULT_FRAME_LIMIT",
    "MAX_FRAME_LIMIT",
    "_TYPE_CASTERS",
    "_POLARS_TYPE_MAP",
    "FrameSourceLoadError",
    "FramePipelineExecutionError",
]
