"""Core backend generation entrypoints."""

from __future__ import annotations

from .generator import generate_backend
from .runtime import _render_runtime_module
from ..state import BackendState, build_backend_state

# SQL compiler for dataset operations
from .sql_compiler import compile_dataset_to_sql

# Import new runtime modules if they exist
try:
    from .runtime.realtime import (
        init_redis,
        close_redis,
        broadcast_dataset_change,
        subscribe_to_dataset_changes,
        DatasetChangeHandler,
    )
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    init_redis = close_redis = broadcast_dataset_change = None
    subscribe_to_dataset_changes = DatasetChangeHandler = None

__all__ = [
    "BackendState", 
    "build_backend_state",
    "generate_backend",
    "_render_runtime_module",
    "compile_dataset_to_sql",
]

# Only export realtime functions if available
if REALTIME_AVAILABLE:
    __all__.extend([
        "init_redis",
        "close_redis", 
        "broadcast_dataset_change",
        "subscribe_to_dataset_changes",
        "DatasetChangeHandler",
    ])
