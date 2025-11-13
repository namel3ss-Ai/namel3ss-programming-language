"""Core backend generation entrypoints."""

from __future__ import annotations

from .generator import generate_backend
from .runtime import _render_runtime_module
from .sql_compiler import compile_dataset_to_sql
from ..state import BackendState, build_backend_state

__all__ = [
    "BackendState",
    "build_backend_state",
    "compile_dataset_to_sql",
    "generate_backend",
    "_render_runtime_module",
]
