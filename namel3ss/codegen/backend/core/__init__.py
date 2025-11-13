"""Core backend generation entrypoints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .generator import generate_backend
from .runtime import _render_runtime_module
from ..state import BackendState, build_backend_state

_CORE_MODULE_NAME = "namel3ss.codegen.backend._core_impl"
_CORE_MODULE_PATH = Path(__file__).resolve().parent.parent / "core.py"


def _load_core_impl() -> object:
    module = sys.modules.get(_CORE_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_CORE_MODULE_NAME, _CORE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load core implementation from {_CORE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CORE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def compile_dataset_to_sql(dataset, metadata, context):
    module = _load_core_impl()
    return module.compile_dataset_to_sql(dataset, metadata, context)


__all__ = [
    "BackendState",
    "build_backend_state",
    "compile_dataset_to_sql",
    "generate_backend",
    "_render_runtime_module",
]
