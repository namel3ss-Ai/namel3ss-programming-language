"""Compatibility shim for the modular backend generator implementation.

Historically the backend generator lived in this module as a monolithic
implementation. The logic now resides in dedicated modules under
``namel3ss.codegen.backend.core``. This file re-exports the public helpers so
imports such as ``from namel3ss.codegen.backend.core import generate_backend``
continue to work.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency at generation time
    from sqlalchemy import bindparam as _sa_bindparam
    from sqlalchemy import column as _sa_column
    from sqlalchemy import table as _sa_table
    from sqlalchemy import text as _sa_text
    from sqlalchemy import update as _sa_update
    from sqlalchemy.sql import Select as _sa_select
except ImportError:  # pragma: no cover - optional dependency
    def _sa_text(sql: str) -> str:  # type: ignore
        return sql

    _sa_bindparam = None  # type: ignore
    _sa_update = None  # type: ignore
    _sa_table = None  # type: ignore
    _sa_column = None  # type: ignore
    _sa_select = Any  # type: ignore

from .core.app_module import _render_app_module
from .core.database import _database_env_var, _render_database_module
from .core.generator import generate_backend
from .core.packages import (
    _render_custom_api_stub,
    _render_custom_readme,
    _render_generated_package,
    _render_helpers_package,
)
from .core.schemas import _render_schemas_module
from .core.routers import (
    _render_component_endpoint,
    _render_experiments_router_module,
    _render_insight_endpoint,
    _render_insights_router_module,
    _render_models_router_module,
    _render_page_endpoint,
    _render_pages_router_module,
    _render_training_router_module,
    _render_routers_package,
)
from .core.runtime import _page_to_dict, _render_page_function, _render_runtime_module
from .core.sql_compiler import compile_dataset_to_sql
from .core.utils import _assign_literal, _format_literal

text = _sa_text
bindparam = _sa_bindparam
update = _sa_update
sql_table = _sa_table
column = _sa_column
Select = _sa_select  # type: ignore

__all__ = [
    "generate_backend",
    "compile_dataset_to_sql",
    "_render_database_module",
    "_render_schemas_module",
    "_render_custom_api_stub",
    "_render_generated_package",
    "_render_helpers_package",
    "_render_custom_readme",
    "_render_routers_package",
    "_render_insights_router_module",
    "_render_models_router_module",
    "_render_experiments_router_module",
    "_render_training_router_module",
    "_render_pages_router_module",
    "_render_app_module",
    "_render_runtime_module",
    "_assign_literal",
    "_format_literal",
    "_page_to_dict",
    "_render_page_function",
    "_render_page_endpoint",
    "_render_component_endpoint",
    "_render_insight_endpoint",
    "_database_env_var",
    "text",
    "bindparam",
    "update",
    "sql_table",
    "column",
    "Select",
]
