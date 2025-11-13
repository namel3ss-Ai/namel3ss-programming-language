"""Render database scaffolding."""

from __future__ import annotations

import re
import textwrap
from typing import Optional

from ..state import BackendState

__all__ = ["_render_database_module", "_database_env_var"]


def _render_database_module(state: BackendState) -> str:
    env_var = _database_env_var(state.app.get("database"))
    template = f'''
"""Database configuration for the generated FastAPI backend."""

from __future__ import annotations

import os
from typing import AsyncGenerator

try:
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "sqlalchemy is required for database scaffolding. "
        "Install it with: pip install 'namel3ss[sql]'"
    ) from exc

DATABASE_URL_ENV = {env_var!r}
DEFAULT_DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/app"


def _build_database_url() -> str:
    """Return the connection string for the application's primary database."""

    return os.getenv(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)


engine: AsyncEngine = create_async_engine(_build_database_url(), echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an :class:`AsyncSession` for FastAPI dependencies."""

    async with SessionLocal() as session:
        yield session
'''
    return textwrap.dedent(template).strip()


def _database_env_var(database_name: Optional[str]) -> str:
    if not database_name:
        return "NAMEL3SS_DATABASE_URL"
    alias = re.sub(r"[^A-Z0-9]+", "_", str(database_name).upper()).strip("_") or "DEFAULT"
    return f"NAMEL3SS_POSTGRES_{alias}_URL"
