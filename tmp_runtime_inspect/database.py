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

DATABASE_URL_ENV = 'NAMEL3SS_POSTGRES_DEMO_DB_URL'
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