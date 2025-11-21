"""
RLHF Database Connection - Database initialization and session management.

This module provides:
- Database engine creation with connection pooling
- Async session management
- Migration support with Alembic
- Health checks and connection validation
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool

from .models import Base
from .errors import RLHFConfigurationError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and sessions.
    
    Handles:
    - Engine creation with connection pooling
    - Async session factory
    - Schema creation
    - Health checks
    """
    
    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
    ):
        """
        Initialize database manager.
        
        Args:
            database_url: SQLAlchemy database URL
                Examples:
                - postgresql+asyncpg://user:pass@localhost/rlhf
                - sqlite+aiosqlite:///./rlhf.db
            echo: Log SQL statements
            pool_size: Number of connections to maintain
            max_overflow: Max connections above pool_size
            pool_timeout: Seconds to wait for connection
            pool_recycle: Recycle connections after seconds
        
        Raises:
            RLHFConfigurationError: If URL is invalid
        """
        if not database_url:
            raise RLHFConfigurationError(
                "Database URL is required",
                error_code="RLHF048",
            )
        
        self.database_url = database_url
        self.echo = echo
        
        # Choose pooling strategy based on database type
        if "sqlite" in database_url:
            # SQLite doesn't support connection pooling
            poolclass = NullPool
            pool_kwargs = {}
        else:
            # PostgreSQL/MySQL use connection pooling
            poolclass = QueuePool
            pool_kwargs = {
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": pool_timeout,
                "pool_recycle": pool_recycle,
            }
        
        # Create async engine
        try:
            self.engine: AsyncEngine = create_async_engine(
                database_url,
                echo=echo,
                poolclass=poolclass,
                **pool_kwargs,
            )
        except Exception as e:
            raise RLHFConfigurationError(
                f"Failed to create database engine: {e}",
                error_code="RLHF048",
            ) from e
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        logger.info(f"Database manager initialized: {self._safe_url()}")
    
    def _safe_url(self) -> str:
        """Get database URL with password masked."""
        url = self.database_url
        if "@" in url:
            # Mask password: user:password@host -> user:***@host
            parts = url.split("@")
            if ":" in parts[0]:
                user_pass = parts[0].split(":")
                parts[0] = f"{user_pass[0]}:***"
            return "@".join(parts)
        return url
    
    async def create_tables(self) -> None:
        """
        Create all tables if they don't exist.
        
        Raises:
            RLHFConfigurationError: If table creation fails
        """
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            raise RLHFConfigurationError(
                f"Failed to create database tables: {e}",
                error_code="RLHF048",
            ) from e
    
    async def drop_tables(self) -> None:
        """
        Drop all tables.
        
        WARNING: This will delete all data!
        
        Raises:
            RLHFConfigurationError: If table drop fails
        """
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.warning("Database tables dropped")
        except Exception as e:
            raise RLHFConfigurationError(
                f"Failed to drop database tables: {e}",
                error_code="RLHF048",
            ) from e
    
    async def health_check(self) -> bool:
        """
        Check database connection health.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Create a new database session.
        
        Usage:
            async with db.session() as session:
                # Use session
                result = await session.execute(...)
        
        Yields:
            AsyncSession: Database session
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self) -> None:
        """Close database engine and connections."""
        await self.engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(
    database_url: str,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> DatabaseManager:
    """
    Initialize global database manager.
    
    Args:
        database_url: SQLAlchemy database URL
        echo: Log SQL statements
        pool_size: Connection pool size
        max_overflow: Max overflow connections
    
    Returns:
        DatabaseManager: Initialized manager
    
    Raises:
        RLHFConfigurationError: If initialization fails
    """
    global _db_manager
    _db_manager = DatabaseManager(
        database_url=database_url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
    )
    return _db_manager


def get_database() -> DatabaseManager:
    """
    Get global database manager.
    
    Returns:
        DatabaseManager: Global instance
    
    Raises:
        RLHFConfigurationError: If not initialized
    """
    if _db_manager is None:
        raise RLHFConfigurationError(
            "Database not initialized. Call initialize_database() first.",
            error_code="RLHF048",
        )
    return _db_manager


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session from global manager.
    
    Usage:
        async with get_session() as session:
            # Use session
            result = await session.execute(...)
    
    Yields:
        AsyncSession: Database session
    
    Raises:
        RLHFConfigurationError: If database not initialized
    """
    db = get_database()
    async with db.session() as session:
        yield session
