"""
Dependency injection for FastAPI routes.

Provides shared dependencies like database connections, settings, and authentication.
"""

from typing import AsyncGenerator, Optional

import asyncpg
from fastapi import Header, Request

from config.settings import Settings, get_settings
from repository import Postgres{{ entity_name }}Repository, {{ entity_name }}Repository


# Database connection pool (initialized at startup)
_db_pool: Optional[asyncpg.Pool] = None


async def init_db_pool(settings: Settings) -> asyncpg.Pool:
    """
    Initialize database connection pool.
    
    Should be called during application startup.
    
    Args:
        settings: Application settings
        
    Returns:
        asyncpg connection pool
    """
    global _db_pool
    
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(
            str(settings.database_url),
            min_size=1,
            max_size=settings.db_pool_size,
            max_inactive_connection_lifetime=settings.db_pool_timeout,
            command_timeout=60,
        )
    
    return _db_pool


async def close_db_pool() -> None:
    """
    Close database connection pool.
    
    Should be called during application shutdown.
    """
    global _db_pool
    
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None


def get_db_pool() -> asyncpg.Pool:
    """
    Get database connection pool.
    
    Raises:
        RuntimeError: If pool not initialized
    """
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool() at startup.")
    return _db_pool


async def get_repository() -> {{ entity_name }}Repository:
    """
    Get repository instance.
    
    FastAPI dependency that provides a repository for database operations.
    
    Returns:
        {{ entity_name }}Repository instance
    """
    pool = get_db_pool()
    return Postgres{{ entity_name }}Repository(pool)


def get_tenant_id(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    settings: Settings = get_settings(),
) -> Optional[str]:
    """
    Extract tenant ID from request headers.
    
    If multi-tenancy is enabled, this extracts the tenant identifier
    from the configured header. Returns None if multi-tenancy is disabled.
    
    Args:
        x_tenant_id: Tenant ID from header
        settings: Application settings
        
    Returns:
        Tenant ID if multi-tenancy enabled, None otherwise
    """
    if not settings.enable_multi_tenancy:
        return None
    
    return x_tenant_id


async def get_current_user(
    request: Request,
    settings: Settings = get_settings(),
) -> Optional[dict]:
    """
    Get current authenticated user.
    
    Extension point for authentication. Current implementation is a placeholder.
    
    To add authentication:
    1. Install authentication library (e.g., python-jose, passlib)
    2. Implement token validation
    3. Return user information
    4. Add this as a dependency to protected routes
    
    Args:
        request: FastAPI request
        settings: Application settings
        
    Returns:
        User information if authenticated, None otherwise
    """
    # TODO: Implement authentication
    # Example with JWT:
    # authorization = request.headers.get("Authorization")
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Not authenticated")
    # token = authorization.split(" ")[1]
    # payload = decode_jwt(token)
    # return {"user_id": payload["sub"], "username": payload["username"]}
    
    return None


def require_auth(user: Optional[dict] = get_current_user) -> dict:
    """
    Require authentication for a route.
    
    Use as a dependency on protected routes.
    
    Args:
        user: Current user from get_current_user
        
    Returns:
        User information
        
    Raises:
        HTTPException: If not authenticated
    """
    from fastapi import HTTPException, status
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = get_settings(),
) -> str:
    """
    Require API key for a route.
    
    Use as a dependency for API key authentication.
    
    Args:
        x_api_key: API key from header
        settings: Application settings
        
    Returns:
        API key
        
    Raises:
        HTTPException: If API key invalid or missing
    """
    from fastapi import HTTPException, status
    
    # TODO: Implement API key validation
    # Example:
    # if not x_api_key:
    #     raise HTTPException(status_code=401, detail="API key required")
    # if not validate_api_key(x_api_key):
    #     raise HTTPException(status_code=403, detail="Invalid API key")
    
    return x_api_key or ""
