"""
Dependency injection for FastAPI routes.

Provides shared dependencies like database connections, settings, and authentication.
"""

from typing import AsyncGenerator, Optional

import asyncpg
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import Settings, get_settings
from repository import Postgres{{ entity_name }}Repository, {{ entity_name }}Repository
from api.security import (
    User,
    decode_jwt_token,
    token_data_to_user,
    AuthenticationError,
    TokenExpiredError,
    TokenInvalidError,
    TokenValidationError,
)


# Database connection pool (initialized at startup)
_db_pool: Optional[asyncpg.Pool] = None

# Security scheme for bearer token authentication
security = HTTPBearer(auto_error=False)


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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings),
) -> Optional[User]:
    """
    Get current authenticated user from JWT bearer token.
    
    Extracts bearer token from Authorization header, validates it, and returns
    the authenticated user. Returns None if no token provided or if auth is disabled.
    
    This dependency can be used directly in routes that need optional authentication,
    or via require_auth() for routes requiring authentication.
    
    **Integration with External Identity Providers:**
    
    To integrate with Auth0, Cognito, Azure AD, or other IdPs:
    
    1. Configure JWT validation settings in environment:
       - JWT_SECRET_KEY: For HS256, or public key for RS256
       - JWT_ALGORITHM: "RS256" for RSA signatures (most IdPs)
       - JWT_ISSUER: Your IdP's issuer URL
       - JWT_AUDIENCE: Your application's audience/client ID
    
    2. For JWKS (public key discovery):
       - Extend this function to fetch JWKS from IdP
       - Cache public keys for performance
       - Validate using public key instead of shared secret
    
    3. Customize User model mapping:
       - Modify token_data_to_user() in security.py
       - Map IdP-specific claims to User fields
       - Extract roles from IdP format (e.g., groups, scopes)
    
    Args:
        credentials: Bearer token from Authorization header
        settings: Application settings
        
    Returns:
        User object if authenticated, None if no token or auth disabled
        
    Raises:
        HTTPException 401: If token is invalid, expired, or validation fails
    """
    # Allow disabling auth for local development only
    if settings.auth_disabled:
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication is disabled but environment is production",
            )
        # Return None to indicate no auth in dev mode
        return None
    
    # No token provided - return None (let require_auth handle if needed)
    if not credentials:
        return None
    
    token = credentials.credentials
    
    try:
        # Decode and validate JWT token
        token_data = decode_jwt_token(token, settings)
        
        # Convert token data to User model
        user = token_data_to_user(token_data)
        
        # Additional user validation can be added here
        # Example: Check if user is active in database, verify tenant access, etc.
        
        return user
        
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    
    except TokenInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    
    except TokenValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    
    except Exception as e:
        # Log unexpected errors but don't leak details to client
        import logging
        logging.error(f"Unexpected authentication error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def require_auth(user: Optional[User] = Depends(get_current_user)) -> User:
    """
    Require authentication for a route.
    
    Use as a dependency on protected routes to enforce authentication.
    Raises 401 if user is not authenticated.
    
    **Usage:**
    ```python
    @router.get("/protected")
    async def protected_route(user: User = Depends(require_auth)):
        return {"user_id": user.id, "message": "Access granted"}
    ```
    
    **Role-Based Access Control:**
    
    To require specific roles, create custom dependencies:
    ```python
    def require_admin(user: User = Depends(require_auth)) -> User:
        if "admin" not in user.roles:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user
    ```
    
    Args:
        user: Current user from get_current_user
        
    Returns:
        User object
        
    Raises:
        HTTPException 401: If not authenticated
    """
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
