"""
Security utilities for authentication and authorization.

Provides JWT token validation, user extraction, and security helpers
for FastAPI dependency injection.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
from pydantic import BaseModel, Field

from config.settings import Settings


class User(BaseModel):
    """
    Authenticated user model.
    
    This represents a user extracted from a validated JWT token.
    Customize fields based on your identity provider's claims.
    """
    
    id: str = Field(..., description="Unique user identifier (typically 'sub' claim)")
    email: Optional[str] = Field(None, description="User email address")
    username: Optional[str] = Field(None, description="Username")
    roles: list[str] = Field(default_factory=list, description="User roles/scopes")
    is_active: bool = Field(default=True, description="Whether user account is active")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier for multi-tenancy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional user claims")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "id": "user_123",
                "email": "user@example.com",
                "username": "johndoe",
                "roles": ["user", "admin"],
                "is_active": True,
                "tenant_id": "tenant_abc",
                "metadata": {"department": "engineering"}
            }
        }


class TokenData(BaseModel):
    """JWT token payload data."""
    
    sub: str = Field(..., description="Subject (user ID)")
    exp: Optional[int] = Field(None, description="Expiration timestamp")
    iat: Optional[int] = Field(None, description="Issued at timestamp")
    iss: Optional[str] = Field(None, description="Issuer")
    aud: Optional[str] = Field(None, description="Audience")
    email: Optional[str] = None
    username: Optional[str] = None
    roles: Optional[list[str]] = Field(default_factory=list)
    tenant_id: Optional[str] = None


class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    pass


class TokenExpiredError(AuthenticationError):
    """Raised when JWT token has expired."""
    pass


class TokenInvalidError(AuthenticationError):
    """Raised when JWT token is invalid."""
    pass


class TokenValidationError(AuthenticationError):
    """Raised when JWT token fails validation."""
    pass


def decode_jwt_token(token: str, settings: Settings) -> TokenData:
    """
    Decode and validate JWT token.
    
    Validates token signature, expiration, issuer, and audience claims.
    Returns decoded token payload as TokenData.
    
    Args:
        token: JWT token string
        settings: Application settings with JWT configuration
        
    Returns:
        TokenData with validated claims
        
    Raises:
        TokenExpiredError: If token has expired
        TokenInvalidError: If token signature is invalid
        TokenValidationError: If token fails validation (issuer, audience, etc.)
    """
    try:
        # Prepare validation options
        options = {
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "require": ["sub", "exp"],
        }
        
        # Build verification kwargs
        decode_kwargs: Dict[str, Any] = {
            "jwt": token,
            "key": settings.jwt_secret_key,
            "algorithms": [settings.jwt_algorithm],
            "options": options,
        }
        
        # Add issuer validation if configured
        if settings.jwt_issuer:
            decode_kwargs["issuer"] = settings.jwt_issuer
        
        # Add audience validation if configured
        if settings.jwt_audience:
            decode_kwargs["audience"] = settings.jwt_audience
        
        # Decode and validate token
        payload = jwt.decode(**decode_kwargs)
        
        # Extract and validate claims
        token_data = TokenData(
            sub=payload["sub"],
            exp=payload.get("exp"),
            iat=payload.get("iat"),
            iss=payload.get("iss"),
            aud=payload.get("aud"),
            email=payload.get("email"),
            username=payload.get("username"),
            roles=payload.get("roles", []),
            tenant_id=payload.get("tenant_id"),
        )
        
        return token_data
        
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredError("Token has expired") from e
    
    except jwt.InvalidSignatureError as e:
        raise TokenInvalidError("Invalid token signature") from e
    
    except jwt.InvalidTokenError as e:
        raise TokenInvalidError(f"Invalid token: {str(e)}") from e
    
    except jwt.InvalidIssuerError as e:
        raise TokenValidationError(f"Invalid token issuer: {str(e)}") from e
    
    except jwt.InvalidAudienceError as e:
        raise TokenValidationError(f"Invalid token audience: {str(e)}") from e
    
    except Exception as e:
        raise TokenValidationError(f"Token validation failed: {str(e)}") from e


def token_data_to_user(token_data: TokenData) -> User:
    """
    Convert TokenData to User model.
    
    Maps JWT claims to User fields. Customize this function to match
    your identity provider's claim structure.
    
    Args:
        token_data: Decoded token data
        
    Returns:
        User instance
    """
    return User(
        id=token_data.sub,
        email=token_data.email,
        username=token_data.username,
        roles=token_data.roles or [],
        is_active=True,  # Assume active if token is valid
        tenant_id=token_data.tenant_id,
        metadata={},
    )


def create_access_token(
    user_id: str,
    settings: Settings,
    email: Optional[str] = None,
    username: Optional[str] = None,
    roles: Optional[list[str]] = None,
    tenant_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a new JWT access token.
    
    This is a helper function for testing or for applications that manage
    their own user authentication (e.g., login endpoints).
    
    Args:
        user_id: User identifier (will be 'sub' claim)
        settings: Application settings with JWT configuration
        email: User email (optional)
        username: Username (optional)
        roles: User roles/scopes (optional)
        tenant_id: Tenant identifier (optional)
        expires_delta: Custom expiration time (optional, defaults to settings)
        
    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    now = datetime.utcnow()
    expire = now + expires_delta
    
    # Build claims
    claims: Dict[str, Any] = {
        "sub": user_id,
        "exp": expire,
        "iat": now,
    }
    
    if settings.jwt_issuer:
        claims["iss"] = settings.jwt_issuer
    
    if settings.jwt_audience:
        claims["aud"] = settings.jwt_audience
    
    if email:
        claims["email"] = email
    
    if username:
        claims["username"] = username
    
    if roles:
        claims["roles"] = roles
    
    if tenant_id:
        claims["tenant_id"] = tenant_id
    
    # Encode token
    token = jwt.encode(
        claims,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    
    return token


def verify_user_has_role(user: User, required_role: str) -> bool:
    """
    Check if user has a specific role.
    
    Args:
        user: User instance
        required_role: Role to check
        
    Returns:
        True if user has role, False otherwise
    """
    return required_role in user.roles


def verify_user_has_any_role(user: User, required_roles: list[str]) -> bool:
    """
    Check if user has any of the specified roles.
    
    Args:
        user: User instance
        required_roles: List of roles to check
        
    Returns:
        True if user has any role, False otherwise
    """
    return any(role in user.roles for role in required_roles)


def verify_user_has_all_roles(user: User, required_roles: list[str]) -> bool:
    """
    Check if user has all of the specified roles.
    
    Args:
        user: User instance
        required_roles: List of roles to check
        
    Returns:
        True if user has all roles, False otherwise
    """
    return all(role in user.roles for role in required_roles)
