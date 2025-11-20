"""Authentication dependencies for FastAPI."""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from n3_server.db.session import get_db
from n3_server.auth.models import User, Role, ProjectMember
from n3_server.auth.security import decode_token, validate_token_type
from n3_server.db.models import Project

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP bearer credentials with JWT token
        db: Database session
    
    Returns:
        Current user
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decode token
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise credentials_exception
    
    # Validate token type
    if not validate_token_type(payload, "access"):
        raise credentials_exception
    
    # Get user ID from token
    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current user from token
    
    Returns:
        Current active user
    
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current superuser.
    
    Args:
        current_user: Current user from token
    
    Returns:
        Current superuser
    
    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Useful for endpoints that work both with and without authentication.
    
    Args:
        credentials: Optional HTTP bearer credentials
        db: Database session
    
    Returns:
        Current user or None
    """
    if credentials is None:
        return None
    
    try:
        token = credentials.credentials
        payload = decode_token(token)
        
        if payload is None or not validate_token_type(payload, "access"):
            return None
        
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user and user.is_active:
            return user
        
        return None
    except Exception:
        return None


async def check_project_access(
    project_id: str,
    user: User,
    required_role: Role,
    db: AsyncSession,
) -> bool:
    """
    Check if user has required access to a project.
    
    Args:
        project_id: Project ID
        user: User to check
        required_role: Minimum required role
        db: Database session
    
    Returns:
        True if user has access, False otherwise
    """
    # Superusers have access to everything
    if user.is_superuser:
        return True
    
    # Check if user owns the project
    result = await db.execute(
        select(Project).where(
            Project.id == project_id,
            Project.owner_id == user.id,
        )
    )
    project = result.scalar_one_or_none()
    
    if project:
        return True  # Owner has all permissions
    
    # Check project membership
    result = await db.execute(
        select(ProjectMember).where(
            ProjectMember.project_id == project_id,
            ProjectMember.user_id == user.id,
        )
    )
    membership = result.scalar_one_or_none()
    
    if membership is None:
        return False
    
    # Check role hierarchy: owner > editor > viewer
    role_hierarchy = {
        Role.OWNER: 3,
        Role.EDITOR: 2,
        Role.VIEWER: 1,
    }
    
    user_role_level = role_hierarchy.get(membership.role, 0)
    required_role_level = role_hierarchy.get(required_role, 0)
    
    return user_role_level >= required_role_level


class ProjectAccessChecker:
    """Dependency for checking project access."""
    
    def __init__(self, required_role: Role = Role.VIEWER):
        """
        Initialize access checker.
        
        Args:
            required_role: Minimum required role
        """
        self.required_role = required_role
    
    async def __call__(
        self,
        project_id: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        """
        Check project access and return user if authorized.
        
        Args:
            project_id: Project ID from path
            current_user: Current authenticated user
            db: Database session
        
        Returns:
            Current user if authorized
        
        Raises:
            HTTPException: If user doesn't have access
        """
        has_access = await check_project_access(
            project_id,
            current_user,
            self.required_role,
            db,
        )
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {self.required_role.value}",
            )
        
        return current_user


# Convenience functions for different access levels
def require_viewer_access() -> ProjectAccessChecker:
    """Require viewer access to project."""
    return ProjectAccessChecker(Role.VIEWER)


def require_editor_access() -> ProjectAccessChecker:
    """Require editor access to project."""
    return ProjectAccessChecker(Role.EDITOR)


def require_owner_access() -> ProjectAccessChecker:
    """Require owner access to project."""
    return ProjectAccessChecker(Role.OWNER)
