"""Authentication package."""

from .models import User, ProjectMember, Role
from .security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from .dependencies import (
    get_current_user,
    get_current_active_user,
    get_current_superuser,
    get_optional_user,
    require_viewer_access,
    require_editor_access,
    require_owner_access,
)

__all__ = [
    "User",
    "ProjectMember",
    "Role",
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "get_current_user",
    "get_current_active_user",
    "get_current_superuser",
    "get_optional_user",
    "require_viewer_access",
    "require_editor_access",
    "require_owner_access",
]
