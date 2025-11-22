"""API package for {{ project_name }}."""

from .routes import router
from .dependencies import (
    get_repository,
    get_settings,
    get_tenant_id,
    get_current_user,
    require_auth,
    require_api_key,
    init_db_pool,
    close_db_pool,
)
from .security import User, create_access_token
from .errors import (
    validation_exception_handler,
    general_exception_handler,
    not_found_handler,
    create_error_response,
)

__all__ = [
    "router",
    "get_repository",
    "get_settings",
    "get_tenant_id",
    "get_current_user",
    "require_auth",
    "require_api_key",
    "init_db_pool",
    "close_db_pool",
    "User",
    "create_access_token",
    "validation_exception_handler",
    "general_exception_handler",
    "not_found_handler",
    "create_error_response",
]
