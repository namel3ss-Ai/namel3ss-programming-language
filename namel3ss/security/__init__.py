"""
Security subsystem for Namel3ss - capability-based security model.

This package implements the security and capability model for Namel3ss,
providing compile-time and runtime enforcement of security constraints.

Modules:
--------
- config: Security configuration and environment profiles
- validation: Static validation of security constraints
- runtime: Runtime security guards and enforcement
- errors: Security-specific exceptions
"""

from .config import (
    SecurityConfig,
    get_security_config,
    set_environment,
    load_security_config,
)
from .validation import (
    validate_tool_access,
    validate_capability_grant,
    validate_permission_level,
    validate_security_policy,
    SecurityValidator,
)
from .runtime import (
    SecurityGuard,
    get_security_guard,
    RateLimiter,
    TokenCounter,
    CostTracker,
)

__all__ = [
    # Configuration
    "SecurityConfig",
    "get_security_config",
    "set_environment",
    "load_security_config",
    
    # Validation
    "validate_tool_access",
    "validate_capability_grant",
    "validate_permission_level",
    "validate_security_policy",
    "SecurityValidator",
    
    # Runtime
    "SecurityGuard",
    "get_security_guard",
    "RateLimiter",
    "TokenCounter",
    "CostTracker",
]
