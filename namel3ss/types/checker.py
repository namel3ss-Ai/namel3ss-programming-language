"""Static type checker for Namel3ss programs."""

from __future__ import annotations

from .checker_original_backup import (
    TypeEnvironment,
    TypeScope,
    AppTypeChecker,
    check_module,
    check_app,
)

from .stdlib_checker import (
    EnhancedAppTypeChecker,
    check_app_with_stdlib,
    check_program_with_stdlib,
    check_module_with_stdlib,
)

# Default exports use stdlib-enhanced checkers
def check_app_enhanced(app, *, path=None):
    """Enhanced app checker with stdlib validation."""
    return check_app_with_stdlib(app, path=path)

def check_module_enhanced(module):
    """Enhanced module checker with stdlib validation."""
    return check_module_with_stdlib(module)

__all__ = [
    # Original implementations (for compatibility)
    "TypeEnvironment",
    "TypeScope", 
    "AppTypeChecker",
    "check_module",
    "check_app",
    
    # Enhanced implementations with stdlib support
    "EnhancedAppTypeChecker",
    "check_app_with_stdlib",
    "check_program_with_stdlib", 
    "check_module_with_stdlib",
    "check_app_enhanced",
    "check_module_enhanced",
]
