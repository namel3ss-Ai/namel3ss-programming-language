"""Static type checker for Namel3ss programs."""

from __future__ import annotations

from .checker_original_backup import (
    TypeEnvironment,
    TypeScope,
    AppTypeChecker,
    check_module,
    check_app,
)

__all__ = [
    "TypeEnvironment",
    "TypeScope",
    "AppTypeChecker",
    "check_module",
    "check_app",
]
