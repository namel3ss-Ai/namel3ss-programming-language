"""
Security base types and permissions.

Re-exports security types from AST module for use in security subsystem.
"""

from namel3ss.ast.security import PermissionLevel

__all__ = ["PermissionLevel"]