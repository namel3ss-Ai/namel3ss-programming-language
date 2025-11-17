"""Utility helpers and dependency diagnostics for Namel3ss."""

from .dependencies import DependencyReport, iter_dependency_reports, require_dependency

__all__ = [
    "DependencyReport",
    "iter_dependency_reports",
    "require_dependency",
]
