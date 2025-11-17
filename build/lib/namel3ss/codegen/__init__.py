"""Code generation entry points for Namel3ss."""

from .backend.core import generate_backend
from .frontend import generate_site

__all__ = ["generate_backend", "generate_site"]
