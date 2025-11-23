from __future__ import annotations

from typing import Optional

from namel3ss.ast import App, Module
from namel3ss.lang.parser import parse_module as new_parse_module
from namel3ss.lang.parser import N3SyntaxError

# Legacy parser fallback for compatibility
from .program import LegacyProgramParser


class Parser:
    """Public parser entry point for Namel3ss programs."""

    def __init__(self, source: str = "", *, module_name: Optional[str] = None, path: str = ""):
        self._source = source or ""
        self._module_name = module_name
        self.source_path = path

    def parse(self) -> Module:
        """Parse source into Module AST using the unified parser."""
        try:
            return new_parse_module(self._source, path=self.source_path, module_name=self._module_name)
        except N3SyntaxError as modern_error:
            # Track fallback usage for diagnostics
            _FallbackTracker.record(
                path=self.source_path,
                error_code=getattr(modern_error, "code", ""),
                message=str(modern_error),
            )

            legacy_parser = LegacyProgramParser(self._source, module_name=self._module_name, path=self.source_path)
            try:
                return legacy_parser.parse()
            except Exception:
                raise modern_error

    def parse_app(self) -> App:
        """Parse source and extract the App node."""
        module = self.parse()
        if not module.body:
            raise N3SyntaxError(
                "Module does not contain any declarations",
                path=self.source_path or None,
                line=0,
                code="SYNTAX_MISSING_APP",
            )
        root = module.body[0]
        if not isinstance(root, App):
            raise ValueError("Module body does not contain an app definition")
        if not module.has_explicit_app:
            raise ValueError("Module does not declare an app definition")
        return root


__all__ = ["Parser", "N3SyntaxError", "enable_parser_cache", "disable_parser_cache", "clear_parser_cache"]


# Performance optimizations (optional)
def enable_parser_cache(max_entries: int = 100):
    """
    Enable parser caching for improved performance.
    
    Args:
        max_entries: Maximum number of cached modules (default: 100)
        
    Returns:
        Cache statistics function for monitoring performance
    """
    try:
        from .optimizations import enable_parser_optimizations, get_parser_cache
        enable_parser_optimizations()
        cache = get_parser_cache()
        cache._max_entries = max_entries
        return cache.stats
    except ImportError:
        # Optimizations not available
        def dummy_stats():
            return {"status": "optimizations not available"}
        return dummy_stats


def disable_parser_cache():
    """Disable parser caching and restore original behavior."""
    try:
        from .optimizations import disable_parser_optimizations
        disable_parser_optimizations()
    except ImportError:
        pass  # Already disabled or not available


def clear_parser_cache():
    """Clear all cached parser results."""
    try:
        from .optimizations import get_parser_cache
        cache = get_parser_cache()
        cache.clear()
    except ImportError:
        pass  # Cache not available


class _FallbackTracker:
    """Simple process-local tracker for legacy parser fallbacks."""

    _count: int = 0
    _last_path: Optional[str] = None
    _last_error_code: Optional[str] = None
    _last_message: Optional[str] = None

    @classmethod
    def record(cls, *, path: str, error_code: Optional[str], message: str) -> None:
        cls._count += 1
        cls._last_path = path
        cls._last_error_code = error_code
        cls._last_message = message

    @classmethod
    def snapshot(cls) -> dict:
        return {
            "count": cls._count,
            "last_path": cls._last_path,
            "last_error_code": cls._last_error_code,
            "last_message": cls._last_message,
        }
