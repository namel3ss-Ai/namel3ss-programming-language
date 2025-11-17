from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Union, Literal as TypingLiteral

from namel3ss.ast import (
    CachePolicy,
    ContextValue,
    Expression,
    LayoutMeta,
    LayoutSpec,
    Literal,
    PaginationPolicy,
    StreamingPolicy,
    WindowFrame,
)


class N3SyntaxError(Exception):
    """Custom exception raised when encountering invalid syntax."""

    def __init__(self, message: str, line_no: int, line: str):
        super().__init__(f"Syntax error on line {line_no}: {message}\n{line}")
        self.line_no = line_no
        self.line = line


class ParserBase:
    """Shared parser state and helper utilities."""

    def __init__(self, source: str):
        self.lines: List[str] = source.splitlines()
        self.pos: int = 0
        self.app = None

    # ------------------------------------------------------------------
    # Cursor helpers
    # ------------------------------------------------------------------
    def _peek(self) -> Optional[str]:
        """Return the current line without consuming it."""
        if self.pos < len(self.lines):
            return self.lines[self.pos]
        return None

    def _advance(self) -> Optional[str]:
        """Return the current line and move the cursor forward."""
        line = self._peek()
        self.pos += 1
        return line

    def _indent(self, line: str) -> int:
        """Compute the indentation level (leading spaces) for *line*."""
        return len(line) - len(line.lstrip(' '))

    def _error(
        self,
        message: str,
        line_no: Optional[int] = None,
        line: Optional[str] = None,
    ) -> N3SyntaxError:
        """Create a consistent :class:`N3SyntaxError` instance."""

        if line_no is None:
            line_no = min(self.pos, len(self.lines))
        if line is None and 0 <= self.pos - 1 < len(self.lines):
            line = self.lines[self.pos - 1]
        elif line is None:
            line = ''
        return N3SyntaxError(message, line_no, line)

    # ------------------------------------------------------------------
    # Scalar coercion helpers
    # ------------------------------------------------------------------
    def _parse_bool(self, raw: str) -> bool:
        """Parse a boolean-like string into a bool."""
        value = raw.strip().lower()
        if value in {"true", "yes", "1", "on"}:
            return True
        if value in {"false", "no", "0", "off"}:
            return False
        raise self._error(f"Expected boolean value, found '{raw}'")

    def _parse_context_reference(self, token: str) -> Optional[ContextValue]:
        match = re.match(r'^(ctx|env):([A-Za-z0-9_\.]+)$', token)
        if not match:
            return None
        scope = match.group(1)
        path_text = match.group(2)
        if not path_text:
            return None
        path = [segment for segment in path_text.split('.') if segment]
        if not path:
            return None
        return ContextValue(scope=scope, path=path)

    def _coerce_scalar(self, raw: str) -> Any:
        """Attempt to coerce a scalar configuration value."""
        text = raw.strip()
        if not text:
            return text

        context_ref = self._parse_context_reference(text)
        if context_ref is not None:
            return context_ref
        lower = text.lower()
        if lower in {"true", "false", "null", "none"}:
            if lower in {"true", "false"}:
                return self._parse_bool(text)
            return None
        if re.fullmatch(r"[-+]?\d+", text):
            try:
                return int(text)
            except ValueError:
                pass
        if re.fullmatch(r"[-+]?\d*\.\d+", text):
            try:
                return float(text)
            except ValueError:
                pass
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            inner = text[1:-1]
            context_ref_inner = self._parse_context_reference(inner)
            if context_ref_inner is not None:
                return context_ref_inner
            return inner
        if text.startswith('[') or text.startswith('{') or text.startswith('('):
            try:
                parsed = ast.literal_eval(text)
                return parsed
            except (SyntaxError, ValueError):
                pass
        return text

    def _coerce_expression(self, value: Any) -> Expression:
        if isinstance(value, Expression):
            return value
        if isinstance(value, str):
            try:
                return self._parse_expression(value)
            except N3SyntaxError:
                return Literal(value)
        return Literal(value)

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            text = str(value).strip()
            if not text:
                return None
            match = re.match(r"(-?\d+)", text)
            if not match:
                return None
            return int(match.group(1))
        except (ValueError, TypeError):
            return None

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, ContextValue):
            path = '.'.join(value.path)
            return f"{value.scope}:{path}" if path else value.scope
        if isinstance(value, Literal):
            inner = value.value
            return '' if inner is None else str(inner)
        if isinstance(value, Expression):
            return str(value)
        return str(value)

    # ------------------------------------------------------------------
    # Block helpers
    # ------------------------------------------------------------------
    def _parse_kv_block(self, parent_indent: int) -> Dict[str, Any]:
        """Parse an indented key/value block and return a dictionary."""
        config: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            match = re.match(r'([\w\.\s]+):\s*(.*)$', stripped)
            if not match:
                raise self._error("Expected 'key: value' inside block", self.pos + 1, line)
            key = match.group(1).strip()
            remainder = match.group(2)
            self._advance()
            if remainder == "":
                nested = self._parse_kv_block(indent)
                config[key] = nested
            else:
                value = self._coerce_scalar(remainder)
                config[key] = value
        return config

    # ------------------------------------------------------------------
    # Windowing & layout helpers
    # ------------------------------------------------------------------
    def _parse_window_frame(self, raw: str) -> WindowFrame:
        """Parse textual window specification into a WindowFrame."""
        text = raw.strip()
        lower = text.lower()
        mode: TypingLiteral["rolling", "expanding", "cumulative"] = "rolling"
        interval_value: Optional[int] = None
        interval_unit: Optional[str] = None

        match_last = re.match(r'last\s+(\d+)\s+([\w]+)', lower)
        if match_last:
            interval_value = int(match_last.group(1))
            interval_unit = match_last.group(2)
        elif lower.startswith('over all') or lower in {'all', 'overall', 'cumulative'}:
            mode = "cumulative"
        elif lower.startswith('expanding'):
            mode = "expanding"

        return WindowFrame(mode=mode, interval_value=interval_value, interval_unit=interval_unit)

    def _build_cache_policy(self, data: Dict[str, Any]) -> CachePolicy:
        if not data:
            return CachePolicy(strategy="none")
        strategy = str(data.get('strategy', 'memory') or 'memory').lower()
        ttl_raw = data.get('ttl_seconds') or data.get('ttl') or data.get('ttl_s')
        ttl_seconds: Optional[int] = None
        if ttl_raw is not None:
            if isinstance(ttl_raw, (int, float)):
                ttl_seconds = int(ttl_raw)
            else:
                ttl_clean = str(ttl_raw).strip()
                match_val = re.match(r'(\d+)', ttl_clean)
                if match_val:
                    ttl_seconds = int(match_val.group(1))
        max_entries = data.get('max_entries') or data.get('max rows') or data.get('max')
        if max_entries is not None and not isinstance(max_entries, int):
            try:
                max_entries = int(str(max_entries))
            except ValueError:
                max_entries = None
        return CachePolicy(strategy=strategy, ttl_seconds=ttl_seconds, max_entries=max_entries)

    def _build_pagination_policy(self, data: Dict[str, Any]) -> PaginationPolicy:
        if not data:
            return PaginationPolicy(enabled=False)
        enabled = self._parse_bool(str(data.get('enabled', 'true'))) if 'enabled' in data else True
        page_size = data.get('page_size') or data.get('page size') or data.get('size')
        if page_size is not None and not isinstance(page_size, int):
            try:
                page_size = int(str(page_size))
            except ValueError:
                page_size = None
        max_pages = data.get('max_pages') or data.get('max pages')
        if max_pages is not None and not isinstance(max_pages, int):
            try:
                max_pages = int(str(max_pages))
            except ValueError:
                max_pages = None
        return PaginationPolicy(enabled=enabled, page_size=page_size, max_pages=max_pages)

    def _build_streaming_policy(self, data: Dict[str, Any]) -> StreamingPolicy:
        if not data:
            return StreamingPolicy(enabled=True)
        enabled = self._parse_bool(str(data.get('enabled', 'true'))) if 'enabled' in data else True
        chunk_size = data.get('chunk_size') or data.get('chunk size') or data.get('batch')
        if chunk_size is not None and not isinstance(chunk_size, int):
            try:
                chunk_size = int(str(chunk_size))
            except ValueError:
                chunk_size = None
        return StreamingPolicy(enabled=enabled, chunk_size=chunk_size)

    def _build_layout_spec(self, data: Dict[str, Any]) -> LayoutSpec:
        layout = LayoutSpec()

        def _to_int(val: Any) -> Optional[int]:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            try:
                if isinstance(val, str) and not val.strip():
                    return None
                return int(val)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                try:
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    return None

        for key, value in data.items():
            lower = key.replace(' ', '_').lower()
            if lower == 'width':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.width = maybe
            elif lower == 'height':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.height = maybe
            elif lower == 'variant':
                layout.variant = str(value)
            elif lower == 'order':
                maybe = _to_int(value)
                if maybe is not None:
                    layout.order = maybe
            elif lower == 'area':
                layout.area = str(value)
            elif lower == 'breakpoint':
                layout.breakpoint = str(value)
            else:
                layout.props[key] = value
        return layout

    def _build_layout_meta(self, data: Dict[str, Any]) -> LayoutMeta:
        meta = LayoutMeta()

        def _to_int(val: Any) -> Optional[int]:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            try:
                if isinstance(val, str) and not val.strip():
                    return None
                return int(val)
            except (ValueError, TypeError):
                try:
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    return None

        extras: Dict[str, Any] = {}
        for key, value in data.items():
            lower = key.replace(' ', '_').lower()
            if lower == 'width':
                maybe = _to_int(value)
                if maybe is not None:
                    meta.width = maybe
            elif lower == 'height':
                maybe = _to_int(value)
                if maybe is not None:
                    meta.height = maybe
            elif lower == 'variant':
                meta.variant = str(value)
            elif lower == 'align':
                meta.align = str(value)
            elif lower == 'emphasis':
                meta.emphasis = str(value)
            else:
                extras[key] = value
        meta.extras = extras
        return meta

    # Placeholder methods expected in subclasses/mixins
    def _parse_expression(self, text: str) -> Expression:  # pragma: no cover - overridden
        raise NotImplementedError
