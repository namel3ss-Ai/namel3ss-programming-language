"""Utility functions for dataset parsing."""

from __future__ import annotations

import ast
import shlex
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import Dataset, DatasetConnectorConfig


class UtilityParserMixin:
    """Mixin providing utility functions for dataset parsing."""
    
    def _parse_tag_list(self, raw: str) -> List[str]:
        """Parse a comma or space-separated tag list."""
        if not raw:
            return []
        tokens: List[str] = []
        try:
            split_tokens = shlex.split(raw)
        except ValueError:
            split_tokens = [raw]
        for token in split_tokens:
            parts = token.split(',')
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    tokens.append(cleaned)
        return tokens

    def _strip_quotes(self, value: str) -> str:
        """Remove surrounding quotes from a string."""
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value

    def _ensure_string_list(self, raw: Any) -> List[str]:
        """Coerce various types into a list of strings."""
        if raw is None:
            return []
        if isinstance(raw, list):
            result: List[str] = []
            for item in raw:
                if item is None:
                    continue
                if isinstance(item, str):
                    result.append(item)
                else:
                    result.append(self._stringify_value(item))
            return [item for item in result if item]
        if isinstance(raw, str):
            if ',' in raw:
                tokens = [part.strip() for part in raw.split(',') if part.strip()]
            else:
                tokens = [part.strip() for part in raw.split() if part.strip()]
            return tokens
        return [self._stringify_value(raw)]

    def _coerce_options_dict(self, raw: Any) -> Dict[str, Any]:
        """Convert various types to options dictionary."""
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        return {"value": raw}

    def _to_bool(self, value: Any, default: bool = True) -> bool:
        """Convert various types to boolean."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'true', 'yes', '1', 'on'}:
                return True
            if lowered in {'false', 'no', '0', 'off'}:
                return False
        return bool(value)
