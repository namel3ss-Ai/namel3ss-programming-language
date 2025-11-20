"""Dataset connector configuration parsing."""

from __future__ import annotations

from typing import Any, Tuple


class ConnectorParserMixin:
    """Mixin for parsing dataset connector configurations."""
    
    def _parse_connector_option(self, option_text: str, line_no: int, line: str) -> Tuple[str, Any]:
        """
        Parse connector option: key = value.
        
        Returns tuple of (key, value).
        """
        if '=' not in option_text:
            raise self._error(
                "Expected: with option key = value",
                line_no,
                line,
                hint='Connector options require key = value format, e.g., with option timeout = 30'
            )
        key, value = option_text.split('=', 1)
        key = key.strip()
        value_str = value.strip()
        coerced_value = self._coerce_connector_option_value(value_str)
        return (key, coerced_value)

    def _apply_connector_option(self, dataset, key: str, value: Any) -> None:
        """
        Apply connector option to dataset configuration.
        
        Options can target connector config, cache policy,
        or general dataset metadata.
        """
        if not dataset.connector:
            dataset.connector.options[key] = value
            return

        lowered = key.lower()
        if lowered in {'timeout', 'retries', 'batch_size', 'max_rows'}:
            if dataset.connector:
                dataset.connector.options[key] = value
        elif lowered in {'ttl', 'max_age', 'cache_size'}:
            if not dataset.cache_policy:
                from ...ast import CachePolicy
                dataset.cache_policy = CachePolicy()
            if lowered == 'ttl' or lowered == 'max_age':
                dataset.cache_policy.ttl = int(value) if value is not None else None  # type: ignore[arg-type]
            elif lowered == 'cache_size':
                dataset.cache_policy.max_size = int(value) if value is not None else None  # type: ignore[arg-type]
        else:
            dataset.metadata[key] = value

    def _coerce_connector_option_value(self, value_str: str) -> Any:
        """
        Coerce connector option string value to appropriate type.
        
        Handles booleans, integers, floats, quoted strings, and context references.
        """
        import ast as python_ast
        
        value_str = value_str.strip()
        if not value_str:
            return ""
        
        # Try context reference first
        context_ref = self._parse_context_reference(value_str)
        if context_ref is not None:
            return context_ref
        
        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or (value_str.startswith("'") and value_str.endswith("'")):
            try:
                parsed = python_ast.literal_eval(value_str)
                if isinstance(parsed, str):
                    context_ref_inner = self._parse_context_reference(parsed)
                    if context_ref_inner is not None:
                        return context_ref_inner
                return parsed
            except (SyntaxError, ValueError):
                inner = value_str[1:-1]
                context_ref_inner = self._parse_context_reference(inner)
                if context_ref_inner is not None:
                    return context_ref_inner
                return inner
        
        # Handle boolean keywords
        value_lower = value_str.lower()
        if value_lower in {'true', 'yes', 'on'}:
            return True
        if value_lower in {'false', 'no', 'off'}:
            return False
        if value_lower in {'null', 'none', ''}:
            return None

        # Try numeric coercion
        try:
            return int(value_str)
        except ValueError:
            pass

        try:
            return float(value_str)
        except ValueError:
            pass

        return value_str
