"""Dataset quality check parsing."""

from __future__ import annotations

import shlex
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import DatasetQualityCheck


class QualityParserMixin:
    """Mixin for parsing dataset quality check specifications."""
    
    def _parse_dataset_quality_check(self, header_line: str, header_indent: int) -> "DatasetQualityCheck":
        """
        Parse dataset quality check specification.
        
        Quality checks define validation rules with conditions, metrics,
        thresholds, severity levels, and alert messages.
        """
        from ...ast import DatasetQualityCheck
        
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse quality check declaration: {exc}",
                self.pos,
                header_line,
                hint='Check for unmatched quotes in quality check name'
            )
        if len(tokens) < 2 or tokens[0].lower() != 'quality':
            raise self._error(
                "Expected: quality \"Name\"",
                self.pos,
                header_line,
                hint='Quality checks require a name, e.g., quality "no_nulls":'
            )

        name = self._strip_quotes(tokens[1])
        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        condition_raw = config.pop('condition', config.pop('expr', None))
        condition = self._coerce_expression(condition_raw) if condition_raw is not None else None
        metric_raw = config.pop('metric', config.pop('measure', None))
        metric = str(metric_raw) if metric_raw is not None else None
        threshold_raw = config.pop('threshold', config.pop('value', None))
        threshold = None
        if threshold_raw is not None:
            try:
                threshold = float(threshold_raw)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                threshold = None
        severity_raw = config.pop('severity', config.pop('level', 'error'))
        severity = str(severity_raw) if severity_raw is not None else 'error'
        message_raw = config.pop('message', config.pop('text', None))
        message = str(message_raw) if message_raw is not None else None
        extras = self._coerce_options_dict(config.pop('extras', {}))
        if config:
            extras.update(config)

        return DatasetQualityCheck(
            name=name,
            condition=condition,
            metric=metric,
            threshold=threshold,
            severity=severity,
            message=message,
            extras=extras,
        )
