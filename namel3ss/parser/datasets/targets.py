"""Dataset target parsing for ML."""

from __future__ import annotations

import shlex
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import DatasetTarget


class TargetParserMixin:
    """Mixin for parsing dataset target specifications."""
    
    def _parse_dataset_target(self, header_line: str, header_indent: int) -> "DatasetTarget":
        """
        Parse dataset target specification for ML.
        
        Targets define prediction objectives for ML models including
        classification, regression, and ranking tasks.
        """
        from ...ast import DatasetTarget
        
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse target declaration: {exc}",
                self.pos,
                header_line,
                hint='Check for unmatched quotes in target name'
            )
        if len(tokens) < 2 or tokens[0].lower() != 'target':
            raise self._error(
                "Expected: target \"Name\"",
                self.pos,
                header_line,
                hint='Targets require a name, e.g., target "conversion":'
            )

        name = self._strip_quotes(tokens[1])
        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        kind_raw = config.pop('kind', 'classification')
        kind = str(kind_raw) if kind_raw is not None else 'classification'
        expr_raw = config.pop('expression', config.pop('expr', None))
        expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
        positive_raw = config.pop('positive_class', config.pop('positive', None))
        positive_class = str(positive_raw) if positive_raw is not None else None
        horizon_raw = config.pop('horizon', config.pop('window', None))
        horizon = self._coerce_int(horizon_raw)
        options = self._coerce_options_dict(config.pop('options', {}))
        if config:
            options.update(config)

        return DatasetTarget(
            name=name,
            kind=kind,
            expression=expression,
            positive_class=positive_class,
            horizon=horizon,
            options=options,
        )
