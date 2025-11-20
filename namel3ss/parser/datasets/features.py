"""Dataset feature parsing for ML."""

from __future__ import annotations

import shlex
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import DatasetFeature


class FeatureParserMixin:
    """Mixin for parsing dataset feature specifications."""
    
    def _parse_dataset_feature(self, header_line: str, header_indent: int) -> "DatasetFeature":
        """
        Parse dataset feature specification for ML.
        
        Features define ML input variables with roles, types, sources,
        transformations, and descriptions.
        """
        from ...ast import DatasetFeature
        
        stripped = header_line.strip()
        has_block = stripped.endswith(':')
        header = stripped[:-1] if has_block else stripped
        try:
            tokens = shlex.split(header)
        except ValueError as exc:
            raise self._error(
                f"Unable to parse feature declaration: {exc}",
                self.pos,
                header_line,
                hint='Check for unmatched quotes in feature name'
            )
        if len(tokens) < 2 or tokens[0].lower() != 'feature':
            raise self._error(
                "Expected: feature \"Name\"",
                self.pos,
                header_line,
                hint='Features require a name, e.g., feature "revenue_log":'
            )

        name = self._strip_quotes(tokens[1])
        config: Dict[str, Any] = {}
        if has_block:
            config = self._parse_kv_block(header_indent)

        role_raw = config.pop('role', 'feature')
        role = str(role_raw) if role_raw is not None else 'feature'
        source_raw = config.pop('source', config.pop('from', None))
        source = str(source_raw) if source_raw is not None else None
        dtype_raw = config.pop('dtype', None)
        dtype = str(dtype_raw) if dtype_raw is not None else None
        expr_raw = config.pop('expression', config.pop('expr', None))
        expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
        description_raw = config.pop('description', config.pop('desc', None))
        description = str(description_raw) if description_raw is not None else None
        options = self._coerce_options_dict(config.pop('options', {}))
        if config:
            options.update(config)

        return DatasetFeature(
            name=name,
            source=source,
            role=role,
            dtype=dtype,
            expression=expression,
            description=description,
            options=options,
        )
