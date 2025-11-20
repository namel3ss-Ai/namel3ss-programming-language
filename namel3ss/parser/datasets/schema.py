"""Dataset schema parsing."""

from __future__ import annotations

import shlex
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import DatasetSchemaField


class SchemaParserMixin:
    """Mixin for parsing dataset schema blocks."""
    
    def _parse_dataset_schema_block(self, parent_indent: int) -> List["DatasetSchemaField"]:
        """
        Parse dataset schema block with column definitions.
        
        Schema defines column names, types, nullability, descriptions,
        constraints, and statistics.
        """
        from ...ast import DatasetSchemaField
        
        fields: List[DatasetSchemaField] = []
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

            name_part, _, remainder = stripped.partition(':')
            name_token = name_part.strip()
            if not name_token:
                raise self._error(
                    "Column name required inside schema block",
                    self.pos + 1,
                    line,
                    hint='Each schema field must have a name, e.g., column_name: type'
                )
            if name_token.lower().startswith('column '):
                name_token = name_token[7:]
            name = self._strip_quotes(name_token.strip())

            self._advance()

            next_line = self._peek()
            has_block = False
            if next_line is not None and self._indent(next_line) > indent:
                has_block = True

            dtype = remainder.strip() or None
            config: Dict[str, Any] = {}
            if has_block:
                config = self._parse_kv_block(indent)
            if 'dtype' in config:
                dtype_value = config.pop('dtype')
                dtype = str(dtype_value) if dtype_value is not None else dtype
            if dtype is None:
                dtype = 'any'

            nullable = self._to_bool(config.pop('nullable', True))
            description_raw = config.pop('description', config.pop('desc', None))
            description = str(description_raw) if description_raw is not None else None
            tags_raw = config.pop('tags', [])
            tags = self._ensure_string_list(tags_raw)
            constraints_raw = config.pop('constraints', {})
            constraints = self._coerce_options_dict(constraints_raw)
            stats_raw = config.pop('stats', {})
            stats = self._coerce_options_dict(stats_raw)
            if config:
                constraints.update(config)

            fields.append(
                DatasetSchemaField(
                    name=name,
                    dtype=str(dtype),
                    nullable=nullable,
                    description=description,
                    tags=tags,
                    constraints=constraints,
                    stats=stats,
                )
            )
        return fields
