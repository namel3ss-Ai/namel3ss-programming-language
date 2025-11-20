"""Dataset transform block parsing."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ...ast import DatasetTransformStep


class TransformParserMixin:
    """Mixin for parsing dataset transform blocks."""
    
    def _parse_dataset_transform_block(self, parent_indent: int) -> List["DatasetTransformStep"]:
        """
        Parse dataset transform block with transformation steps.
        
        Transforms define data processing steps including filtering,
        aggregation, joins, derived columns, and custom operations.
        """
        from ...ast import DatasetTransformStep
        
        steps: List[DatasetTransformStep] = []
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

            parts = stripped.split(None, 1)
            if not parts:
                self._advance()
                continue

            step_type = parts[0].lower()
            step_desc = parts[1] if len(parts) > 1 else ''
            self._advance()

            next_line = self._peek()
            has_block = False
            if next_line is not None and self._indent(next_line) > indent:
                has_block = True

            config: Dict[str, Any] = {}
            if has_block:
                config = self._parse_kv_block(indent)

            expr_raw = config.pop('expression', config.pop('expr', step_desc if step_desc else None))
            expression = self._coerce_expression(expr_raw) if expr_raw is not None else None
            columns_raw = config.pop('columns', config.pop('fields', []))
            columns = self._ensure_string_list(columns_raw)
            join_on_raw = config.pop('join_on', config.pop('on', None))
            join_on = str(join_on_raw) if join_on_raw is not None else None
            options = self._coerce_options_dict(config.pop('options', {}))
            if config:
                options.update(config)

            steps.append(
                DatasetTransformStep(
                    step_type=step_type,
                    expression=expression,
                    columns=columns,
                    join_on=join_on,
                    options=options,
                )
            )
        return steps
