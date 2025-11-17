from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional

from namel3ss.ast import (
    ActionOperationType,
    ContextValue,
    FormField,
    InferenceTarget,
    LayoutMeta,
    LayoutSpec,
    PredictStatement,
    ShowChart,
    ShowForm,
    ShowTable,
    ShowText,
    VariableAssignment,
)

from .actions import ActionParserMixin
from .base import N3SyntaxError


class ComponentParserMixin(ActionParserMixin):
    """Parsing helpers for component-level statements within pages."""

    def _parse_show_text(self, line: str, base_indent: int) -> ShowText:
        match = re.match(r'show\s+text\s+"([^"]+)"\s*$', line.strip())
        if not match:
            raise self._error("Expected: show text \"Message\"", self.pos, line)
        text = match.group(1)
        styles: Dict[str, str] = {}
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            match_style = re.match(r'([\w\s]+):\s*(.+)', stripped)
            if not match_style:
                break
            key = match_style.group(1).strip()
            value = match_style.group(2).strip()
            styles[key] = value
            self._advance()
        return ShowText(text=text, styles=styles)

    def _parse_show_table(self, line: str, base_indent: int) -> ShowTable:
        match = re.match(
            r'show\s+table\s+"([^\"]+)"\s+from\s+(table|dataset|frame)\s+([^\s]+)\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show table \"Title\" from table|dataset|frame SOURCE",
                self.pos,
                line,
            )
        title = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        columns: Optional[List[str]] = None
        filter_by: Optional[str] = None
        sort_by: Optional[str] = None
        style_values: Dict[str, Any] = {}
        style_block: Optional[Dict[str, Any]] = None
        layout_meta: Optional[LayoutMeta] = None
        insight_name: Optional[str] = None
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if stripped.startswith('columns:'):
                cols = stripped[len('columns:'):].strip()
                columns = [c.strip() for c in cols.split(',') if c.strip()]
                self._advance()
            elif stripped.startswith('filter by:'):
                filter_by = stripped[len('filter by:'):].strip()
                self._advance()
            elif stripped.startswith('sort by:'):
                sort_by = stripped[len('sort by:'):].strip()
                self._advance()
            elif stripped.startswith('style:'):
                block_indent = indent
                self._advance()
                style_block = self._parse_kv_block(block_indent)
            elif stripped.startswith('layout:'):
                block_indent = indent
                self._advance()
                block = self._parse_kv_block(block_indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('insight:'):
                insight_raw = stripped[len('insight:'):].strip()
                value = self._coerce_scalar(insight_raw) if insight_raw else None
                if isinstance(value, ContextValue):
                    insight_name = f"{value.scope}:{'.'.join(value.path)}"
                elif value is not None:
                    insight_name = str(value)
                else:
                    insight_name = None
                self._advance()
            else:
                match_style = re.match(r'([\w\s]+):\s*(.+)', stripped)
                if not match_style:
                    raise self._error(
                        "Expected table property ('columns:', 'filter by:', 'sort by:' or style)",
                        self.pos + 1,
                        nxt,
                    )
                key = match_style.group(1).strip()
                value = self._coerce_scalar(match_style.group(2).strip())
                style_values[key] = value
                self._advance()
        combined_style: Dict[str, Any] = {}
        if style_block:
            combined_style.update(style_block)
        if style_values:
            combined_style.update(style_values)
        style_payload = combined_style or None
        return ShowTable(
            title=title,
            source_type=source_type,
            source=source,
            columns=columns,
            filter_by=filter_by,
            sort_by=sort_by,
            style=style_payload,
            layout=layout_meta,
            insight=insight_name,
        )

    def _parse_show_chart(self, line: str, base_indent: int) -> ShowChart:
        match = re.match(
            r'show\s+chart\s+"([^\"]+)"\s+from\s+(table|dataset|frame|file)\s+([^\s]+)\s*$',
            line.strip(),
        )
        if not match:
            raise self._error(
                "Expected: show chart \"Title\" from table|dataset|frame|file SOURCE",
                self.pos,
                line,
            )
        heading = match.group(1)
        source_type = match.group(2)
        source = match.group(3)
        chart_type = 'bar'
        x = None
        y = None
        color = None
        style_inline: Dict[str, Any] = {}
        style_blocks: Dict[str, Any] = {}
        general_style: Dict[str, Any] = {}
        layout_meta: Optional[LayoutMeta] = None
        chart_title_value: Optional[Any] = None
        legend_config: Optional[Dict[str, Any]] = None
        insight_name: Optional[str] = None
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if stripped.startswith('type:'):
                chart_type = stripped[len('type:'):].strip()
                self._advance()
            elif stripped.startswith('x:'):
                x = stripped[len('x:'):].strip()
                self._advance()
            elif stripped.startswith('y:'):
                y = stripped[len('y:'):].strip()
                self._advance()
            elif stripped.startswith('color:'):
                color = stripped[len('color:'):].strip()
                self._advance()
            elif stripped.startswith('title:'):
                remainder = stripped[len('title:'):].strip()
                self._advance()
                title_meta: Dict[str, Any] = {}
                if remainder:
                    value = self._coerce_scalar(remainder)
                    if isinstance(value, dict):
                        title_meta.update(value)
                        text_value = title_meta.get('text') or title_meta.get('value')
                        if text_value is not None:
                            chart_title_value = text_value
                    else:
                        if isinstance(value, bool):
                            title_meta['show'] = value
                        else:
                            chart_title_value = value
                else:
                    title_meta = self._parse_kv_block(indent)
                    text_value = title_meta.get('text') or title_meta.get('value')
                    if text_value is not None:
                        chart_title_value = text_value
                if title_meta:
                    style_blocks.setdefault('title', {}).update(title_meta)
                elif chart_title_value is not None:
                    style_blocks.setdefault('title', {})['text'] = chart_title_value
            elif stripped.startswith('layout:'):
                block_indent = indent
                self._advance()
                block = self._parse_kv_block(block_indent)
                layout_meta = self._build_layout_meta(block)
            elif stripped.startswith('legend:'):
                remainder = stripped[len('legend:'):].strip()
                self._advance()
                if remainder:
                    value = self._coerce_scalar(remainder)
                    if isinstance(value, dict):
                        legend_config = value
                    else:
                        if isinstance(value, bool):
                            legend_config = {'show': value}
                        else:
                            legend_config = {'position': value}
                else:
                    legend_config = self._parse_kv_block(indent)
            elif stripped.startswith('colors:'):
                block_indent = indent
                self._advance()
                style_blocks['colors'] = self._parse_kv_block(block_indent)
            elif stripped.startswith('axes:'):
                block_indent = indent
                self._advance()
                style_blocks['axes'] = self._parse_kv_block(block_indent)
            elif stripped.startswith('style:'):
                block_indent = indent
                self._advance()
                general_style.update(self._parse_kv_block(block_indent))
            elif stripped.startswith('insight:'):
                insight_raw = stripped[len('insight:'):].strip()
                value = self._coerce_scalar(insight_raw) if insight_raw else None
                if isinstance(value, ContextValue):
                    insight_name = f"{value.scope}:{'.'.join(value.path)}"
                elif value is not None:
                    insight_name = str(value)
                else:
                    insight_name = None
                self._advance()
            else:
                match_style = re.match(r'([\w\s]+):\s*(.+)', stripped)
                if not match_style:
                    raise self._error(
                        "Expected chart property ('type:', 'x:', 'y:', 'color:', 'title:', 'legend:', 'colors:', 'axes:', 'style:' or layout)",
                        self.pos + 1,
                        nxt,
                    )
                key = match_style.group(1).strip()
                value = self._coerce_scalar(match_style.group(2).strip())
                style_inline[key] = value
                self._advance()
        combined_style: Dict[str, Any] = {}
        if general_style:
            combined_style.update(general_style)
        if style_inline:
            combined_style.update(style_inline)
        if style_blocks:
            for key, value in style_blocks.items():
                combined_style[key] = value
        style_payload = combined_style or None
        return ShowChart(
            heading=heading,
            source_type=source_type,
            source=source,
            chart_type=chart_type,
            x=x,
            y=y,
            color=color,
            layout=layout_meta,
            insight=insight_name,
            style=style_payload,
            title=chart_title_value,
            legend=legend_config,
        )

    def _parse_show_form(self, line: str, base_indent: int) -> ShowForm:
        match = re.match(r'show\s+form\s+"([^\"]+)"\s*:?', line.strip())
        if not match:
            raise self._error("Expected: show form \"Title\":", self.pos, line)
        title = match.group(1)
        fields: List[FormField] = []
        on_submit_ops: List[ActionOperationType] = []
        styles: Dict[str, str] = {}
        layout_spec = LayoutSpec()
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if stripped.startswith('fields:'):
                rest = stripped[len('fields:'):].strip()
                field_parts = [p.strip() for p in rest.split(',') if p.strip()]
                for fp in field_parts:
                    if ':' in fp:
                        fname, ftype = [part.strip() for part in fp.split(':', 1)]
                        fields.append(FormField(name=fname, field_type=ftype))
                    else:
                        fields.append(FormField(name=fp))
                self._advance()
            elif stripped.startswith('on submit:'):
                op_base_indent = self._indent(nxt)
                self._advance()
                while self.pos < len(self.lines):
                    sub = self._peek()
                    if sub is None:
                        break
                    sub_indent = self._indent(sub)
                    sub_stripped = sub.strip()
                    if not sub_stripped or sub_stripped.startswith('#'):
                        self._advance()
                        continue
                    if sub_indent <= op_base_indent:
                        break
                    op = self._parse_action_operation(sub_stripped)
                    on_submit_ops.append(op)
            elif stripped.startswith('layout:'):
                block_indent = indent
                self._advance()
                block = self._parse_kv_block(block_indent)
                layout_spec = self._build_layout_spec(block)
            else:
                match_style = re.match(r'([\w\s]+):\s*(.+)', stripped)
                if not match_style:
                    raise self._error(
                        "Expected form property ('fields:', 'on submit:' or style)",
                        self.pos + 1,
                        nxt,
                    )
                key = match_style.group(1).strip()
                value = match_style.group(2).strip()
                styles[key] = value
                self._advance()
        return ShowForm(
            title=title,
            fields=fields,
            on_submit_ops=on_submit_ops,
            styles=styles,
            layout=layout_spec,
        )

    def _parse_predict_statement(self, line: str, base_indent: int) -> PredictStatement:
        stripped = line.strip()
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            raise self._error(f"Unable to parse predict statement: {exc}", self.pos, line)
        if not tokens or tokens[0] != 'predict':
            raise self._error("Predict statements must start with 'predict'", self.pos, line)

        idx = 1
        if idx < len(tokens) and tokens[idx].lower() == 'using':
            idx += 1
        if idx >= len(tokens) or tokens[idx].lower() != 'model':
            raise self._error("Predict statements require 'using model'", self.pos, line)
        idx += 1
        if idx >= len(tokens):
            raise self._error("Model name is required for predict statements", self.pos, line)
        model_name = tokens[idx]
        idx += 1

        input_kind = 'dataset'
        input_ref: Optional[str] = None
        if idx < len(tokens) and tokens[idx].lower() == 'with':
            idx += 1
            if idx >= len(tokens):
                raise self._error("Expected input kind after 'with'", self.pos, line)
            possible_kind = tokens[idx].lower()
            if possible_kind in {'dataset', 'table', 'payload', 'variables'}:
                input_kind = possible_kind
                idx += 1
                if idx >= len(tokens):
                    raise self._error("Expected input reference after input kind", self.pos, line)
                input_ref = tokens[idx]
                idx += 1
            else:
                input_kind = 'dataset'
                input_ref = tokens[idx]
                idx += 1

        assign = InferenceTarget()
        if idx < len(tokens) and tokens[idx].lower() == 'into':
            idx += 1
            if idx >= len(tokens):
                raise self._error("Expected target after 'into'", self.pos, line)
            possible_target = tokens[idx].lower()
            if possible_target in {'variable', 'dataset', 'insight', 'component'}:
                target_kind = possible_target
                idx += 1
                if idx >= len(tokens):
                    raise self._error("Expected name after target kind", self.pos, line)
                target_name = tokens[idx]
                idx += 1
            else:
                target_kind = 'variable'
                target_name = tokens[idx]
                idx += 1
            assign = InferenceTarget(kind=target_kind, name=target_name)

        parameters: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_next = nxt.strip()
            if not stripped_next or stripped_next.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            lowered = stripped_next.lower()
            if lowered.startswith('params:') or lowered.startswith('parameters:') or lowered.startswith('options:'):
                block_indent = indent
                self._advance()
                block_data = self._parse_kv_block(block_indent)
                parameters.update(block_data)
            else:
                break

        return PredictStatement(
            model_name=model_name,
            input_kind=input_kind,
            input_ref=input_ref,
            assign=assign,
            parameters=parameters,
        )

    def _parse_variable_assignment(self, line: str, line_no: int, base_indent: int) -> VariableAssignment:
        stripped = line.strip()
        if not stripped.startswith('set '):
            raise self._error("Expected 'set' at start of variable assignment", line_no, line)
        assignment = stripped[4:].strip()
        if '=' not in assignment:
            raise self._error(
                "Expected '=' in variable assignment: set name = expression",
                line_no,
                line,
            )
        parts = assignment.split('=', 1)
        if len(parts) != 2:
            raise self._error("Invalid variable assignment syntax", line_no, line)
        var_name = parts[0].strip()
        expr_text = parts[1].strip()
        if not var_name:
            raise self._error("Variable name cannot be empty", line_no, line)
        if not (var_name[0].isalpha() or var_name[0] == '_'):
            raise self._error(
                f"Variable name must start with letter or underscore: '{var_name}'",
                line_no,
                line,
            )
        for ch in var_name:
            if not (ch.isalnum() or ch == '_'):
                raise self._error(
                    f"Variable name can only contain letters, numbers, and underscores: '{var_name}'",
                    line_no,
                    line,
                )
        if not expr_text:
            raise self._error("Expression cannot be empty in variable assignment", line_no, line)
        try:
            expression = self._parse_expression(expr_text)
        except N3SyntaxError:
            raise
        except Exception as exc:
            raise self._error(f"Failed to parse expression: {exc}", line_no, line)
        return VariableAssignment(name=var_name, value=expression)
