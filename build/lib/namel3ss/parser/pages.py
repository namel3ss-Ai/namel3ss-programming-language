from __future__ import annotations

import re

from namel3ss.ast import Page, PageStatement, RefreshPolicy

from .components import ComponentParserMixin
from .control_flow import ControlFlowParserMixin


class PageParserMixin(ComponentParserMixin, ControlFlowParserMixin):
    """Parsing logic for page declarations and statements."""

    def _parse_page(self, line: str, line_no: int, base_indent: int) -> Page:
        match = re.match(r'page\s+"([^"]+)"\s+at\s+"([^"]+)"\s*:?', line.strip())
        if not match:
            raise self._error('Expected: page "Name" at "/route":', line_no, line)
        name = match.group(1)
        route = match.group(2)
        page = Page(name=name, route=route)
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
            lowered = stripped.lower()
            if lowered.startswith('reactive:'):
                page.reactive = self._parse_bool(stripped.split(':', 1)[1])
                self._advance()
                continue
            if lowered.startswith('auto refresh'):
                refresh_text = stripped.split('auto refresh', 1)[1].strip()
                match_refresh = re.match(
                    r'(?:every\s+)?(\d+)\s*(seconds|second|minutes|minute|ms|milliseconds)?',
                    refresh_text,
                    re.IGNORECASE,
                )
                if not match_refresh:
                    raise self._error(
                        "Expected: auto refresh every <number> [seconds|minutes|ms]",
                        self.pos + 1,
                        nxt,
                    )
                value = int(match_refresh.group(1))
                unit = (match_refresh.group(2) or 'seconds').lower()
                interval_seconds = value
                if unit.startswith('minute'):
                    interval_seconds = value * 60
                elif unit in {'ms', 'millisecond', 'milliseconds'}:
                    interval_seconds = max(1, value // 1000)
                page.refresh_policy = RefreshPolicy(interval_seconds=interval_seconds, mode='polling')
                self._advance()
                continue
            if lowered.startswith('layout:'):
                block_indent = indent
                self._advance()
                config = self._parse_kv_block(block_indent)
                page.layout.update(config)
                continue
            stmt = self._parse_page_statement(indent)
            page.statements.append(stmt)
        return page

    def _parse_page_statement(self, parent_indent: int) -> PageStatement:
        line = self._advance()
        line_no = self.pos
        if line is None:
            raise self._error("Unexpected end of input inside page", line_no, '')
        stripped = line.strip()
        if stripped.startswith('set '):
            return self._parse_variable_assignment(line, line_no, parent_indent)
        if stripped.startswith('if '):
            return self._parse_if_block(line, line_no, parent_indent)
        if stripped.startswith('for '):
            return self._parse_for_loop(line, line_no, parent_indent)
        if stripped.startswith('while '):
            return self._parse_while_loop(line, line_no, parent_indent)
        if stripped.startswith('break'):
            if stripped != 'break':
                raise self._error("Expected 'break' with no trailing content", line_no, line)
            return self._parse_loop_control('break', line_no, line)
        if stripped.startswith('continue'):
            if stripped != 'continue':
                raise self._error("Expected 'continue' with no trailing content", line_no, line)
            return self._parse_loop_control('continue', line_no, line)
        if stripped.startswith('elif ') or stripped.startswith('else:'):
            raise self._error("'elif' and 'else' must follow an if block", line_no, line)
        if stripped.startswith('show text '):
            return self._parse_show_text(line, parent_indent)
        if stripped.startswith('show table '):
            return self._parse_show_table(line, parent_indent)
        if stripped.startswith('show chart '):
            return self._parse_show_chart(line, parent_indent)
        if stripped.startswith('show form '):
            return self._parse_show_form(line, parent_indent)
        if stripped.startswith('predict '):
            return self._parse_predict_statement(line, parent_indent)
        if stripped.startswith('action '):
            return self._parse_action(line, parent_indent)
        raise self._error(
            "Expected 'set', 'if', 'for', 'show text', 'show table', 'show chart', 'show form', or 'action' inside page",
            line_no,
            line,
        )

