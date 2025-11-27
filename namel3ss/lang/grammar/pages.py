"""Page and page statement parsing."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .helpers import _Line

from namel3ss.ast import (
    LogLevel,
    LogStatement, 
    Literal,
    Page,
    PageStatement,
    ShowChart,
    ShowTable,
    ShowText,
)
from namel3ss.ast.pages import ElifBlock, ForLoop, IfBlock


class PagesParserMixin:
    """Mixin providing page and page statement parsing methods."""

    def _parse_page(self, line: _Line) -> None:
        """Parse page declaration: page "name" at "/route":"""
        from .constants import PAGE_HEADER_RE
        
        match = PAGE_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "page declaration")
        page_name = match.group(1)
        route = match.group(2)
        base_indent = self._indent(line.text)
        self._advance()
        statements = self._parse_page_statements(base_indent)
        page = Page(name=page_name, route=route, statements=statements)
        self._ensure_app(line)
        if self._app:
            self._app.pages.append(page)

    def _parse_page_statements(self, parent_indent: int) -> List[PageStatement]:
        """Parse statements within a page block."""
        statements: List[PageStatement] = []
        while True:
            line = self._peek_line()
            if line is None:
                break
            stripped = line.text.strip()
            indent = self._indent(line.text)
            if self._should_skip_comment(stripped, line.number, line.text):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            if stripped.startswith('show text '):
                statements.append(self._parse_show_text(line))
                continue
            if stripped.startswith('show table '):
                statements.append(self._parse_show_table(line))
                continue
            if stripped.startswith('show chart '):
                statements.append(self._parse_show_chart(line))
                continue
            if stripped.startswith('if '):
                statements.append(self._parse_if_block(line, indent))
                continue
            if stripped.startswith('for '):
                statements.append(self._parse_for_loop(line, indent))
                continue
            if stripped.startswith('log '):
                statements.append(self._parse_log_statement(line))
                continue
            self._unsupported(line, "page statement")
        return statements

    def _parse_show_text(self, line: _Line) -> ShowText:
        """Parse show text statement."""
        match = re.match(r'^\s*show\s+text\s+"([^"]+)"\s*$', line.text)
        if not match:
            raise self._error('Expected: show text "message"', line)
        self._advance()
        return ShowText(text=match.group(1))

    def _parse_show_table(self, line: _Line) -> ShowTable:
        """Parse show table statement."""
        match = re.match(
            r'^\s*show\s+table\s+"([^"]+)"\s+from\s+(dataset|table|frame)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$',
            line.text,
        )
        if not match:
            self._unsupported(line, "show table statement")
        self._advance()
        return ShowTable(title=match.group(1), source_type=match.group(2), source=match.group(3))

    def _parse_show_chart(self, line: _Line) -> ShowChart:
        """Parse show chart statement."""
        match = re.match(
            r'^\s*show\s+chart\s+"([^"]+)"\s+from\s+(dataset|table)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$',
            line.text,
        )
        if not match:
            self._unsupported(line, "show chart statement")
        self._advance()
        return ShowChart(heading=match.group(1), source_type=match.group(2), source=match.group(3))

    def _parse_if_block(self, line: _Line, indent: int) -> IfBlock:
        """Parse if/elif/else control flow block."""
        condition_text = line.text.strip()
        if not condition_text.endswith(':'):
            raise self._error("if statement must end with ':'", line)
        condition_src = condition_text[len('if') : -1].strip()
        condition = self._expression_helper.parse(condition_src, line_no=line.number, line=line.text)
        self._advance()
        body = self._parse_page_statements(indent)
        elifs: List[ElifBlock] = []
        else_body: Optional[List[PageStatement]] = None
        while True:
            next_line = self._peek_line()
            if next_line is None:
                break
            stripped = next_line.text.strip()
            next_indent = self._indent(next_line.text)
            if not stripped:
                self._advance()
                continue
            if next_indent != indent:
                break
            if stripped.startswith('elif '):
                if not stripped.endswith(':'):
                    raise self._error("elif statement must end with ':'", next_line)
                expr_text = stripped[len('elif') : -1].strip()
                condition_expr = self._expression_helper.parse(expr_text, line_no=next_line.number, line=next_line.text)
                self._advance()
                elif_body = self._parse_page_statements(indent)
                elifs.append(ElifBlock(condition=condition_expr, body=elif_body))
                continue
            if stripped.startswith('else:'):
                self._advance()
                else_body = self._parse_page_statements(indent)
                break
            break
        return IfBlock(condition=condition, body=body, elifs=elifs, else_body=else_body)

    def _parse_for_loop(self, line: _Line, indent: int) -> ForLoop:
        """Parse for loop statement."""
        match = re.match(r'^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(dataset|table|frame)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$', line.text)
        if not match:
            raise self._error('Expected: for name in dataset foo:', line)
        loop_var = match.group(1)
        source_kind = match.group(2)
        source_name = match.group(3)
        self._advance()
        body = self._parse_page_statements(indent)
        return ForLoop(loop_var=loop_var, source_kind=source_kind, source_name=source_name, body=body)

    def _parse_kv_block(self, parent_indent: int) -> dict[str, str]:
        """Parse key-value block (used by theme and other declarations)."""
        from namel3ss.parser.base import ParserBase
        
        if self._in_ai_block:
            # Delegate to ParserBase implementation when AIParserMixin is driving parsing.
            return ParserBase._parse_kv_block(self, parent_indent)
        entries: dict[str, str] = {}
        while True:
            line = self._peek_line()
            if line is None:
                break
            stripped = line.text.strip()
            indent = self._indent(line.text)
            if self._should_skip_comment(stripped, line.number, line.text):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            match = re.match(r'^([A-Za-z0-9_\-\s]+):\s*(.+)$', stripped)
            if not match:
                raise self._error("Expected 'key: value' inside block", line)
            key = match.group(1).strip()
            value = match.group(2).strip()
            entries[key] = value
            self._advance()
        return entries

    def _parse_log_statement(self, line: _Line) -> LogStatement:
        """Parse log statement: log [level] "message" """
        from namel3ss.ast.source_location import SourceLocation
        
        stripped = line.text.strip()
        
        # Pattern for log statement with optional level
        # Matches: log "message" OR log info "message" OR log error "message"
        match = re.match(
            r'^log(?:\s+(debug|info|warn|error))?\s+"([^"]*)"$',
            stripped,
        )
        
        if not match:
            raise self._error(
                'Expected: log "message" or log level "message" where level is debug|info|warn|error',
                line
            )
        
        level_str = match.group(1)
        message_text = match.group(2)
        
        # Default to info level if not specified
        if level_str is None:
            level = LogLevel.INFO
        else:
            try:
                level = LogLevel(level_str)
            except ValueError:
                raise self._error(
                    f'Invalid log level "{level_str}". Valid levels: debug, info, warn, error',
                    line
                )
        
        # For now, treat message as a literal string
        # TODO: Support interpolated expressions like "Score: {{score}}"
        message = Literal(message_text)
        
        # Create source location for error reporting
        source_location = SourceLocation(
            file=getattr(self, '_source_path', '<unknown>'),
            line=line.number,
            column=0
        )
        
        self._advance()
        return LogStatement(
            level=level,
            message=message,
            source_location=source_location
        )


__all__ = ['PagesParserMixin']
