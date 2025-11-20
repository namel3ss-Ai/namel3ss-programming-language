from __future__ import annotations

import re

from typing import Optional

from namel3ss.ast import Page, PageStatement, RefreshPolicy
from namel3ss.lang import (
    PAGE_STATEMENT_KEYWORDS,
    suggest_keyword,
    valid_keywords_for_context,
)

from .components import ComponentParserMixin
from .control_flow import ControlFlowParserMixin


class PageParserMixin(ComponentParserMixin, ControlFlowParserMixin):
    """Parsing logic for page declarations and statements."""

    def _parse_page(self, line: str, line_no: int, base_indent: int) -> Page:
        """
        Parse a page declaration with route, reactive mode, and body statements.
        
        Syntax:
            page "Name" [at "route"] [reactive|static]:
                [reactive: true|false]
                [auto refresh every <n> seconds|minutes|ms]
                [layout:]
                    [key: value]
                <page statements>
        
        Args:
            line: The page declaration line
            line_no: Line number for error reporting
            base_indent: Base indentation level
            
        Returns:
            Page: Parsed page AST node
            
        Raises:
            N3SyntaxError: On syntax errors with helpful hints
        """
        stripped = line.strip()
        if not stripped.endswith(':'):
            raise self._error(
                'Page declaration must end with ":"',
                line_no,
                line,
                hint='Syntax: page "Name" [at "route"] [reactive|static]:'
            )
        stripped = stripped[:-1].rstrip()
        match = re.match(r'page\s+"([^"]+)"(.*)', stripped)
        if not match:
            raise self._error(
                'Invalid page declaration syntax',
                line_no,
                line,
                hint='Expected: page "Name" [at "route"] [reactive|static]:'
            )
        name = match.group(1)
        remainder = (match.group(2) or '').strip()
        route: Optional[str] = None
        reactive_flag = False

        # Parse optional modifiers (at "route", reactive, static, kind reactive/static)
        while remainder:
            lowered = remainder.lower()
            if lowered.startswith('at '):
                route_match = re.match(r'at\s+"([^"]+)"(.*)', remainder, flags=re.IGNORECASE)
                if not route_match:
                    raise self._error(
                        'Invalid route specification',
                        line_no,
                        line,
                        hint='Expected: at "route"'
                    )
                route = route_match.group(1)
                remainder = (route_match.group(2) or '').strip()
                continue
            if lowered.startswith('kind '):
                remainder = remainder[5:].strip()
                lowered = remainder.lower()
                if lowered.startswith('reactive'):
                    remainder = remainder[len('reactive') :].strip()
                    reactive_flag = True
                    continue
                if lowered.startswith('static'):
                    remainder = remainder[len('static') :].strip()
                    reactive_flag = False
                    continue
                raise self._error(
                    "Unknown page kind",
                    line_no,
                    line,
                    hint='Valid page kinds: reactive, static'
                )
            if lowered.startswith('reactive'):
                remainder = remainder[len('reactive') :].strip()
                reactive_flag = True
                continue
            if lowered.startswith('static'):
                remainder = remainder[len('static') :].strip()
                reactive_flag = False
                continue
            raise self._error(
                'Unexpected page modifier',
                line_no,
                line,
                hint='Valid modifiers: at "route", reactive, static, kind reactive/static'
            )

        if remainder:
            raise self._error(
                'Unexpected trailing content in page declaration',
                line_no,
                line,
                hint='Remove extra text after page declaration'
            )

        if route is None:
            route = self._default_page_route(name)

        page = Page(name=name, route=route, reactive=reactive_flag)
        
        # Parse page body with centralized indentation validation
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            # Use centralized indentation validation
            nxt_line_no = self.pos + 1
            try:
                indent_info = self._expect_indent_greater_than(
                    nxt,
                    base_indent,
                    nxt_line_no,
                    f'page "{name}" body'
                )
            except Exception:
                # If indentation is not greater, we've reached the end of the page body
                break
            lowered = stripped.lower()
            
            # Parse page-level configuration
            if lowered.startswith('reactive:'):
                bool_str = stripped.split(':', 1)[1].strip()
                page.reactive = self._coerce_bool_with_context(
                    bool_str,
                    'reactive',
                    nxt_line_no,
                    nxt
                )
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
                        "Invalid auto refresh syntax",
                        nxt_line_no,
                        nxt,
                        hint='Expected: auto refresh every <number> [seconds|minutes|ms]'
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
                block_indent = indent_info.effective_level
                self._advance()
                config = self._parse_kv_block(block_indent)
                page.layout.update(config)
                continue
            
            # Parse page statement
            stmt = self._parse_page_statement(indent_info.effective_level)
            page.statements.append(stmt)
        return page

    def _default_page_route(self, name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip()).strip('-').lower()
        if not slug:
            slug = "page"
        return f"/{slug}"

    def _parse_page_statement(self, parent_indent: int) -> PageStatement:
        """
        Parse a page statement with keyword validation and helpful error messages.
        
        Valid page statements:
            - set <variable> = <expression>
            - if/elif/else (control flow)
            - for/while (loops)
            - break/continue (loop control)
            - show text/table/chart/form (components)
            - predict (ML prediction)
            - action (action handler)
        
        Args:
            parent_indent: Parent indentation level
            
        Returns:
            PageStatement: Parsed statement AST node
            
        Raises:
            N3SyntaxError: On syntax errors with keyword suggestions
        """
        line = self._advance()
        line_no = self.pos
        if line is None:
            raise self._error(
                "Unexpected end of input inside page body",
                line_no,
                '',
                hint='Expected page statement (set, show, if, for, etc.)'
            )
        stripped = line.strip()
        
        # Variable assignment
        if stripped.startswith('set '):
            return self._parse_variable_assignment(line, line_no, parent_indent)
        
        # Control flow
        if stripped.startswith('if '):
            return self._parse_if_block(line, line_no, parent_indent)
        if stripped.startswith('for '):
            return self._parse_for_loop(line, line_no, parent_indent)
        if stripped.startswith('while '):
            return self._parse_while_loop(line, line_no, parent_indent)
            
        # Loop control
        if stripped.startswith('break'):
            if stripped != 'break':
                raise self._error(
                    "'break' statement cannot have trailing content",
                    line_no,
                    line,
                    hint="Use just 'break' by itself"
                )
            return self._parse_loop_control('break', line_no, line)
        if stripped.startswith('continue'):
            if stripped != 'continue':
                raise self._error(
                    "'continue' statement cannot have trailing content",
                    line_no,
                    line,
                    hint="Use just 'continue' by itself"
                )
            return self._parse_loop_control('continue', line_no, line)
            
        # Misplaced elif/else
        if stripped.startswith('elif ') or stripped.startswith('else:'):
            raise self._error(
                "'elif' and 'else' must follow an if block",
                line_no,
                line,
                hint='Check your if/elif/else structure'
            )
        
        # Component display
        if stripped.startswith('show text '):
            return self._parse_show_text(line, parent_indent)
        if stripped.startswith('show table '):
            return self._parse_show_table(line, parent_indent)
        if stripped.startswith('show chart '):
            return self._parse_show_chart(line, parent_indent)
        if stripped.startswith('show form '):
            return self._parse_show_form(line, parent_indent)
            
        # ML prediction
        if stripped.startswith('predict '):
            return self._parse_predict_statement(line, parent_indent)
            
        # Action handler
        if stripped.startswith('action '):
            return self._parse_action(line, parent_indent)
        
        # Unknown statement - provide helpful suggestion
        first_word = stripped.split()[0] if stripped.split() else stripped
        suggestion = suggest_keyword(first_word, 'page-statement')
        
        valid_keywords = ', '.join(sorted([
            'set', 'show', 'if', 'for', 'while', 'break', 'continue', 'predict', 'action'
        ]))
        
        error_msg = f"Unknown page statement: '{first_word}'"
        if suggestion and suggestion != first_word:
            hint = f"Did you mean '{suggestion}'? Valid keywords: {valid_keywords}"
        else:
            hint = f"Valid page statement keywords: {valid_keywords}"
        
        raise self._error(error_msg, line_no, line, hint=hint)

