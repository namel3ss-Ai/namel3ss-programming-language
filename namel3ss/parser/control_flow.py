from __future__ import annotations

import re
from typing import List, Optional

from namel3ss.ast import (
    BreakStatement,
    ContinueStatement,
    ElifBlock,
    ForLoop,
    IfBlock,
    PageStatement,
    WhileLoop,
)

from .datasets import DatasetParserMixin
from .base import N3SyntaxError


class ControlFlowParserMixin(DatasetParserMixin):
    """Parsing helpers for control-flow statements within pages."""

    def _parse_control_body(self, base_indent: int, header_line: str, header_no: int) -> List[PageStatement]:
        body: List[PageStatement] = []
        block_indent: Optional[int] = None
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
            if block_indent is None:
                block_indent = indent
            elif indent < block_indent:
                break
            stmt = self._parse_page_statement(indent)
            body.append(stmt)
        if not body:
            raise self._error("Expected at least one statement in block", header_no, header_line)
        return body

    def _parse_if_block(self, line: str, line_no: int, base_indent: int) -> IfBlock:
        stripped = line.strip()
        if not stripped.endswith(':'):
            raise self._error("Expected ':' after if condition", line_no, line)
        condition_text = stripped[3:-1].strip()
        if not condition_text:
            raise self._error("Expected condition after 'if'", line_no, line)
        try:
            condition = self._parse_expression(condition_text)
        except N3SyntaxError:
            raise
        except Exception as exc:
            raise self._error(f"Failed to parse if condition: {exc}", line_no, line)
        body = self._parse_control_body(base_indent, line, line_no)
        elif_blocks: List[ElifBlock] = []
        else_body: Optional[List[PageStatement]] = None
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_nxt = nxt.strip()
            if not stripped_nxt or stripped_nxt.startswith('#'):
                self._advance()
                continue
            if indent != base_indent:
                if indent < base_indent:
                    break
                # Indentation greater than base belongs to previous branch and
                # will be handled there.
                break
            if stripped_nxt.startswith('elif '):
                self._advance()
                if not stripped_nxt.endswith(':'):
                    raise self._error("Expected ':' after elif condition", self.pos, nxt)
                elif_condition_text = stripped_nxt[5:-1].strip()
                if not elif_condition_text:
                    raise self._error("Expected condition after 'elif'", self.pos, nxt)
                try:
                    elif_condition = self._parse_expression(elif_condition_text)
                except N3SyntaxError:
                    raise
                except Exception as exc:
                    raise self._error(f"Failed to parse elif condition: {exc}", self.pos, nxt)
                elif_body = self._parse_control_body(base_indent, nxt, self.pos)
                elif_blocks.append(ElifBlock(condition=elif_condition, body=elif_body))
                continue
            if stripped_nxt.startswith('else:'):
                self._advance()
                else_body = self._parse_control_body(base_indent, nxt, self.pos)
                break
            break
        return IfBlock(condition=condition, body=body, elifs=elif_blocks, else_body=else_body)

    def _parse_for_loop(self, line: str, line_no: int, base_indent: int) -> ForLoop:
        stripped = line.strip()
        if not stripped.endswith(':'):
            raise self._error("Expected ':' after for loop header", line_no, line)
        match = re.match(r'for\s+(\w+)\s+in\s+(dataset|table|frame)\s+(\w+)\s*:$', stripped)
        if not match:
            raise self._error("Expected: for <var> in dataset|table|frame <name>:", line_no, line)
        loop_var = match.group(1)
        source_kind = match.group(2)
        source_name = match.group(3)
        self._loop_depth += 1
        try:
            body = self._parse_control_body(base_indent, line, line_no)
        finally:
            self._loop_depth = max(0, self._loop_depth - 1)
        return ForLoop(loop_var=loop_var, source_kind=source_kind, source_name=source_name, body=body)

    def _parse_while_loop(self, line: str, line_no: int, base_indent: int) -> WhileLoop:
        stripped = line.strip()
        if not stripped.endswith(':'):
            raise self._error("Expected ':' after while condition", line_no, line)
        condition_text = stripped[5:-1].strip()
        if not condition_text:
            raise self._error("Expected condition after 'while'", line_no, line)
        try:
            condition = self._parse_expression(condition_text)
        except N3SyntaxError:
            raise
        except Exception as exc:
            raise self._error(f"Failed to parse while condition: {exc}", line_no, line)
        self._loop_depth += 1
        try:
            body = self._parse_control_body(base_indent, line, line_no)
        finally:
            self._loop_depth = max(0, self._loop_depth - 1)
        return WhileLoop(condition=condition, body=body)

    def _parse_loop_control(self, kind: str, line_no: int, line: str) -> PageStatement:
        if self._loop_depth <= 0:
            raise self._error(f"'{kind}' is only permitted inside a loop", line_no, line)
        if kind == 'break':
            return BreakStatement()
        if kind == 'continue':
            return ContinueStatement()
        raise self._error(f"Unsupported loop control '{kind}'", line_no, line)
