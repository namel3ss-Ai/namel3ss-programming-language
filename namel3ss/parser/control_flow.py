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
        """
        Parse the body of a control flow block (if, for, while, etc.).
        
        Uses centralized indentation validation to ensure consistent block structure.
        
        Args:
            base_indent: The indentation level of the control flow header
            header_line: The header line text (for error messages)
            header_no: The header line number
            
        Returns:
            List of parsed statements in the block body
            
        Raises:
            N3SyntaxError: If body is empty or indentation is inconsistent
        """
        body: List[PageStatement] = []
        block_indent: Optional[int] = None
        block_start_line: Optional[int] = None
        
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            
            current_line = self.pos + 1
            stripped = nxt.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            # First line of block: expect indentation greater than base
            if block_indent is None:
                try:
                    info = self._expect_indent_greater_than(
                        nxt, base_indent, current_line, "control flow body"
                    )
                    block_indent = info.effective_level
                    block_start_line = current_line
                except N3SyntaxError:
                    # No indented content - block is empty
                    break
            else:
                # Subsequent lines: validate consistent indentation
                indent = self._indent(nxt)
                
                if indent < block_indent:
                    # Dedent - end of block
                    break
                elif indent > block_indent:
                    # Over-indented - inconsistent
                    hint = (
                        f"This line uses {indent} spaces, but the block started with "
                        f"{block_indent} spaces at line {block_start_line}.\n"
                        f"Hint: Use consistent indentation throughout the block."
                    )
                    raise self._error(
                        "Inconsistent indentation in control flow body",
                        current_line,
                        nxt,
                        hint=hint
                    )
                # indent == block_indent is OK, continue parsing
            
            stmt = self._parse_page_statement(block_indent)
            body.append(stmt)
        
        if not body:
            hint = "Control flow blocks must contain at least one statement"
            raise self._error(
                "Expected at least one statement in block",
                header_no,
                header_line,
                hint=hint
            )
        
        return body

    def _parse_if_block(self, line: str, line_no: int, base_indent: int) -> IfBlock:
        """
        Parse an if/elif/else conditional block.
        
        Args:
            line: The if statement line
            line_no: Line number
            base_indent: Parent indentation level
            
        Returns:
            Parsed IfBlock with condition, body, elifs, and else_body
        """
        stripped = line.strip()
        if not stripped.endswith(':'):
            hint = "All control flow statements (if, for, while) must end with ':'"
            raise self._error("Expected ':' after if condition", line_no, line, hint=hint)
        
        condition_text = stripped[3:-1].strip()
        if not condition_text:
            hint = "Provide a boolean expression after 'if' (e.g., if x > 10:)"
            raise self._error("Expected condition after 'if'", line_no, line, hint=hint)
        
        try:
            condition = self._parse_expression(condition_text)
        except N3SyntaxError:
            raise
        except Exception as exc:
            hint = "Condition must be a valid expression (e.g., x > 0, name == 'test')"
            raise self._error(f"Failed to parse if condition: {exc}", line_no, line, hint=hint)
        
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
                # Indentation greater than base belongs to previous branch
                break
            
            if stripped_nxt.startswith('elif '):
                self._advance()
                if not stripped_nxt.endswith(':'):
                    hint = "elif statements must end with ':' (e.g., elif x < 5:)"
                    raise self._error("Expected ':' after elif condition", self.pos, nxt, hint=hint)
                
                elif_condition_text = stripped_nxt[5:-1].strip()
                if not elif_condition_text:
                    hint = "Provide a boolean expression after 'elif'"
                    raise self._error("Expected condition after 'elif'", self.pos, nxt, hint=hint)
                
                try:
                    elif_condition = self._parse_expression(elif_condition_text)
                except N3SyntaxError:
                    raise
                except Exception as exc:
                    hint = "Condition must be a valid expression"
                    raise self._error(f"Failed to parse elif condition: {exc}", self.pos, nxt, hint=hint)
                
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
        """
        Parse a for loop statement.
        
        Args:
            line: The for loop line
            line_no: Line number
            base_indent: Parent indentation level
            
        Returns:
            Parsed ForLoop
        """
        stripped = line.strip()
        if not stripped.endswith(':'):
            hint = "for loops must end with ':' (e.g., for item in dataset items:)"
            raise self._error("Expected ':' after for loop header", line_no, line, hint=hint)
        
        match = re.match(r'for\s+(\w+)\s+in\s+(dataset|table|frame)\s+(\w+)\s*:$', stripped)
        if not match:
            hint = "Syntax: for <variable> in dataset|table|frame <source_name>:"
            raise self._error(
                "Invalid for loop syntax",
                line_no,
                line,
                hint=hint
            )
        
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
        """
        Parse a while loop statement.
        
        Args:
            line: The while loop line
            line_no: Line number
            base_indent: Parent indentation level
            
        Returns:
            Parsed WhileLoop
        """
        stripped = line.strip()
        if not stripped.endswith(':'):
            hint = "while loops must end with ':' (e.g., while count > 0:)"
            raise self._error("Expected ':' after while condition", line_no, line, hint=hint)
        
        condition_text = stripped[5:-1].strip()
        if not condition_text:
            hint = "Provide a boolean expression after 'while' (e.g., while x < 100:)"
            raise self._error("Expected condition after 'while'", line_no, line, hint=hint)
        
        try:
            condition = self._parse_expression(condition_text)
        except N3SyntaxError:
            raise
        except Exception as exc:
            hint = "Condition must be a valid expression"
            raise self._error(f"Failed to parse while condition: {exc}", line_no, line, hint=hint)
        
        self._loop_depth += 1
        try:
            body = self._parse_control_body(base_indent, line, line_no)
        finally:
            self._loop_depth = max(0, self._loop_depth - 1)
        
        return WhileLoop(condition=condition, body=body)

    def _parse_loop_control(self, kind: str, line_no: int, line: str) -> PageStatement:
        """
        Parse a loop control statement (break or continue).
        
        Args:
            kind: The control statement type ('break' or 'continue')
            line_no: Line number
            line: The line text
            
        Returns:
            BreakStatement or ContinueStatement
            
        Raises:
            N3SyntaxError: If not inside a loop
        """
        if self._loop_depth <= 0:
            hint = f"'{kind}' can only be used inside for or while loops"
            raise self._error(
                f"'{kind}' is only permitted inside a loop",
                line_no,
                line,
                hint=hint
            )
        
        if kind == 'break':
            return BreakStatement()
        if kind == 'continue':
            return ContinueStatement()
        
        raise self._error(f"Unsupported loop control '{kind}'", line_no, line)
