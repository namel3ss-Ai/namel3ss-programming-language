from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from namel3ss.ast import (
    Action,
    ActionOperationType,
    AskConnectorOperation,
    CallPythonOperation,
    GoToPageOperation,
    RunChainOperation,
    ToastOperation,
    UpdateOperation,
)

from .datasets import DatasetParserMixin


class ActionParserMixin(DatasetParserMixin):
    """Parsing logic for action declarations inside pages."""

    def _parse_action(self, line: str, base_indent: int) -> Action:
        match = re.match(r'action\s+"([^"]+)"\s*:?$', line.strip())
        if not match:
            raise self._error("Expected: action \"Name\":", self.pos, line)
        name = match.group(1)
        trigger: Optional[str] = None
        operations: List[ActionOperationType] = []
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
            if stripped.startswith('when '):
                trigger = stripped[len('when '):].rstrip(':').strip()
                self._advance()
                op_base_indent = self._indent(nxt)
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
                    operations.append(op)
            else:
                op = self._parse_action_operation(stripped)
                operations.append(op)
        if trigger is None:
            trigger = ''
        return Action(name=name, trigger=trigger, operations=operations)

    def _parse_action_operation(self, stripped: str) -> ActionOperationType:
        line = self._advance()
        indent = self._indent(line or '') if line is not None else 0
        if stripped.startswith('update'):
            match = re.match(r'update\s+([^\s]+)\s+set\s+(.+)', stripped)
            if not match:
                raise self._error(
                    "Expected: update table set column = value [where ...]",
                    self.pos,
                    line,
                )
            table = match.group(1)
            rest = match.group(2)
            if ' where ' in rest:
                set_expr, where_expr = rest.split(' where ', 1)
                return UpdateOperation(
                    table=table,
                    set_expression=set_expr.strip(),
                    where_expression=where_expr.strip(),
                )
            return UpdateOperation(table=table, set_expression=rest.strip())
        if stripped.startswith('insert into'):
            return ToastOperation(message=f"Insert operation: {stripped}")
        if stripped.startswith('show toast'):
            match = re.match(r'show\s+toast\s+"([^"]+)"', stripped)
            if not match:
                raise self._error("Expected: show toast \"Message\"", self.pos, line)
            return ToastOperation(message=match.group(1))
        if stripped.startswith('go to page'):
            match = re.match(r'go\s+to\s+page\s+"([^"]+)"', stripped)
            if not match:
                raise self._error("Expected: go to page \"Page Name\"", self.pos, line)
            return GoToPageOperation(page_name=match.group(1))
        if stripped.startswith('call python'):
            return self._parse_call_python_operation(line or stripped, indent)
        if stripped.startswith('ask connector'):
            return self._parse_ask_connector_operation(line or stripped, indent)
        if stripped.startswith('run chain') or stripped.startswith('execute chain'):
            return self._parse_run_chain_operation(line or stripped, indent)
        return ToastOperation(message=stripped)

    def _parse_call_python_operation(self, line: str, base_indent: int) -> CallPythonOperation:
        stripped = line.strip()
        match = re.match(r'call\s+python\s+"([^\"]+)"\s+method\s+"([^\"]+)"(?:\s+with)?\s*:?', stripped)
        if not match:
            raise self._error(
                'Expected: call python "module.py" method "function" [with:]',
                self.pos,
                line,
            )
        module = match.group(1)
        method = match.group(2)
        arguments = {}
        if stripped.endswith(':'):
            arguments = self._parse_argument_block(base_indent)
        return CallPythonOperation(module=module, method=method, arguments=arguments)

    def _parse_ask_connector_operation(self, line: str, base_indent: int) -> AskConnectorOperation:
        stripped = line.strip()
        match = re.match(r'ask\s+connector\s+([\w\.\-]+)(?:\s+with)?\s*:?', stripped)
        if not match:
            raise self._error(
                'Expected: ask connector NAME [with:]',
                self.pos,
                line,
            )
        name = match.group(1)
        arguments = {}
        if stripped.endswith(':'):
            arguments = self._parse_argument_block(base_indent)
        return AskConnectorOperation(connector_name=name, arguments=arguments)

    def _parse_run_chain_operation(self, line: str, base_indent: int) -> RunChainOperation:
        stripped = line.strip()
        match = re.match(r'(?:run|execute)\s+chain\s+([\w\.\-]+)(?:\s+with)?\s*:?', stripped)
        if not match:
            raise self._error(
                'Expected: run chain NAME [with:]',
                self.pos,
                line,
            )
        name = match.group(1)
        inputs = {}
        if stripped.endswith(':'):
            inputs = self._parse_argument_block(base_indent)
        return RunChainOperation(chain_name=name, inputs=inputs)

    def _parse_argument_block(self, parent_indent: int) -> Dict[str, Any]:
        arguments: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            match = re.match(r'([\w_]+)\s*=\s*(.+)$', stripped)
            if not match:
                raise self._error(
                    "Expected 'name = expression' inside argument block",
                    self.pos + 1,
                    nxt,
                )
            key = match.group(1)
            expr_text = match.group(2).strip()
            self._advance()
            arguments[key] = self._parse_expression(expr_text)
        return arguments
