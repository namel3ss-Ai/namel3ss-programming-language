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
    RunPromptOperation,
    ToastOperation,
    UpdateOperation,
)
# KeywordRegistry import removed - class does not exist

from .datasets import DatasetParserMixin


class ActionParserMixin(DatasetParserMixin):
    """
    Parse action declarations with centralized validation.
    
    Handles parsing of interactive actions triggered by user events:
    - Database operations: update, insert
    - UI operations: show toast, go to page
    - Python integrations: call python module/method
    - Connector operations: ask connector with arguments
    - AI operations: run chain, run prompt
    
    Uses centralized indentation validation for consistent error messages.
    """

    def _parse_action(self, line: str, base_indent: int) -> Action:
        """
        Parse action definition with operations.
        
        Syntax:
            action "Name" [effect EFFECT]:
                when EVENT:
                    operation1
                    operation2
        
        Example:
            action "Save Record":
                when button_click:
                    update users set name = form.name where id = current_user
                    show toast "Saved successfully"
        """
        line_no = self.pos
        stripped_line = line.strip()
        match = re.match(r'action\s+"([^"]+)"(?:\s+effect\s+([\w\-]+))?\s*:?$', stripped_line, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                'Expected: action "Name":',
                self.pos,
                line,
                hint='Action definitions must have a name in quotes'
            )
        name = match.group(1)
        declared_effect = self._parse_effect_annotation(match.group(2), line_no, line)
        trigger: Optional[str] = None
        operations: List[ActionOperationType] = []
        
        # Validate indented block
        indent_info = self._expect_indent_greater_than(
            base_indent,
            context=f'action "{name}"',
            line_no=line_no
        )
        if not indent_info:
            raise self._error(
                f'Action "{name}" requires an indented block with operations',
                line_no,
                line,
                hint='Add indented operations like "update", "show toast", etc.'
            )
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
        return Action(name=name, trigger=trigger, operations=operations, declared_effect=declared_effect)

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
        if stripped.startswith('run prompt') or stripped.startswith('execute prompt'):
            return self._parse_run_prompt_operation(line or stripped, indent)
        return ToastOperation(message=stripped)

    def _parse_call_python_operation(self, line: str, base_indent: int) -> CallPythonOperation:
        """
        Parse Python function call operation.
        
        Syntax:
            call python "module.py" method "function" [with:]
                arg1 = value1
                arg2 = value2
        """
        stripped = line.strip()
        match = re.match(r'call\s+python\s+"([^\"]+)"\s+method\s+"([^\"]+)"(?:\s+with)?\s*:?', stripped)
        if not match:
            raise self._error(
                'Expected: call python "module.py" method "function" [with:]',
                self.pos,
                line,
                hint='Python calls need both module and method names in quotes'
            )
        module = match.group(1)
        method = match.group(2)
        arguments = {}
        if stripped.endswith(':'):
            arguments = self._parse_argument_block(base_indent)
        return CallPythonOperation(module=module, method=method, arguments=arguments)

    def _parse_ask_connector_operation(self, line: str, base_indent: int) -> AskConnectorOperation:
        """
        Parse connector invocation operation.
        
        Syntax:
            ask connector NAME [with:]
                arg1 = value1
        """
        stripped = line.strip()
        match = re.match(r'ask\s+connector\s+([\w\.\-]+)(?:\s+with)?\s*:?', stripped)
        if not match:
            raise self._error(
                'Expected: ask connector NAME [with:]',
                self.pos,
                line,
                hint='Connector operations need the connector name'
            )
        name = match.group(1)
        arguments = {}
        if stripped.endswith(':'):
            arguments = self._parse_argument_block(base_indent)
        return AskConnectorOperation(connector_name=name, arguments=arguments)

    def _parse_run_chain_operation(self, line: str, base_indent: int) -> RunChainOperation:
        """
        Parse AI chain execution operation.
        
        Syntax:
            run chain NAME [with:]
                input = value
        """
        stripped = line.strip()
        match = re.match(r'(?:run|execute)\s+chain\s+([\w\.\-]+)(?:\s+with)?\s*:?', stripped)
        if not match:
            raise self._error(
                'Expected: run chain NAME [with:]',
                self.pos,
                line,
                hint='Chain operations need the chain name'
            )
        name = match.group(1)
        inputs = {}
        if stripped.endswith(':'):
            inputs = self._parse_argument_block(base_indent)
        return RunChainOperation(chain_name=name, inputs=inputs)

    def _parse_run_prompt_operation(self, line: str, base_indent: int) -> RunPromptOperation:
        """
        Parse AI prompt execution operation.
        
        Syntax:
            run prompt NAME [with:]
                arg1 = value1
        """
        stripped = line.strip()
        match = re.match(r'(?:run|execute)\s+prompt\s+([\w\.\-]+)(?:\s+with)?\s*:?', stripped)
        if not match:
            raise self._error(
                'Expected: run prompt NAME [with:]',
                self.pos,
                line,
                hint='Prompt operations need the prompt name'
            )
        name = match.group(1)
        arguments = {}
        if stripped.endswith(':'):
            arguments = self._parse_argument_block(base_indent)
        return RunPromptOperation(prompt_name=name, arguments=arguments)

    def _parse_argument_block(self, parent_indent: int) -> Dict[str, Any]:
        """
        Parse indented block of name=value arguments.
        
        Validates that each line follows the 'name = expression' pattern.
        """
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
                    hint='Arguments use key = value syntax'
                )
            key = match.group(1)
            expr_text = match.group(2).strip()
            self._advance()
            arguments[key] = self._parse_expression(expr_text)
        return arguments
