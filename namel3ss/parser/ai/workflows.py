"""Workflow parsing for chain control flow structures."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from namel3ss.ast import (
    ChainStep,
    Expression,
    WorkflowForBlock,
    WorkflowIfBlock,
    WorkflowNode,
    WorkflowWhileBlock,
)

if TYPE_CHECKING:
    from ..base import ParserBase


class WorkflowParserMixin:
    """Mixin for parsing workflow blocks and control flow structures."""
    
    def _parse_workflow_block(self: 'ParserBase', parent_indent: int) -> List[WorkflowNode]:
        """
        Parse structured workflow block containing multiple workflow nodes.
        
        Parses sequences of workflow steps, conditionals (if/elif/else),
        and loops (for/while) within chain definitions.
        
        Syntax:
            steps:
                - step "Process":
                    kind: prompt
                    target: "classifier"
                - if condition:
                    then:
                        - step "HandleTrue":
                - for item in dataset "Data":
                    - step "ProcessItem":
        """
        nodes: List[WorkflowNode] = []
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            if indent <= parent_indent:
                break
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if not stripped.startswith('-'):
                raise self._error("Workflow entries must begin with '-'", self.pos + 1, nxt)
            line_no = self.pos + 1
            entry_line = stripped[1:].strip()
            self._advance()
            nodes.append(self._parse_workflow_entry(entry_line, indent, line_no))
        return nodes

    def _parse_workflow_entry(self: 'ParserBase', header: str, indent: int, line_no: int) -> WorkflowNode:
        """
        Dispatch workflow entry parsing based on entry type.
        
        Routes to specialized parsers for different workflow node types:
        - step: Execute operations (prompts, functions, database)
        - if: Conditional branching with then/elif/else
        - for: Iteration over collections or datasets
        - while: Conditional looping with max_iterations
        """
        lowered = header.lower()
        if lowered.startswith('step'):
            return self._parse_workflow_step_entry(header, indent, line_no)
        if lowered.startswith('if '):
            return self._parse_workflow_if_entry(header, indent, line_no)
        if lowered.startswith('for '):
            return self._parse_workflow_for_entry(header, indent, line_no)
        if lowered.startswith('while '):
            return self._parse_workflow_while_entry(header, indent, line_no)
        raise self._error("Unsupported workflow entry", line_no, header)

    def _parse_workflow_step_entry(self: 'ParserBase', header: str, indent: int, line_no: int) -> ChainStep:
        """
        Parse individual workflow step with configuration.
        
        Steps define executable operations including AI prompts, Python functions,
        database queries, and API calls. Supports error handling and evaluation.
        
        Syntax:
            - step "StepName":
                kind: prompt|python|database|api
                target: "ResourceName"
                stop_on_error: true|false
                evaluation:
                    evaluators: ["eval1", "eval2"]
                    guardrail: "GuardrailName"
        """
        match = re.match(r'step(?:\s+"([^"]+)")?(?:\s*:)?$', header, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                "Expected: - step \"Name\":",
                line_no,
                header,
                hint='Workflow steps require format: - step "Name":'
            )
        name = match.group(1)
        config = self._parse_kv_block(indent)
        normalized = {key: self._transform_config(value) for key, value in config.items()}
        kind_value = normalized.pop('kind', normalized.pop('type', None))
        if kind_value is None:
            raise self._error("Workflow step must define 'kind:'", line_no, header)
        target_value = normalized.pop('target', normalized.pop('connector', None))
        if target_value is None:
            target_value = kind_value
        stop_on_error_raw = normalized.pop('stop_on_error', None)
        continue_on_error_raw = normalized.pop('continue_on_error', None)
        if stop_on_error_raw is not None:
            stop_on_error = self._parse_bool(str(stop_on_error_raw))
        elif continue_on_error_raw is not None:
            stop_on_error = not self._parse_bool(str(continue_on_error_raw))
        else:
            stop_on_error = True
        evaluation_value = normalized.pop('evaluation', None)
        evaluation = None
        if evaluation_value is not None:
            evaluation = self._parse_step_evaluation_config(evaluation_value, line_no, header)
        options_value = normalized.pop('options', {})
        options = self._coerce_options_dict(options_value)
        for key, value in normalized.items():
            options[key] = value
        return ChainStep(
            kind=str(kind_value),
            target=str(target_value),
            options=options,
            name=name,
            stop_on_error=stop_on_error,
            evaluation=evaluation,
        )

    def _parse_workflow_if_entry(self: 'ParserBase', header: str, indent: int, line_no: int) -> WorkflowIfBlock:
        """
        Parse conditional workflow block with if/elif/else branches.
        
        Supports complex conditional logic with multiple branches,
        allowing dynamic workflow routing based on runtime conditions.
        
        Syntax:
            - if expression:
                then:
                    - step "TrueCase":
                elif other_condition:
                    - step "ElseIf":
                else:
                    - step "FalseCase":
        """
        if not header.rstrip().endswith(':'):
            raise self._error(
                "Workflow if entries must end with ':'",
                line_no,
                header,
                hint='Add colon at end: - if condition:'
            )
        condition_text = header[:-1].strip()[2:].strip()
        if not condition_text:
            raise self._error(
                "Workflow if condition cannot be empty",
                line_no,
                header,
                hint='Provide a condition expression after if'
            )
        condition_expr = self._parse_expression(condition_text)
        then_steps: List[WorkflowNode] = []
        elif_branches: List[Tuple[Expression, List[WorkflowNode]]] = []
        else_steps: List[WorkflowNode] = []
        seen_then = False
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            current_indent = self._indent(nxt)
            if current_indent <= indent:
                break
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            lowered = stripped.lower()
            if lowered.startswith('then:'):
                self._advance()
                then_steps = self._parse_workflow_block(current_indent)
                seen_then = True
                continue
            if lowered.startswith('elif '):
                self._advance()
                branch_text = stripped[4:].strip()
                if not branch_text.endswith(':'):
                    raise self._error("Workflow elif must end with ':'", self.pos, stripped)
                branch_expr = self._parse_expression(branch_text[:-1].strip())
                branch_steps = self._parse_workflow_block(current_indent)
                elif_branches.append((branch_expr, branch_steps))
                continue
            if lowered.startswith('else:'):
                self._advance()
                else_steps = self._parse_workflow_block(current_indent)
                continue
            raise self._error("Unexpected entry inside workflow if block", self.pos + 1, nxt)
        if not then_steps:
            raise self._error("Workflow if block must define a 'then:' section", line_no, header)
        return WorkflowIfBlock(condition=condition_expr, then_steps=then_steps, elif_steps=elif_branches, else_steps=else_steps)

    def _parse_workflow_for_entry(self: 'ParserBase', header: str, indent: int, line_no: int) -> WorkflowForBlock:
        """
        Parse for-loop workflow block for iteration.
        
        Iterates over datasets, lists, or expressions, executing workflow steps
        for each item. Supports max_iterations limit for safety.
        
        Syntax:
            - for item in dataset "DatasetName":
                max_iterations: 100
                body:
                    - step "ProcessItem":
                        
            - for row in expression:
                - step "Transform":
        """
        match = re.match(r'for\s+([A-Za-z_][\w]*)\s+in\s+(.+):$', header, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                "Expected: - for item in <expression>:",
                line_no,
                header,
                hint='Use format: - for item in dataset "Name":'
            )
        loop_var = match.group(1)
        source_text = match.group(2).strip()
        source_kind = "expression"
        source_name: Optional[str] = None
        source_expression: Optional[Expression] = None
        lowered = source_text.lower()
        if lowered.startswith('dataset '):
            source_kind = "dataset"
            source_name = self._strip_quotes(source_text.split(None, 1)[1].strip())
        else:
            source_expression = self._parse_expression(source_text)
        config = self._parse_workflow_optional_config(indent)
        max_iterations = None
        if config:
            max_iterations = self._coerce_int(config.pop('max_iterations', config.pop('limit', None)))
            if max_iterations is not None and max_iterations <= 0:
                max_iterations = None
            if config:
                unknown = ", ".join(config.keys())
                raise self._error(f"Unsupported options in workflow for block: {unknown}", line_no, header)
        body = self._parse_workflow_block(indent)
        return WorkflowForBlock(
            loop_var=loop_var,
            source_kind=source_kind,
            source_name=source_name,
            source_expression=source_expression,
            body=body,
            max_iterations=max_iterations,
        )

    def _parse_workflow_while_entry(self: 'ParserBase', header: str, indent: int, line_no: int) -> WorkflowWhileBlock:
        """
        Parse while-loop workflow block for conditional iteration.
        
        Repeats workflow steps while condition remains true,
        with optional max_iterations safety limit.
        
        Syntax:
            - while condition:
                max_iterations: 10
                body:
                    - step "Retry":
        """
        if not header.rstrip().endswith(':'):
            raise self._error(
                "Workflow while entries must end with ':'",
                line_no,
                header,
                hint='Add colon at end: - while condition:'
            )
        condition_text = header[:-1].strip()[5:].strip()
        if not condition_text:
            raise self._error(
                "Workflow while condition cannot be empty",
                line_no,
                header,
                hint='Provide a condition expression after while'
            )
        condition_expr = self._parse_expression(condition_text)
        config = self._parse_workflow_optional_config(indent)
        max_iterations = None
        if config:
            max_iterations = self._coerce_int(config.pop('max_iterations', config.pop('limit', None)))
            if max_iterations is not None and max_iterations <= 0:
                max_iterations = None
            if config:
                unknown = ", ".join(config.keys())
                raise self._error(f"Unsupported options in workflow while block: {unknown}", line_no, header)
        body = self._parse_workflow_block(indent)
        return WorkflowWhileBlock(condition=condition_expr, body=body, max_iterations=max_iterations)

    def _parse_workflow_optional_config(self: 'ParserBase', parent_indent: int) -> Dict[str, Any]:
        """
        Parse optional configuration block for workflow control structures.
        
        Extracts configuration parameters like max_iterations for loops,
        stopping when encountering workflow entries (lines starting with -).
        
        Returns:
            Dictionary of configuration key-value pairs
        """
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent or stripped.startswith('-'):
                break
            return self._parse_kv_block(parent_indent)
        return {}
