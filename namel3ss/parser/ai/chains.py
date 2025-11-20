"""Chain parsing for multi-step AI workflows."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from namel3ss.ast import Chain, ChainStep, StepEvaluationConfig, WorkflowNode

if TYPE_CHECKING:
    from ..base import ParserBase


class ChainsParserMixin:
    """Mixin for parsing chain (multi-step workflow) definitions."""
    
    def _parse_chain(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> Chain:
        """
        Parse multi-step AI workflow chain definitions.
        
        Chains define sequences of AI operations (prompts, models, connectors)
        with optional control flow, memory management, and evaluation.
        
        Supported Syntaxes:
            1. Workflow Block (modern):
                chain "Name":
                    workflow:
                        step_1 using model_a
                        if condition: step_2
                        for item in dataset: step_3
            
            2. Pipeline Syntax (legacy):
                chain "Name":
                    steps:
                        - target: "prompt1"
                          kind: "prompt"
                        - target: "model1"
                          kind: "model"
        
        Features:
            - Sequential and parallel execution
            - Conditional branching (if/elif/else)
            - Iteration (for, while)
            - Memory read/write
            - Step evaluation and guardrails
            - Error handling per step
        """
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'chain\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                'Expected: chain "Name":',
                line_no,
                line,
                hint='Chain definitions require a name, e.g., chain "MyWorkflow":'
            )
        name = match.group(1)
        config_block = self._parse_kv_block(base_indent)
        description_raw = config_block.pop("description", config_block.pop("desc", None))
        description = str(description_raw) if description_raw is not None else None
        workflow_raw = config_block.pop("workflow", None)
        steps_raw = config_block.pop("steps", None)
        metadata_raw = config_block.pop("metadata", {})
        metadata = self._coerce_options_dict(metadata_raw)
        error_handling = config_block.pop("error_handling", None)
        chain_config = {}
        for key, val in config_block.items():
            if key not in {"workflow", "steps", "description", "metadata", "error_handling"}:
                chain_config[key] = self._transform_config(val)
        nodes: List[WorkflowNode] = []
        if workflow_raw is not None:
            if not isinstance(workflow_raw, list):
                raise self._error(
                    "Chain workflow must be a list of nodes",
                    line_no,
                    line,
                    hint='Use workflow: block with indented step declarations'
                )
            nodes = self._parse_workflow_block(workflow_raw)
        elif steps_raw is not None:
            if not isinstance(steps_raw, list):
                raise self._error(
                    "Chain steps must be a list",
                    line_no,
                    line,
                    hint='Use steps: with list of step configurations'
                )
            for step_raw in steps_raw:
                if not isinstance(step_raw, dict):
                    raise self._error(
                        "Each chain step must be a dict with target/kind/options",
                        line_no,
                        line,
                        hint='Step format: {target: "name", kind: "prompt", options: {...}}'
                    )
                target = step_raw.get("target")
                if not target:
                    raise self._error(
                        "Chain step missing 'target'",
                        line_no,
                        line,
                        hint='Each step requires target: "resource_name"'
                    )
                kind = step_raw.get("kind", "prompt")
                options = self._transform_config(step_raw.get("options", {}))
                evaluation_raw = step_raw.get("evaluation")
                evaluation_config: Optional[StepEvaluationConfig] = None
                if evaluation_raw is not None:
                    evaluation_config = self._parse_step_evaluation_config(
                        evaluation_raw, line_no, f'chain "{name}" step'
                    )
                memory_read = step_raw.get("memory_read", [])
                memory_write = step_raw.get("memory_write", [])
                on_error = step_raw.get("on_error")
                step = ChainStep(
                    target=str(target),
                    kind=str(kind),
                    options=options if isinstance(options, dict) else {},
                    evaluation=evaluation_config,
                    memory_read=memory_read if isinstance(memory_read, list) else [],
                    memory_write=memory_write if isinstance(memory_write, list) else [],
                    on_error=str(on_error) if on_error is not None else None,
                )
                node = WorkflowNode(kind="step", step=step, config={})
                nodes.append(node)
        else:
            raise self._error(
                "Chain must define either 'workflow:' or 'steps:'",
                line_no,
                line,
                hint='Use workflow: for modern control flow or steps: for simple pipelines'
            )
        return Chain(
            name=name,
            description=description,
            workflow=nodes,
            metadata=metadata,
            config=chain_config,
            error_handling=str(error_handling) if error_handling is not None else None,
        )
