from __future__ import annotations

import re
from typing import Any, Dict, List

from namel3ss.ast import Evaluator, Metric, Guardrail, EvalSuiteDefinition, EvalMetricSpec
# KeywordRegistry import removed - class does not exist

from .base import ParserBase, N3SyntaxError


class EvaluationParserMixin(ParserBase):
    """
    Parse evaluation system declarations with centralized validation.
    
    Handles parsing of AI/ML evaluation constructs:
    - Evaluators: LLM-based or rule-based evaluation functions
    - Metrics: Aggregated evaluation measurements
    - Guardrails: Safety checks with actions on failure
    - Eval Suites: Complete evaluation specifications with datasets and metrics
    
    Syntax examples:
        evaluator "Accuracy":
            kind: classification
            provider: sklearn
            
        metric "F1 Score":
            evaluator: accuracy_eval
            aggregation: mean
            
        guardrail "Content Safety":
            evaluators: [toxicity_check, pii_detector]
            action: block
            
        eval_suite test_suite:
            dataset: test_data
            target_chain: my_chain
            metrics:
                - { name: "accuracy", type: "classification" }
    
    Uses centralized indentation validation for consistent error messages.
    """

    def _parse_evaluator(self, line: str, line_no: int, base_indent: int) -> Evaluator:
        """
        Parse evaluator definition with kind and provider.
        
        Syntax:
            evaluator "Name":
                kind: classification
                provider: sklearn
                config:
                    threshold: 0.5
        """
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'evaluator\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                'Expected: evaluator "Name":',
                line_no,
                line,
                hint='Evaluator definitions must have a name in quotes'
            )
        name = match.group(1)
        
        # Validate indented block
        indent_info = self._expect_indent_greater_than(
            base_indent,
            context=f'evaluator "{name}"',
            line_no=line_no
        )
        if not indent_info:
            raise self._error(
                f'Evaluator "{name}" requires an indented configuration block',
                line_no,
                line,
                hint='Add indented lines with kind, provider, and config'
            )
        
        block = self._parse_kv_block(base_indent)
        kind_raw = block.get("kind")
        provider_raw = block.get("provider")
        if kind_raw is None:
            raise self._error("Evaluator must define 'kind:'", line_no, line)
        if provider_raw is None:
            raise self._error("Evaluator must define 'provider:'", line_no, line)
        config_raw = block.get("config", {})
        config = self._transform_config(config_raw) if isinstance(config_raw, dict) else {}
        return Evaluator(
            name=name,
            kind=str(kind_raw),
            provider=str(provider_raw),
            config=config if isinstance(config, dict) else {},
        )

    def _parse_metric(self, line: str, line_no: int, base_indent: int) -> Metric:
        """
        Parse metric definition with evaluator reference.
        
        Syntax:
            metric "Name":
                evaluator: evaluator_name
                aggregation: mean
                params:
                    threshold: 0.8
        """
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'metric\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                'Expected: metric "Name":',
                line_no,
                line,
                hint='Metric definitions must have a name in quotes'
            )
        name = match.group(1)
        
        # Validate indented block
        indent_info = self._expect_indent_greater_than(
            base_indent,
            context=f'metric "{name}"',
            line_no=line_no
        )
        if not indent_info:
            raise self._error(
                f'Metric "{name}" requires an indented configuration block',
                line_no,
                line,
                hint='Add indented "evaluator:" line'
            )
        
        block = self._parse_kv_block(base_indent)
        evaluator_name = block.get("evaluator")
        if evaluator_name is None:
            raise self._error("Metric must define 'evaluator:'", line_no, line)
        aggregation = block.get("aggregation")
        params_raw = block.get("params", {})
        params = self._transform_config(params_raw) if isinstance(params_raw, dict) else {}
        return Metric(
            name=name,
            evaluator=str(evaluator_name),
            aggregation=str(aggregation) if aggregation is not None else None,
            params=params if isinstance(params, dict) else {},
        )

    def _parse_guardrail(self, line: str, line_no: int, base_indent: int) -> Guardrail:
        """
        Parse guardrail definition with evaluators and action.
        
        Syntax:
            guardrail "Name":
                evaluators: [check1, check2]
                action: block
                message: "Safety check failed"
        """
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'guardrail\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                'Expected: guardrail "Name":',
                line_no,
                line,
                hint='Guardrail definitions must have a name in quotes'
            )
        name = match.group(1)
        
        # Validate indented block
        indent_info = self._expect_indent_greater_than(
            base_indent,
            context=f'guardrail "{name}"',
            line_no=line_no
        )
        if not indent_info:
            raise self._error(
                f'Guardrail "{name}" requires an indented configuration block',
                line_no,
                line,
                hint='Add indented lines with evaluators and action'
            )
        
        block = self._parse_kv_block(base_indent)
        evaluators_raw = block.get("evaluators")
        evaluators: List[str] = []
        if isinstance(evaluators_raw, (list, tuple)):
            evaluators = [str(item) for item in evaluators_raw if item]
        elif isinstance(evaluators_raw, str):
            evaluators = [evaluators_raw]
        if not evaluators:
            raise self._error("Guardrail must define at least one evaluator.", line_no, line)
        action_raw = block.get("action")
        if not action_raw:
            raise self._error("Guardrail must define 'action:'", line_no, line)
        message_raw = block.get("message")
        return Guardrail(
            name=name,
            evaluators=[str(item) for item in evaluators],
            action=str(action_raw),
            message=str(message_raw) if message_raw is not None else None,
        )

    def _parse_eval_suite(self, line: str, line_no: int, base_indent: int) -> EvalSuiteDefinition:
        """
        Parse eval_suite definition with comprehensive evaluation specification.
        
        Syntax:
            eval_suite test_suite:
                dataset: test_data
                target_chain: my_chain
                judge_llm: gpt4
                rubric: '''Evaluation criteria...'''
                metrics:
                    - { name: "accuracy", type: "classification" }
                    - { name: "latency", type: "performance" }
        """
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'eval_suite\s+(\w+)', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error(
                'Expected: eval_suite <identifier>',
                line_no,
                line,
                hint='Eval suite names must be valid identifiers'
            )
        name = match.group(1)
        
        # Validate indented block
        indent_info = self._expect_indent_greater_than(
            base_indent,
            context=f'eval_suite {name}',
            line_no=line_no
        )
        if not indent_info:
            raise self._error(
                f'Eval suite {name} requires an indented configuration block',
                line_no,
                line,
                hint='Add indented lines with dataset, target_chain, and metrics'
            )
        
        dataset_name: str = ""
        target_chain_name: str = ""
        metrics: List[EvalMetricSpec] = []
        judge_llm_name: str | None = None
        rubric: str | None = None
        description: str | None = None
        metadata: Dict[str, Any] = {}
        
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped_line = nxt.strip()
            if not stripped_line or stripped_line.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            
            lowered = stripped_line.lower()
            
            if lowered.startswith('dataset:'):
                value = stripped_line[len('dataset:'):].strip().strip('"')
                dataset_name = value
                self._advance()
                continue
            
            if lowered.startswith('target_chain:'):
                value = stripped_line[len('target_chain:'):].strip().strip('"')
                target_chain_name = value
                self._advance()
                continue
            
            if lowered.startswith('description:'):
                value = stripped_line[len('description:'):].strip().strip('"')
                description = value
                self._advance()
                continue
            
            if lowered.startswith('judge_llm:'):
                value = stripped_line[len('judge_llm:'):].strip().strip('"')
                judge_llm_name = value
                self._advance()
                continue
            
            if lowered.startswith('rubric:'):
                # Handle multiline rubric
                value = stripped_line[len('rubric:'):].strip()
                if value.startswith('"""') or value.startswith("'''"):
                    # Multiline string
                    delimiter = value[:3]
                    content = value[3:]
                    if content.endswith(delimiter):
                        rubric = content[:-3].strip()
                        self._advance()
                    else:
                        lines = [content]
                        self._advance()
                        while self.pos < len(self.lines):
                            nxt_line = self._peek()
                            if nxt_line is None:
                                break
                            lines.append(nxt_line.rstrip())
                            self._advance()
                            if nxt_line.rstrip().endswith(delimiter):
                                break
                        rubric = '\n'.join(lines).strip()
                        if rubric.endswith(delimiter):
                            rubric = rubric[:-3].strip()
                else:
                    rubric = value.strip('"')
                    self._advance()
                continue
            
            if lowered.startswith('metrics:'):
                block_indent = indent
                self._advance()
                metrics = self._parse_eval_metrics_list(block_indent)
                continue
            
            if lowered.startswith('metadata:') or lowered.startswith('config:'):
                block_indent = indent
                self._advance()
                metadata_block = self._parse_kv_block(block_indent)
                metadata.update(metadata_block)
                continue
            
            # Unknown directive - treat as metadata
            key_match = re.match(r'([\w\s]+):\s*(.*)$', stripped_line)
            if key_match:
                key = key_match.group(1).strip()
                remainder = key_match.group(2)
                self._advance()
                if remainder:
                    metadata[key] = self._coerce_scalar(remainder)
                else:
                    block = self._parse_kv_block(indent)
                    metadata[key] = block
                continue
            
            raise self._error("Unknown directive inside eval_suite block", self.pos + 1, nxt)
        
        if not dataset_name:
            raise self._error("eval_suite must define 'dataset:'", line_no, line)
        if not target_chain_name:
            raise self._error("eval_suite must define 'target_chain:'", line_no, line)
        
        return EvalSuiteDefinition(
            name=name,
            dataset_name=dataset_name,
            target_chain_name=target_chain_name,
            metrics=metrics,
            judge_llm_name=judge_llm_name,
            rubric=rubric,
            description=description,
            metadata=metadata,
        )
    
    def _parse_eval_metrics_list(self, base_indent: int) -> List[EvalMetricSpec]:
        """
        Parse list of metric specifications in eval suite.
        
        Supports both inline and block formats:
            - { name: "accuracy", type: "classification" }
            - name: accuracy
              type: classification
        """
        metrics: List[EvalMetricSpec] = []
        
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
            
            if not stripped.startswith('-'):
                break
            
            # Remove leading dash and parse metric spec
            content = stripped[1:].strip()
            
            # Check if it's inline object syntax: - { name: "...", type: "..." }
            if content.startswith('{') and content.endswith('}'):
                # Inline dict
                try:
                    # Simple parser for inline dict
                    metric_dict = self._parse_inline_dict(content)
                    metrics.append(self._parse_metric_spec_from_dict(metric_dict))
                    self._advance()
                    continue
                except Exception as exc:
                    raise self._error(f"Invalid metric specification: {exc}", self.pos + 1, nxt)
            else:
                # Block format - advance and parse indented block
                self._advance()
                metric_indent = indent + 2
                metric_dict = self._parse_kv_block(metric_indent)
                metrics.append(self._parse_metric_spec_from_dict(metric_dict))
        
        return metrics
    
    def _parse_inline_dict(self, text: str) -> Dict[str, Any]:
        """Parse inline dict like { name: "x", type: "y" }."""
        if not text.startswith('{') or not text.endswith('}'):
            raise ValueError("Not a dict")
        
        content = text[1:-1].strip()
        result: Dict[str, Any] = {}
        
        # Simple split by comma (not robust for nested structures, but sufficient here)
        pairs = []
        current = []
        depth = 0
        in_quotes = False
        quote_char = None
        
        for char in content:
            if char in ('"', "'") and (not in_quotes or char == quote_char):
                if in_quotes:
                    in_quotes = False
                    quote_char = None
                else:
                    in_quotes = True
                    quote_char = char
            elif char in ('{', '[') and not in_quotes:
                depth += 1
            elif char in ('}', ']') and not in_quotes:
                depth -= 1
            elif char == ',' and depth == 0 and not in_quotes:
                pairs.append(''.join(current).strip())
                current = []
                continue
            current.append(char)
        
        if current:
            pairs.append(''.join(current).strip())
        
        for pair in pairs:
            if ':' not in pair:
                continue
            key, value = pair.split(':', 1)
            key = key.strip().strip('"').strip("'")
            value = value.strip()
            result[key] = self._coerce_scalar(value)
        
        return result
    
    def _parse_metric_spec_from_dict(self, data: Dict[str, Any]) -> EvalMetricSpec:
        """Convert parsed dict to EvalMetricSpec."""
        name = data.get("name")
        metric_type = data.get("type")
        
        if not name:
            raise ValueError("Metric must have 'name' field")
        if not metric_type:
            raise ValueError("Metric must have 'type' field")
        
        # All other fields go into config
        config = {k: v for k, v in data.items() if k not in ("name", "type")}
        
        return EvalMetricSpec(
            name=str(name),
            type=str(metric_type),
            config=config,
        )
