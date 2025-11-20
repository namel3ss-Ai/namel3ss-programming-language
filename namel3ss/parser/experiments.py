from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List

from namel3ss.ast import Experiment, ExperimentMetric, ExperimentVariant, ExperimentComparison
# KeywordRegistry import removed - class does not exist

from .base import ParserBase


class ExperimentParserMixin(ParserBase):
    """
    Parse experiment declarations with centralized validation.
    
    Handles parsing of ML/AI experiment specifications including:
    - Experiment metadata and configuration
    - Variant definitions (different models/chains to compare)
    - Evaluation metrics and datasets
    - Comparison specifications (baseline vs challengers)
    - Training and tuning job references
    
    Uses centralized indentation validation for consistent error messages.
    """

    _EXPERIMENT_HEADER = re.compile(r'^experiment\s+"([^"]+)"\s*:?', re.IGNORECASE)

    def _parse_experiment(self, line: str, line_no: int, base_indent: int) -> Experiment:
        """
        Parse experiment definition with variants and metrics.
        
        Syntax:
            experiment "Name":
                description: "..."
                variants:
                    variant_a uses model model_a
                    variant_b uses chain chain_b
                metrics:
                    accuracy from evaluator eval_1 goal maximize
                compare:
                    baseline: variant_a
                    challengers: [variant_b]
        """
        match = self._EXPERIMENT_HEADER.match(line.strip())
        if not match:
            raise self._error(
                'Expected: experiment "Name":',
                line_no,
                line,
                hint='Experiment definitions must have a name in quotes'
            )
        name = match.group(1)
        experiment = Experiment(name=name)
        
        # Validate indented block
        indent_info = self._expect_indent_greater_than(
            base_indent,
            context=f'experiment "{name}"',
            line_no=line_no
        )
        if not indent_info:
            raise self._error(
                f'Experiment "{name}" requires an indented configuration block',
                line_no,
                line,
                hint='Add indented lines with variants, metrics, etc.'
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
            lowered = stripped.lower()
            if lowered.startswith('description:'):
                experiment.description = stripped[len('description:'):].strip().strip('"')
                self._advance()
                continue
            if lowered.startswith('metadata:') or lowered.startswith('config:'):
                block_indent = indent
                self._advance()
                metadata = self._parse_kv_block(block_indent)
                experiment.metadata.update(metadata)
                continue
            if lowered.startswith('training_jobs:'):
                block_indent = indent
                self._advance()
                experiment.training_jobs.extend(self._parse_string_list(block_indent))
                continue
            if lowered.startswith('tuning_jobs:'):
                block_indent = indent
                self._advance()
                experiment.tuning_jobs.extend(self._parse_string_list(block_indent))
                continue
            if lowered.startswith('eval_datasets:') or lowered.startswith('evaluation_datasets:'):
                block_indent = indent
                self._advance()
                experiment.eval_datasets.extend(self._parse_string_list(block_indent))
                continue
            if lowered.startswith('eval_metrics:') or lowered.startswith('evaluation_metrics:'):
                block_indent = indent
                self._advance()
                experiment.eval_metrics.extend(self._parse_string_list(block_indent))
                continue
            if lowered.startswith('variants:'):
                block_indent = indent
                self._advance()
                experiment.variants.extend(self._parse_experiment_variants(block_indent))
                continue
            if lowered.startswith('metrics:'):
                block_indent = indent
                self._advance()
                lookahead = self._peek_next_content_line()
                if lookahead and self._indent(lookahead) > block_indent and lookahead.strip().startswith('-'):
                    experiment.eval_metrics.extend(self._parse_string_list(block_indent))
                else:
                    experiment.metrics.extend(self._parse_experiment_metrics(block_indent))
                continue
            if lowered.startswith('compare:') or lowered.startswith('comparison:'):
                block_indent = indent
                self._advance()
                data = self._parse_kv_block(block_indent)
                experiment.comparison = self._parse_experiment_comparison(data)
                continue
            # Treat unknown directives inside experiment as metadata entries using nested blocks
            key_match = re.match(r'([\w\s]+):\s*(.*)$', stripped)
            if key_match:
                key = key_match.group(1).strip()
                remainder = key_match.group(2)
                self._advance()
                if remainder:
                    experiment.metadata[key] = self._coerce_scalar(remainder)
                else:
                    block = self._parse_kv_block(indent)
                    experiment.metadata[key] = block
                continue
            raise self._error("Unknown directive inside experiment block", self.pos + 1, nxt)

        return experiment

    def _parse_experiment_variants(self, parent_indent: int) -> List[ExperimentVariant]:
        """
        Parse experiment variant specifications.
        
        Each variant specifies a model or chain to include in the experiment.
        
        Example:
            variant_a uses model gpt4
            variant_b uses chain my_chain with temperature=0.7
        """
        variants: List[ExperimentVariant] = []
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            has_block = stripped.endswith(':')
            if has_block:
                stripped = stripped[:-1].strip()
            variant = self._parse_variant_line(stripped, self.pos + 1, line)
            self._advance()
            if has_block:
                block = self._parse_kv_block(indent)
                variant.config.update(block)
            variants.append(variant)
        return variants

    def _parse_variant_line(self, text: str, line_no: int, raw_line: str) -> ExperimentVariant:
        """
        Parse a single variant specification line.
        
        Format: <name> uses model|chain <target> [with key=value ...]
        """
        match = re.match(r'^([A-Za-z0-9_\-]+)\s+uses\s+(model|chain)\s+([A-Za-z0-9_\-\.]+)(?:\s+(.*))?$', text)
        if not match:
            raise self._error(
                "Expected '<name> uses model|chain <target>'",
                line_no,
                raw_line,
                hint='Variants specify what model or chain to test'
            )
        name = match.group(1)
        target_type = match.group(2)
        target_name = match.group(3)
        remainder = match.group(4) or ''
        config: Dict[str, Any] = {}
        remainder = remainder.strip()
        if remainder:
            if remainder.lower().startswith('with '):
                remainder = remainder[5:].strip()
            if remainder:
                config.update(self._parse_inline_assignments(remainder))
        return ExperimentVariant(name=name, target_type=target_type, target_name=target_name, config=config)

    def _parse_inline_assignments(self, text: str) -> Dict[str, Any]:
        assignments: Dict[str, Any] = {}
        # Allow comma-separated assignments or space-separated key=value pairs
        parts: List[str] = []
        if ',' in text:
            parts = [segment.strip() for segment in text.split(',') if segment.strip()]
        else:
            try:
                parts = shlex.split(text)
            except ValueError:
                parts = text.split()
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                assignments[key.strip()] = self._coerce_scalar(value.strip())
            else:
                assignments[part] = True
        return assignments

    def _parse_experiment_metrics(self, parent_indent: int) -> List[ExperimentMetric]:
        """
        Parse experiment metric specifications.
        
        Metrics define how to evaluate variant performance.
        
        Example:
            accuracy from evaluator eval_1 goal maximize
            latency from evaluator eval_2 goal minimize
        """
        metrics: List[ExperimentMetric] = []
        while self.pos < len(self.lines):
            line = self._peek()
            if line is None:
                break
            indent = self._indent(line)
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            has_block = stripped.endswith(':')
            if has_block:
                stripped = stripped[:-1].strip()
            metric = self._parse_metric_line(stripped, self.pos + 1, line)
            self._advance()
            if has_block:
                block = self._parse_kv_block(indent)
                metric.metadata.update(block)
            metrics.append(metric)
        return metrics

    def _parse_experiment_comparison(self, data: Dict[str, Any]) -> ExperimentComparison:
        """
        Parse comparison specification from key-value data.
        
        Defines baseline model and challenger variants for comparison.
        """
        baseline = data.pop('baseline_model', data.pop('baseline', None))
        best_of = data.pop('best_of', data.pop('select', None))
        challengers = data.pop('challengers', data.pop('compare', []))
        if isinstance(challengers, str):
            challengers_list = [self._strip_quotes(challengers)]
        elif isinstance(challengers, list):
            challengers_list = [self._strip_quotes(str(item)) for item in challengers if item is not None]
        else:
            challengers_list = []
        metadata = {key: self._coerce_scalar(value) for key, value in data.items()}
        baseline_value = self._strip_quotes(str(baseline)) if baseline is not None else None
        best_of_value = self._strip_quotes(str(best_of)) if best_of is not None else None
        challengers_value = [entry for entry in challengers_list if entry]
        return ExperimentComparison(
            baseline_model=baseline_value or None,
            best_of=best_of_value or None,
            challengers=challengers_value,
            metadata=metadata,
        )

    def _parse_metric_line(self, text: str, line_no: int, raw_line: str) -> ExperimentMetric:
        """
        Parse a single metric specification line.
        
        Format: <name> [from <kind> <source>] [goal <maximize|minimize>] [with ...]
        """
        tokens = text.split()
        if not tokens:
            raise self._error(
                "Empty metric definition",
                line_no,
                raw_line,
                hint='Metrics need at least a name'
            )
        name = tokens[0]
        source_kind = None
        source_name = None
        goal = None
        idx = 1
        while idx < len(tokens):
            token = tokens[idx].lower()
            if token == 'from' and idx + 2 < len(tokens):
                source_kind = tokens[idx + 1].lower()
                source_name = tokens[idx + 2]
                idx += 3
                continue
            if token == 'goal' and idx + 1 < len(tokens):
                goal = tokens[idx + 1]
                idx += 2
                continue
            break
        remainder_tokens = tokens[idx:]
        metadata: Dict[str, Any] = {}
        if remainder_tokens:
            remainder = ' '.join(remainder_tokens)
            if remainder.lower().startswith('with '):
                remainder = remainder[5:].strip()
            if remainder:
                metadata.update(self._parse_inline_assignments(remainder))
        return ExperimentMetric(
            name=name,
            source_kind=source_kind,
            source_name=source_name,
            goal=goal,
            metadata=metadata,
        )
