from __future__ import annotations

import re
from typing import Any, Dict, List

from namel3ss.ast import Evaluator, Metric, Guardrail

from .base import ParserBase, N3SyntaxError


class EvaluationParserMixin(ParserBase):
    """Parser helpers for evaluator, metric, and guardrail declarations."""

    def _parse_evaluator(self, line: str, line_no: int, base_indent: int) -> Evaluator:
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'evaluator\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error('Expected: evaluator "Name":', line_no, line)
        name = match.group(1)
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
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'metric\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error('Expected: metric "Name":', line_no, line)
        name = match.group(1)
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
        stripped = line.strip()
        if stripped.endswith(":"):
            stripped = stripped[:-1]
        match = re.match(r'guardrail\s+"([^"]+)"', stripped, flags=re.IGNORECASE)
        if not match:
            raise self._error('Expected: guardrail "Name":', line_no, line)
        name = match.group(1)
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
