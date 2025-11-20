"""Evaluation (evaluators, metrics, guardrails) encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from .expressions import _encode_value

if TYPE_CHECKING:
    from ....ast import Evaluator, EvalMetricSpec, EvalSuiteDefinition, Guardrail, Metric


def _encode_evaluator(evaluator: "Evaluator", env_keys: set) -> Dict[str, Any]:
    """Encode an evaluator definition."""
    config_encoded = _encode_value(evaluator.config, env_keys)
    config_payload = config_encoded if isinstance(config_encoded, dict) else {}
    return {
        "name": evaluator.name,
        "kind": evaluator.kind,
        "provider": evaluator.provider,
        "config": config_payload,
    }


def _encode_metric(metric: "Metric") -> Dict[str, Any]:
    """Encode a metric definition."""
    return {
        "name": metric.name,
        "evaluator": metric.evaluator,
        "aggregation": metric.aggregation,
        "params": dict(metric.params or {}),
    }


def _encode_guardrail(guardrail: "Guardrail") -> Dict[str, Any]:
    """Encode a guardrail definition."""
    return {
        "name": guardrail.name,
        "evaluators": list(guardrail.evaluators),
        "action": guardrail.action,
        "message": guardrail.message,
    }


def _encode_eval_suite(suite: "EvalSuiteDefinition") -> Dict[str, Any]:
    """Encode an eval_suite definition for backend runtime."""
    return {
        "name": suite.name,
        "dataset_name": suite.dataset_name,
        "target_chain_name": suite.target_chain_name,
        "metrics": [_encode_eval_metric_spec(spec) for spec in suite.metrics],
        "judge_llm_name": suite.judge_llm_name,
        "rubric": suite.rubric,
        "description": suite.description,
        "metadata": dict(suite.metadata or {}),
    }


def _encode_eval_metric_spec(spec: "EvalMetricSpec") -> Dict[str, Any]:
    """Encode an eval metric specification."""
    return {
        "name": spec.name,
        "type": spec.type,
        "config": dict(spec.config or {}),
    }
