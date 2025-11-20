"""Insight encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Set, TYPE_CHECKING

from .datasets import _encode_dataset_transform
from .expressions import _collect_template_markers, _encode_value, _expression_to_runtime, _expression_to_source

if TYPE_CHECKING:
    from ....ast import (
        Insight,
        InsightAssignment,
        InsightAudience,
        InsightDatasetRef,
        InsightDeliveryChannel,
        InsightEmit,
        InsightLogicStep,
        InsightMetric,
        InsightNarrative,
        InsightSelect,
        InsightThreshold,
    )


def _encode_insight(insight: "Insight", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an insight definition for backend state."""
    logic = [_encode_insight_step(step, env_keys) for step in insight.logic]
    metrics = [_encode_insight_metric(metric, env_keys) for metric in insight.metrics]
    thresholds = [_encode_insight_threshold(threshold, env_keys) for threshold in insight.thresholds]
    narratives = [_encode_insight_narrative(narrative, env_keys) for narrative in insight.narratives]
    expose_sources = {key: _expression_to_source(value) for key, value in insight.expose_as.items()}
    expose_exprs = {key: _expression_to_runtime(value) for key, value in insight.expose_as.items()}
    parameter_sources = {key: _expression_to_source(value) for key, value in insight.parameters.items()}
    parameter_exprs = {key: _expression_to_runtime(value) for key, value in insight.parameters.items()}
    return {
        "name": insight.name,
        "source_dataset": insight.source_dataset,
        "logic": logic,
        "metrics": metrics,
        "thresholds": thresholds,
        "narratives": narratives,
        "expose_as": expose_sources,
        "expose_expr": expose_exprs,
        "datasets": [_encode_insight_dataset_ref(ref, env_keys) for ref in insight.datasets],
        "parameters": parameter_sources,
        "parameters_expr": parameter_exprs,
        "audiences": [_encode_insight_audience(audience, env_keys) for audience in insight.audiences],
        "channels": [_encode_insight_channel(channel, env_keys) for channel in insight.channels],
        "tags": list(insight.tags or []),
        "metadata": _encode_value(insight.metadata, env_keys),
    }


def _encode_insight_step(step: "InsightLogicStep", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an insight logic step."""
    from ....ast import InsightAssignment, InsightEmit, InsightSelect
    
    if isinstance(step, InsightAssignment):
        return {
            "type": "assign",
            "name": step.name,
            "expression": _expression_to_source(step.expression),
            "expression_expr": _expression_to_runtime(step.expression),
        }
    if isinstance(step, InsightSelect):
        return {
            "type": "select",
            "dataset": step.dataset,
            "condition": _expression_to_source(step.condition),
            "condition_expr": _expression_to_runtime(step.condition),
            "limit": step.limit,
            "order_by": list(step.order_by or []),
        }
    if isinstance(step, InsightEmit):
        _collect_template_markers(step.content, env_keys)
        return {
            "type": "emit",
            "kind": step.kind,
            "content": step.content,
            "props": _encode_value(step.props, env_keys),
        }
    return {"type": type(step).__name__}


def _encode_insight_metric(metric: "InsightMetric", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an insight metric definition."""
    return {
        "name": metric.name,
        "value": _expression_to_source(metric.value),
        "value_expr": _expression_to_runtime(metric.value),
        "label": metric.label,
        "format": metric.format,
        "unit": metric.unit,
        "baseline": _expression_to_source(metric.baseline),
        "baseline_expr": _expression_to_runtime(metric.baseline),
        "target": _expression_to_source(metric.target),
        "target_expr": _expression_to_runtime(metric.target),
        "window": metric.window,
        "extras": _encode_value(metric.extras, env_keys),
    }


def _encode_insight_threshold(threshold: "InsightThreshold", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an insight threshold definition."""
    return {
        "name": threshold.name,
        "metric": threshold.metric,
        "operator": threshold.operator,
        "value": _expression_to_source(threshold.value),
        "value_expr": _expression_to_runtime(threshold.value),
        "level": threshold.level,
        "message": threshold.message,
        "window": threshold.window,
        "extras": _encode_value(threshold.extras, env_keys),
    }


def _encode_insight_narrative(narrative: "InsightNarrative", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an insight narrative template."""
    _collect_template_markers(narrative.template, env_keys)
    return {
        "name": narrative.name,
        "template": narrative.template,
        "variant": narrative.variant,
        "style": _encode_value(narrative.style, env_keys),
        "extras": _encode_value(narrative.extras, env_keys),
    }


def _encode_insight_dataset_ref(ref: "InsightDatasetRef", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an insight dataset reference."""
    return {
        "name": ref.name,
        "role": ref.role,
        "transforms": [_encode_dataset_transform(step, env_keys) for step in ref.transforms],
        "options": _encode_value(ref.options, env_keys),
    }


def _encode_insight_audience(audience: "InsightAudience", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an insight audience definition."""
    return {
        "name": audience.name,
        "persona": audience.persona,
        "needs": _encode_value(audience.needs, env_keys),
        "channels": list(audience.channels or []),
    }


def _encode_insight_channel(channel: "InsightDeliveryChannel", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an insight delivery channel."""
    return {
        "kind": channel.kind,
        "target": channel.target,
        "schedule": channel.schedule,
        "options": _encode_value(channel.options, env_keys),
    }
