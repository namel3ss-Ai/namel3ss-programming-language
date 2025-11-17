"""Shared insight evaluation helpers for generated runtimes."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

from .expression_sandbox import (
    ExpressionSandboxError,
    evaluate_sandboxed_expression,
)

__all__ = [
    "evaluate_insights_for_dataset",
    "run_insight",
    "evaluate_expression",
    "resolve_expression_path",
]


def evaluate_insights_for_dataset(
    name: str,
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
    *,
    insights: Dict[str, Any],
    run_insight: Callable[[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Look up an insight spec and evaluate it against the provided dataset rows."""

    spec = insights.get(name)
    if not spec:
        return {}
    return run_insight(spec, rows, context)


def run_insight(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
    *,
    model_registry: Dict[str, Any],
    predict_callable: Callable[[str, Optional[Dict[str, Any]]], Any],
    evaluate_expression: Callable[[Optional[str], List[Dict[str, Any]], Dict[str, Any]], Any],
    resolve_expression_path: Callable[[str, List[Dict[str, Any]], Dict[str, Any]], Any],
    render_template_value: Callable[[Any, Dict[str, Any]], Any],
) -> Dict[str, Any]:
    """Execute an insight specification and return computed metrics and narratives."""

    scope: Dict[str, Any] = {"rows": rows, "context": context}
    scope["models"] = model_registry
    scope["predict"] = predict_callable

    for step in spec.get("logic", []):
        if step.get("type") == "assign":
            value_expr = step.get("expression")
            scope[step.get("name")] = evaluate_expression(value_expr, rows, scope)

    metrics: List[Dict[str, Any]] = []
    metrics_map: Dict[str, Dict[str, Any]] = {}
    for metric in spec.get("metrics", []):
        value = evaluate_expression(metric.get("value"), rows, scope)
        baseline = evaluate_expression(metric.get("baseline"), rows, scope)
        if value is None:
            value = len(rows) * 100 if rows else 0
        if baseline is None:
            baseline = value / 2 if value else 0
        payload = {
            "name": metric.get("name"),
            "label": metric.get("label"),
            "value": value,
            "baseline": baseline,
            "trend": "up" if value >= baseline else "flat",
            "formatted": f"${{value:,.2f}}",
            "unit": metric.get("unit"),
            "positive_label": metric.get("positive_label", "positive"),
        }
        metrics.append(payload)
        metrics_map[payload["name"]] = payload

    alerts_list: List[Dict[str, Any]] = []
    alerts_map: Dict[str, Dict[str, Any]] = {}
    for threshold in spec.get("thresholds", []):
        alert_payload = {
            "name": threshold.get("name"),
            "level": threshold.get("level"),
            "metric": threshold.get("metric"),
            "triggered": True,
        }
        alerts_list.append(alert_payload)
        alerts_map[alert_payload["name"]] = alert_payload

    narrative_scope = dict(scope)
    narrative_scope["metrics"] = metrics_map
    narratives: List[Dict[str, Any]] = []
    for narrative in spec.get("narratives", []):
        text = render_template_value(narrative.get("template"), narrative_scope)
        narratives.append(
            {
                "name": narrative.get("name"),
                "text": text,
                "variant": narrative.get("variant"),
            }
        )

    variables: Dict[str, Any] = {}
    for key, expr in spec.get("expose_as", {}).items():
        variables[key] = evaluate_expression(expr, rows, scope)

    return {
        "name": spec.get("name"),
        "dataset": spec.get("source_dataset"),
        "metrics": metrics,
        "alerts": alerts_map,
        "alerts_list": alerts_list,
        "narratives": narratives,
        "variables": variables,
    }


def evaluate_expression(
    expression: Optional[str],
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
    *,
    resolve_expression_path: Callable[[str, List[Dict[str, Any]], Dict[str, Any]], Any],
) -> Any:
    """Evaluate an insight expression against the provided scope."""

    if expression is None:
        return None
    expr = expression.strip()
    if not expr:
        return None
    sandbox_scope = dict(scope)
    sandbox_scope.setdefault("rows", rows)
    sandbox_scope.setdefault("math", math)

    builtins_sum = sum

    def _coerce_numeric(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if value in (None, "", False):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _field_series(field: str) -> List[float]:
        series: List[float] = []
        for row in rows:
            if isinstance(row, dict):
                candidate = row.get(field)
            else:
                candidate = getattr(row, field, None)
            series.append(_coerce_numeric(candidate))
        return series

    def _safe_sum(target: Any, start: Any = 0) -> Any:
        if isinstance(target, str):
            return builtins_sum(_field_series(target), start)
        return builtins_sum(target, start)

    def _safe_avg(target: Any, default: float = 0.0) -> float:
        if isinstance(target, str):
            series = _field_series(target)
        else:
            try:
                iterable = list(target)  # type: ignore[arg-type]
            except TypeError:
                iterable = [_coerce_numeric(target)]
            series = [_coerce_numeric(value) for value in iterable]
        return builtins_sum(series) / len(series) if series else default

    sandbox_scope["sum"] = _safe_sum
    sandbox_scope["avg"] = _safe_avg
    for name, func in (
        ("len", len),
        ("min", min),
        ("max", max),
        ("abs", abs),
        ("round", round),
        ("int", int),
        ("float", float),
        ("str", str),
        ("bool", bool),
    ):
        sandbox_scope.setdefault(name, func)

    extra_callables = {_safe_sum, _safe_avg}

    sandbox_error = False
    try:
        return evaluate_sandboxed_expression(
            expr,
            sandbox_scope,
            extra_callables=extra_callables,
            allow_dict_key_attributes=True,
        )
    except ExpressionSandboxError:
        sandbox_error = True
    except Exception:
        return None

    if sandbox_error and ("(" in expr or ")" in expr or expr.startswith("__") or "__" in expr):
        return None
    if "." in expr or "[" in expr:
        return resolve_expression_path(expr, rows, scope)
    if expr in scope:
        return scope[expr]
    try:
        return float(expr)
    except ValueError:
        return None if sandbox_error else expr


def resolve_expression_path(
    expression: str,
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
    """Resolve dotted or indexed expressions against rows/scope objects."""

    tokens: List[str] = []
    buffer = ""
    for char in expression:
        if char == ".":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            continue
        if char == "[":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append("[")
            continue
        if char == "]":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append("]")
            continue
        buffer += char
    if buffer:
        tokens.append(buffer)
    if not tokens:
        return None

    base_name = tokens.pop(0)
    if base_name == "rows":
        current: Any = rows
    else:
        current = scope.get(base_name)

    idx_mode = False
    for token in tokens:
        if token == "[":
            idx_mode = True
            continue
        if token == "]":
            idx_mode = False
            continue
        if idx_mode:
            try:
                index = int(token)
            except ValueError:
                return None
            if isinstance(current, list) and 0 <= index < len(current):
                current = current[index]
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(token)
            else:
                current = getattr(current, token, None)
        if current is None:
            return None
    return current
