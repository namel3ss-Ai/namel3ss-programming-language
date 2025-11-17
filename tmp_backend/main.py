"""Generated FastAPI backend for Namel3ss (N3)."""

from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import httpx
from fastapi import Depends, FastAPI
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from .database import get_session
from .schemas import (
    ActionResponse,
    ChartResponse,
    FormResponse,
    InsightResponse,
    TableResponse,
)

logger = logging.getLogger(__name__)
_HTTPX_CLIENT_CLS = httpx.AsyncClient
CONTEXT_MARKER_KEY = "__context__"

class ContextRegistry:
    """Simple registry for runtime context values."""

    def __init__(self) -> None:
        self._global: Dict[str, Any] = {}
        self._pages: Dict[str, Dict[str, Any]] = {}

    def set_global(self, values: Dict[str, Any]) -> None:
        self._global = dict(values)

    def set_page(self, slug: str, values: Dict[str, Any]) -> None:
        self._pages[slug] = dict(values)

    def build(self, slug: Optional[str]) -> Dict[str, Any]:
        base = dict(self._global)
        if slug and slug in self._pages:
            base.update(self._pages[slug])
        return base


CONTEXT = ContextRegistry()

APP: Dict[str, Any] = {'name': 'RuntimeEnv', 'database': None, 'theme': {}, 'variables': []}
DATASETS: Dict[str, Dict[str, Any]] = {'orders': {'name': 'orders',
            'source_type': 'rest',
            'source': 'https://api.example.com/orders',
            'operations': [],
            'connector': {'type': 'rest',
                          'name': 'ORDERS_API',
                          'options': {'endpoint': 'https://api.example.com/orders',
                                      'headers': {'Authorization': {'__context__': {'scope': 'env',
                                                                                    'path': ['NAMEL3SS_ORDERS_TOKEN']}}},
                                      'params': {'api_key': '${NAMEL3SS_ORDERS_TOKEN}',
                                                 'user_id': {'__context__': {'scope': 'ctx',
                                                                             'path': ['user',
                                                                                      'id']}}}}},
            'reactive': False,
            'refresh_policy': None,
            'cache_policy': None,
            'pagination': None,
            'streaming': None,
            'sample_rows': [{'id': 1, 'value': 10},
                            {'id': 2, 'value': 20},
                            {'id': 3, 'value': 30}]}}
CONNECTORS: Dict[str, Dict[str, Any]] = {'orders': {'type': 'rest',
            'name': 'ORDERS_API',
            'options': {'endpoint': 'https://api.example.com/orders',
                        'headers': {'Authorization': {'__context__': {'scope': 'env',
                                                                      'path': ['NAMEL3SS_ORDERS_TOKEN']}}},
                        'params': {'api_key': '${NAMEL3SS_ORDERS_TOKEN}',
                                   'user_id': {'__context__': {'scope': 'ctx',
                                                               'path': ['user', 'id']}}}}}}
INSIGHTS: Dict[str, Dict[str, Any]] = {}
MODELS: Dict[str, Dict[str, Any]] = {}
PAGES: List[Dict[str, Any]] = [{'name': 'Dashboard',
  'route': '/',
  'slug': 'root',
  'index': 0,
  'api_path': '/api/pages/root',
  'layout': {},
  'components': [{'type': 'text',
                  'payload': {'text': 'Welcome {ctx:user.name}! Token {env:NAMEL3SS_ORDERS_TOKEN}',
                              'styles': {}}},
                 {'type': 'table',
                  'payload': {'title': 'Orders',
                              'source_type': 'dataset',
                              'source': 'orders',
                              'columns': [],
                              'filter': None,
                              'sort': None,
                              'style': {},
                              'layout': None,
                              'insight': None,
                              'dynamic_columns': None}}]}]
ENV_KEYS: List[str] = ['NAMEL3SS_ORDERS_TOKEN']
EMBED_INSIGHTS: bool = False

app = FastAPI(title=APP.get('name', 'Namel3ss App'))



def build_context(page_slug: Optional[str]) -> Dict[str, Any]:
    base = CONTEXT.build(page_slug)
    context: Dict[str, Any] = dict(base)
    context.setdefault("vars", {})
    for variable in APP.get("variables", []):
        context["vars"][variable["name"]] = _resolve_placeholders(
            variable.get("value"), context
        )
    env_values = {name: os.getenv(name) for name in ENV_KEYS}
    context.setdefault("env", {}).update(env_values)
    if page_slug:
        context.setdefault("page", page_slug)
    context.setdefault("app", APP)
    return context


def _resolve_placeholders(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if CONTEXT_MARKER_KEY in value:
            marker = value[CONTEXT_MARKER_KEY]
            scope = marker.get("scope")
            path = marker.get("path", [])
            default = marker.get("default")
            return _resolve_context_scope(scope, path, context, default)
        return {key: _resolve_placeholders(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(item, context) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name, value)
    return value


def _resolve_context_scope(
    scope: Optional[str],
    path: Iterable[str],
    context: Dict[str, Any],
    default: Any = None,
) -> Any:
    parts = list(path)
    if scope in (None, "ctx", "context"):
        return _resolve_context_path(context, parts, default)
    if scope == "env":
        if not parts:
            return default
        return os.getenv(parts[0], default)
    if scope == "vars":
        return _resolve_context_path(context.get("vars", {}), parts, default)
    target = context.get(scope)
    return _resolve_context_path(target, parts, default)


def _resolve_context_path(
    context: Optional[Dict[str, Any]],
    path: Iterable[str],
    default: Any = None,
) -> Any:
    current: Any = context
    for segment in path:
        if isinstance(current, dict):
            current = current.get(segment)
        else:
            current = None
        if current is None:
            return default
    return current


TEMPLATE_PATTERN = re.compile(r"{([^{}]+)}")


def _render_template_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            token = match.group(1).strip()
            if not token:
                return match.group(0)
            if token.startswith("$"):
                return os.getenv(token[1:], "")
            if ":" in token:
                scope, _, path = token.partition(":")
                parts = [segment for segment in path.split(".") if segment]
                resolved = _resolve_context_scope(scope, parts, context, "")
                return "" if resolved is None else str(resolved)
            value = _resolve_context_path(context.get("vars"), [token], None)
            if value is not None:
                return str(value)
            resolved = _resolve_context_path(context, [token], "")
            return "" if resolved is None else str(resolved)
        return TEMPLATE_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {
            key: _render_template_value(item, context) for key, item in value.items()
        }
    return value


async def fetch_dataset_rows(
    key: str,
    session: AsyncSession,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dataset = DATASETS.get(key)
    if not dataset:
        return []
    _resolve_connector(dataset, context)
    return list(dataset.get("sample_rows", []))


def compile_dataset_to_sql(
    dataset: Dict[str, Any],
    metadata: MetaData,
    context: Dict[str, Any],
) -> Select:
    raise NotImplementedError(
        "compile_dataset_to_sql is a stub in the generated backend"
    )


def evaluate_insights_for_dataset(
    name: str,
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    spec = INSIGHTS.get(name)
    if not spec:
        return {}
    return _run_insight(spec, rows, context)


def _resolve_connector(dataset: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    connector = dataset.get("connector")
    if not connector:
        return {}
    resolved = copy.deepcopy(connector)
    resolved["options"] = _resolve_placeholders(connector.get("options"), context)
    return resolved


def _run_insight(
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    scope: Dict[str, Any] = {"rows": rows, "context": context}
    for step in spec.get("logic", []):
        if step.get("type") == "assign":
            value_expr = step.get("expression")
            scope[step.get("name")] = _evaluate_expression(value_expr, rows, scope)
    metrics: List[Dict[str, Any]] = []
    metrics_map: Dict[str, Dict[str, Any]] = {}
    for metric in spec.get("metrics", []):
        value = _evaluate_expression(metric.get("value"), rows, scope)
        baseline = _evaluate_expression(metric.get("baseline"), rows, scope)
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
        text = _render_template_value(narrative.get("template"), narrative_scope)
        narratives.append(
            {
                "name": narrative.get("name"),
                "text": text,
                "variant": narrative.get("variant"),
            }
        )
    variables: Dict[str, Any] = {}
    for key, expr in spec.get("expose_as", {}).items():
        variables[key] = _evaluate_expression(expr, rows, scope)
    return {
        "name": spec.get("name"),
        "dataset": spec.get("source_dataset"),
        "metrics": metrics,
        "alerts": alerts_map,
        "alerts_list": alerts_list,
        "narratives": narratives,
        "variables": variables,
    }


def _evaluate_expression(
    expression: Optional[str],
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
    if expression is None:
        return None
    expr = expression.strip()
    if not expr:
        return None
    if expr == "rows":
        return rows
    if expr.startswith("len(") and expr.endswith(")"):
        target = expr[4:-1]
        value = scope.get(target)
        if value is None and target == "rows":
            value = rows
        if isinstance(value, (list, tuple)):
            return len(value)
        return 0
    if expr.startswith("sum(") and expr.endswith(")"):
        field = expr[4:-1].strip().strip('"').strip("'")
        return sum(float(row.get(field, 0) or 0) for row in rows)
    if expr.startswith("avg(") and expr.endswith(")"):
        field = expr[4:-1].strip().strip('"').strip("'")
        values = [float(row.get(field, 0) or 0) for row in rows]
        return sum(values) / len(values) if values else 0
    if "." in expr or "[" in expr:
        return _resolve_expression_path(expr, rows, scope)
    if expr in scope:
        return scope[expr]
    try:
        return float(expr)
    except ValueError:
        return expr


def _resolve_expression_path(
    expression: str,
    rows: List[Dict[str, Any]],
    scope: Dict[str, Any],
) -> Any:
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


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

async def page_root_0() -> Dict[str, Any]:
    context = build_context('root')
    components: List[Dict[str, Any]] = []
    components.append({
        'type': 'text',
        'text': _render_template_value('Welcome {ctx:user.name}! Token {env:NAMEL3SS_ORDERS_TOKEN}', context),
        'styles': {},
    })
    components.append({
        'type': 'table',
        'title': 'Orders',
        'source_type': 'dataset',
        'source': 'orders',
        'columns': [],
        'filter': None,
        'sort': None,
        'style': {},
        'layout': None,
        'insight': None,
    })
    return {
        'name': 'Dashboard',
        'route': '/',
        'components': components,
        'layout': {},
    }

@app.get('/api/pages/root')
async def page_root_0_view(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    return await page_root_0()

@app.get('/api/pages/root' + '/tables/1', response_model=TableResponse)
async def root_table_1(session: AsyncSession = Depends(get_session)) -> TableResponse:
    context = build_context('root')
    dataset = DATASETS.get('orders')
    rows = await fetch_dataset_rows('orders', session, context)
    insights: Dict[str, Any] = {}
    if EMBED_INSIGHTS and dataset.get('name') if dataset else False:
        if None:
            try:
                insights = evaluate_insights_for_dataset(None, rows, context)
            except Exception:
                logger.exception('Failed to evaluate insight %s', None)
                insights = {}
    return TableResponse(
        title='Orders',
        source={'type': 'dataset', 'name': 'orders'},
        columns=[],
        filter=None,
        sort=None,
        style={},
        insight=None,
        rows=rows,
        insights=insights,
    )
