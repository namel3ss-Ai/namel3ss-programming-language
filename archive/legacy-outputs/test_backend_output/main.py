"""Auto-generated FastAPI backend scaffold for Namel3ss app."""

from __future__ import annotations

from typing import Any, Dict, List

import re

from fastapi import Depends, FastAPI
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from .database import get_session
from .schemas import ChartResponse, TableResponse

metadata = MetaData()

APP_VAR_DEFS: List[tuple[str, Dict[str, Any]]] = []
PAGE_VARIABLES: Dict[str, List[tuple[str, Dict[str, Any]]]] = {'home': [], 'feedback': [], 'admin': []}

def _escape_sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"

def _expression_to_sql(node: Dict[str, Any], context: Dict[str, Any]) -> str:
    kind = node.get('type')
    if kind == 'literal':
        return _escape_sql_literal(node.get('value'))
    if kind == 'name':
        name = node['name']
        if name in context:
            return _escape_sql_literal(context[name])
        return name
    if kind == 'attribute':
        base_name = node['base']
        attr_name = node['attr']
        if base_name in context:
            base_obj = context[base_name]
            if isinstance(base_obj, dict) and attr_name in base_obj:
                return _escape_sql_literal(base_obj[attr_name])
            if hasattr(base_obj, attr_name):
                return _escape_sql_literal(getattr(base_obj, attr_name))
        return f"{base_name}_{attr_name}"
    if kind == 'unary':
        op = node['op']
        operand = _expression_to_sql(node['operand'], context)
        if op == 'not':
            return f"NOT ({operand})"
        if op == '-':
            return f"-({operand})"
        raise ValueError(f"Unsupported unary operator: {op}")
    if kind == 'binary':
        op = node['op']
        left = _expression_to_sql(node['left'], context)
        right = _expression_to_sql(node['right'], context)
        op_map = {
            'and': 'AND',
            'or': 'OR',
            '==': '=',
            '!=': '!=',
            '>': '>',
            '<': '<',
            '>=': '>=',
            '<=': '<=',
        }
        mapped = op_map.get(op, op)
        return f"({left} {mapped} {right})"
    if kind == 'raw':
        return node.get('value', '')
    raise ValueError(f"Unsupported expression node: {kind}")

def _evaluate_expression(node: Dict[str, Any], context: Dict[str, Any]) -> Any:
    kind = node.get('type')
    if kind == 'literal':
        return node.get('value')
    if kind == 'name':
        name = node['name']
        if name not in context:
            raise KeyError(f"Name '{name}' not found in context")
        return context[name]
    if kind == 'attribute':
        base_name = node['base']
        attr_name = node['attr']
        if base_name not in context:
            raise KeyError(f"Base '{base_name}' not found in context")
        base_obj = context[base_name]
        if isinstance(base_obj, dict):
            if attr_name not in base_obj:
                raise KeyError(f"Key '{attr_name}' not found on dict '{base_name}'")
            return base_obj[attr_name]
        if hasattr(base_obj, attr_name):
            return getattr(base_obj, attr_name)
        raise AttributeError(f"Attribute '{attr_name}' not found on '{base_name}'")
    if kind == 'unary':
        op = node['op']
        operand = _evaluate_expression(node['operand'], context)
        if op == 'not':
            return not bool(operand)
        if op == '-':
            return -operand
        raise ValueError(f"Unsupported unary operator: {op}")
    if kind == 'binary':
        op = node['op']
        if op in {'and', 'or'}:
            left_val = _evaluate_expression(node['left'], context)
            if op == 'and':
                return left_val and _evaluate_expression(node['right'], context)
            return left_val or _evaluate_expression(node['right'], context)
        left = _evaluate_expression(node['left'], context)
        right = _evaluate_expression(node['right'], context)
        if op == '+':
            return left + right
        if op == '-':
            return left - right
        if op == '*':
            return left * right
        if op == '/':
            return left / right
        if op == '==':
            return left == right
        if op == '!=':
            return left != right
        if op == '>':
            return left > right
        if op == '<':
            return left < right
        if op == '>=':
            return left >= right
        if op == '<=':
            return left <= right
        raise ValueError(f"Unsupported binary operator: {op}")
    if kind == 'raw':
        return node.get('value')
    raise ValueError(f"Unsupported expression node: {kind}")

def _evaluate_app_variables() -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for name, expr in APP_VAR_DEFS:
        context[name] = _evaluate_expression(expr, context)
    return context

def _default_request_context() -> Dict[str, Any]:
    return {'user': {'name': 'Guest'}}

APP_VARS: Dict[str, Any] = _evaluate_app_variables()

def build_context(page_key: str | None = None) -> Dict[str, Any]:
    context = dict(APP_VARS)
    context.update(_default_request_context())
    if page_key and page_key in PAGE_VARIABLES:
        for name, expr in PAGE_VARIABLES[page_key]:
            context[name] = _evaluate_expression(expr, context)
    return context

def compile_dataset_to_sql(dataset: Dict[str, Any], metadata: MetaData, context: Dict[str, Any]) -> Select:
    from sqlalchemy import func, literal_column, select, text

    if dataset.get('source_type') != 'table':
        raise ValueError('Only table-backed datasets are supported')

    table_name = dataset['source']
    table_clause = metadata.tables.get(table_name) or text(table_name)

    filter_clauses: List[Any] = []
    group_columns: List[str] = []
    aggregate_specs: List[tuple[str, str]] = []
    computed_specs: List[tuple[str, str]] = []

    for op in dataset.get('operations', []):
        op_type = op.get('type')
        if op_type == 'filter':
            condition = op.get('condition')
            if not condition:
                continue
            if isinstance(condition, dict) and condition.get('type') == 'raw':
                filter_clauses.append(text(condition.get('value', '')))
            else:
                sql = _expression_to_sql(condition, context)
                filter_clauses.append(text(sql))
        elif op_type == 'group_by':
            group_columns.extend(op.get('columns', []))
        elif op_type == 'aggregate':
            aggregate_specs.append((op['function'], op['expression']))
        elif op_type == 'computed_column':
            expression = op.get('expression')
            if expression:
                sql = _expression_to_sql(expression, context)
                computed_specs.append((op['name'], sql))
        else:
            raise ValueError(f'Unsupported dataset operation: {op_type}')

    select_columns: List[Any] = []
    if aggregate_specs:
        compiled_aggs: List[Any] = []
        for func_name, expr in aggregate_specs:
            expression = expr
            alias = None
            if re.search(r"\s+as\s+", expr, flags=re.IGNORECASE):
                parts = re.split(r"\s+as\s+", expr, maxsplit=1, flags=re.IGNORECASE)
                expression = parts[0].strip()
                alias = parts[1].strip()
            column = literal_column(expression)
            func_expr = getattr(func, func_name.lower())(column)
            if alias:
                func_expr = func_expr.label(alias)
            compiled_aggs.append(func_expr)
        if group_columns:
            select_columns.extend(literal_column(col) for col in group_columns)
        select_columns.extend(compiled_aggs)
    elif group_columns:
        select_columns.extend(literal_column(col) for col in group_columns)
    else:
        select_columns.append(literal_column('*'))

    for name, sql in computed_specs:
        select_columns.append(text(sql).label(name))

    query = select(*select_columns).select_from(table_clause)

    for clause in filter_clauses:
        query = query.where(clause)

    if group_columns:
        query = query.group_by(*(literal_column(col) for col in group_columns))

    return query

DATASETS: Dict[str, Dict[str, Any]] = {
    'admin_table_0': {'name': 'orders', 'source_type': 'table', 'source': 'orders', 'operations': [], 'context_page': 'admin'},
    'home_table_0': {'name': 'orders', 'source_type': 'table', 'source': 'orders', 'operations': [{'type': 'filter', 'condition': {'type': 'raw', 'value': 'status == "Pending"'}}], 'context_page': 'home'},
    'monthly_sales': {'name': 'monthly_sales', 'source_type': 'table', 'source': 'sales', 'operations': [{'type': 'filter', 'condition': {'type': 'binary', 'op': '==', 'left': {'type': 'name', 'name': 'region'}, 'right': {'type': 'literal', 'value': 'EU'}}}, {'type': 'group_by', 'columns': ['month']}, {'type': 'aggregate', 'function': 'sum', 'expression': 'revenue as total_revenue'}], 'context_page': 'home'},
}

app = FastAPI(title='CoffeeHub')

@app.get('/')
async def page_home_0() -> Dict[str, Any]:
    context = build_context('home')
    return {
        "page": 'Home',
        "route": '/',
        "components": [{'type': 'text', 'text': 'Welcome to CoffeeHub!'}, {'type': 'chart', 'title': 'Revenue Growth', 'endpoint': '/api/pages/home/charts/0', 'chart_type': 'line'}, {'type': 'table', 'title': 'Recent Orders', 'endpoint': '/api/pages/home/tables/0', 'source': {'type': 'table', 'name': 'orders'}}],
        "context": context,
    }
@app.get('/feedback')
async def page_feedback_1() -> Dict[str, Any]:
    context = build_context('feedback')
    return {
        "page": 'Feedback',
        "route": '/feedback',
        "components": [{'type': 'form', 'title': 'Submit Feedback', 'endpoint': '/api/pages/feedback/forms/0', 'fields': ['name', 'email', 'message']}],
        "context": context,
    }
@app.get('/admin')
async def page_admin_2() -> Dict[str, Any]:
    context = build_context('admin')
    return {
        "page": 'Admin',
        "route": '/admin',
        "components": [{'type': 'table', 'title': 'Orders', 'endpoint': '/api/pages/admin/tables/0', 'source': {'type': 'table', 'name': 'orders'}}, {'type': 'action', 'title': 'Approve Order', 'endpoint': '/api/pages/admin/actions/0', 'trigger': 'user clicks "Approve"'}],
        "context": context,
    }

@app.get('/api/pages/home/charts/0', response_model=ChartResponse)
async def home_chart_0(session: AsyncSession = Depends(get_session)) -> ChartResponse:
    labels: List[Any] = []
    series_values: List[Any] = []
    try:
        dataset = DATASETS['monthly_sales']
        context = build_context(dataset.get('context_page'))
        query = compile_dataset_to_sql(dataset, metadata, context)
        result = await session.execute(query)
        rows = [dict(row) for row in result.mappings().all()]
        x_key = 'month'
        y_key = 'total_revenue'
        if rows:
            if x_key:
                labels = [row.get(x_key) for row in rows]
            if y_key:
                series_values = [row.get(y_key) for row in rows]
    except Exception:
        rows = []
        if not labels and {chart.x!r}:
            labels = list(range(1, 6))
        if not series_values and {chart.y!r}:
            series_values = [idx * 10 for idx in range(1, 6)]
    if not labels:
        labels = list(range(1, len(series_values) + 1)) if series_values else []
    if not series_values:
        series_values = list(range(1, len(labels) + 1)) if labels else []
    series = [{'label': 'Revenue Growth', 'data': series_values}] if series_values else []
    return ChartResponse(
        title='Revenue Growth',
        source={'type': 'dataset', 'name': 'monthly_sales'},
        chart_type='line',
        x='month',
        y='total_revenue',
        color='var(--primary)',
        labels=labels,
        series=series,
    )
@app.get('/api/pages/home/tables/0', response_model=TableResponse)
async def home_table_0(session: AsyncSession = Depends(get_session)) -> TableResponse:
    try:
        dataset = DATASETS['home_table_0']
        context = build_context(dataset.get('context_page'))
        query = compile_dataset_to_sql(dataset, metadata, context)
        result = await session.execute(query)
        rows = [dict(row) for row in result.mappings().all()]
    except Exception:
        rows = [{'id': 'id_1', 'customer_name': 'customer_name_1', 'total': 'total_1', 'status': 'status_1'}, {'id': 'id_2', 'customer_name': 'customer_name_2', 'total': 'total_2', 'status': 'status_2'}, {'id': 'id_3', 'customer_name': 'customer_name_3', 'total': 'total_3', 'status': 'status_3'}]
    return TableResponse(
        title='Recent Orders',
        source={'type': 'table', 'name': 'orders'},
        columns=['id', 'customer_name', 'total', 'status'],
        filter='status == "Pending"',
        sort=None,
        rows=rows,
    )
@app.post('/api/pages/feedback/forms/0')
async def feedback_form_0(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": 'Submit Feedback',
        "payload": payload,
        "operations": ['toast: Thank you for your feedback!'],
    }
@app.get('/api/pages/admin/tables/0', response_model=TableResponse)
async def admin_table_0(session: AsyncSession = Depends(get_session)) -> TableResponse:
    try:
        dataset = DATASETS['admin_table_0']
        context = build_context(dataset.get('context_page'))
        query = compile_dataset_to_sql(dataset, metadata, context)
        result = await session.execute(query)
        rows = [dict(row) for row in result.mappings().all()]
    except Exception:
        rows = [{'id': 'id_1', 'customer_name': 'customer_name_1', 'total': 'total_1', 'status': 'status_1'}, {'id': 'id_2', 'customer_name': 'customer_name_2', 'total': 'total_2', 'status': 'status_2'}, {'id': 'id_3', 'customer_name': 'customer_name_3', 'total': 'total_3', 'status': 'status_3'}]
    return TableResponse(
        title='Orders',
        source={'type': 'table', 'name': 'orders'},
        columns=['id', 'customer_name', 'total', 'status'],
        filter=None,
        sort=None,
        rows=rows,
    )
@app.post('/api/pages/admin/actions/0')
async def admin_action_0() -> Dict[str, Any]:
    return {
        "action": 'Approve Order',
        "trigger": 'user clicks "Approve"',
        "operations": ['update orders set status = "APPROVED"', 'toast: Order approved', 'navigate: Admin'],
    }