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
PAGE_VARIABLES: Dict[str, List[tuple[str, Dict[str, Any]]]] = {'home': [], 'admin': []}

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
    'active_users': {'name': 'active_users', 'source_type': 'table', 'source': 'users', 'operations': [{'type': 'filter', 'condition': {'type': 'binary', 'op': '==', 'left': {'type': 'name', 'name': 'status'}, 'right': {'type': 'literal', 'value': 'active'}}}], 'context_page': 'home'},
}

app = FastAPI(title='Demo API')

@app.get('/')
async def page_home_0() -> Dict[str, Any]:
    context = build_context('home')
    return {
        "page": 'Home',
        "route": '/',
        "components": [{'type': 'text', 'text': 'Welcome to Demo API'}, {'type': 'table', 'title': 'Active Users', 'endpoint': '/api/pages/home/tables/0', 'source': {'type': 'dataset', 'name': 'active_users'}}],
        "context": context,
    }
@app.get('/admin')
async def page_admin_1() -> Dict[str, Any]:
    context = build_context('admin')
    return {
        "page": 'Admin',
        "route": '/admin',
        "components": [{'type': 'unknown'}],
        "context": context,
    }

@app.get('/api/pages/home/tables/0', response_model=TableResponse)
async def home_table_0(session: AsyncSession = Depends(get_session)) -> TableResponse:
    try:
        dataset = DATASETS['active_users']
        context = build_context(dataset.get('context_page'))
        query = compile_dataset_to_sql(dataset, metadata, context)
        result = await session.execute(query)
        rows = [dict(row) for row in result.mappings().all()]
    except Exception:
        rows = [{'value': 1}, {'value': 2}, {'value': 3}]
    return TableResponse(
        title='Active Users',
        source={'type': 'dataset', 'name': 'active_users'},
        columns=[],
        filter=None,
        sort=None,
        rows=rows,
    )