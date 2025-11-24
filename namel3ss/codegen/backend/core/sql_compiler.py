"""Dataset-to-SQL compiler shared between generator and runtime."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple
import re
import logging

logger = logging.getLogger(__name__)

__all__ = ["compile_dataset_to_sql", "compile_dataset_insert", "compile_dataset_update", "compile_dataset_delete"]


def compile_dataset_to_sql(
    dataset: Any,
    metadata: Any,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile a :class:`Dataset` AST into a parameterised SQL statement."""

    try:  # Late import keeps runtime lightweight when the AST package is absent.
        from namel3ss.ast import (
            AttributeRef,
            BinaryOp,
            CallExpression,
            ComputedColumnOp,
            ContextValue,
            Dataset,
            FilterOp,
            JoinOp,
            Literal,
            NameRef,
            OrderByOp,
            PaginationPolicy,
            WindowFrame,
            WindowOp,
            UnaryOp,
        )
    except Exception:  # pragma: no cover - defensive fallback if AST unavailable
        Dataset = FilterOp = ComputedColumnOp = GroupByOp = OrderByOp = WindowOp = JoinOp = PaginationPolicy = ()  # type: ignore
        Literal = NameRef = AttributeRef = BinaryOp = CallExpression = ContextValue = UnaryOp = ()  # type: ignore
    else:  # pragma: no cover - imported lazily to avoid circulars during generation
        from namel3ss.ast.datasets import GroupByOp

    params: Dict[str, Any] = {}
    notes: List[str] = []
    tables: List[str] = []
    columns: List[str] = []
    select_expressions: List[str] = []
    where_clauses: List[str] = []
    group_by: List[str] = []
    order_by: List[str] = []
    joins: List[str] = []
    window_expressions: List[str] = []
    computed_expressions: List[str] = []
    param_index = 1

    def _note(level: str, message: str) -> None:
        notes.append(f"{level}: {message}")

    def _q(identifier: str) -> str:
        parts = [segment.strip() for segment in str(identifier or "").split(".") if segment.strip()]
        if not parts:
            return '""'
        return ".".join('"{}"'.format(part.replace('"', '""')) for part in parts)

    def _q_table(identifier: str) -> str:
        return _q(identifier)

    def _new_param(value: Any) -> str:
        nonlocal param_index
        key = f"p{param_index}"
        param_index += 1
        params[key] = value
        return f":{key}"

    def _dataset_attr(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _ensure_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return list(value)

    def _resolve_context_value(node: Any) -> Tuple[str, Set[str], bool]:
        if not isinstance(node, ContextValue):
            _note("warning", "unsupported context value expression → fallback")
            return "NULL", set(), True
        value: Any = None
        if node.scope == "env":
            resolver = getattr(context, "get_env", None)
            if callable(resolver):
                target = node.path[0] if node.path else None
                value = resolver(target) if target else None
            elif isinstance(context, dict):
                env_map = context.get("env") if isinstance(context.get("env"), dict) else context
                target = node.path[0] if node.path else None
                value = env_map.get(target) if target and isinstance(env_map, dict) else None
        else:
            resolver = getattr(context, "get_ctx", None)
            if callable(resolver):
                value = resolver(node.path)
            elif isinstance(context, dict):
                current: Any = context.get("ctx", context)
                for part in node.path:
                    if isinstance(current, dict):
                        current = current.get(part)
                    else:
                        current = getattr(current, part, None)
                    if current is None:
                        break
                value = current
        if value is None:
            value = node.default
        if value is None:
            _note("warning", "context value resolved to NULL → fallback")
            return "NULL", set(), True
        return _new_param(value), set(), False

    def _fallback_sql(purpose: str) -> str:
        return "TRUE" if purpose in {"filter", "join"} else "NULL"

    def _describe_expression(node: Any) -> str:
        if node is None:
            return "None"
        label = getattr(node, "op", None)
        if label:
            return f"{type(node).__name__}(op={label!r})"
        if isinstance(node, (Literal, NameRef)):
            return f"{type(node).__name__}"
        return type(node).__name__

    def _compile_expression(node: Any, purpose: str) -> Tuple[str, Set[str], bool]:
        if isinstance(node, Literal):
            return _new_param(node.value), set(), False
        if isinstance(node, NameRef):
            return _q(node.name), {node.name}, False
        if isinstance(node, AttributeRef):
            identifier = f"{node.base}.{node.attr}" if node.base else node.attr
            return _q(identifier), {identifier}, False
        if isinstance(node, ContextValue):
            return _resolve_context_value(node)
        if isinstance(node, UnaryOp):
            operand_sql, refs, fallback = _compile_expression(node.operand, purpose)
            op = str(node.op or "").lower()
            if fallback:
                return _fallback_sql(purpose), refs, True
            if op in {"not"}:
                return f"(NOT ({operand_sql}))", refs, False
            if op in {"-", "neg"}:
                return f"((-1) * ({operand_sql}))", refs, False
            _note("warning", f"unsupported unary operator '{node.op}' → fallback")
            return _fallback_sql(purpose), refs, True
        if isinstance(node, BinaryOp):
            left_sql, left_refs, left_fb = _compile_expression(node.left, purpose)
            right_sql, right_refs, right_fb = _compile_expression(node.right, purpose)
            op = str(node.op or "").lower()
            if op in {"==", "="}:
                op_sql = "="
            elif op in {"!=", "<>"}:
                op_sql = "<>"
            elif op in {"<", "<=", ">", ">="}:
                op_sql = op
            elif op in {"and", "or"}:
                op_sql = op.upper()
            elif op in {"+", "-", "*", "/", "%"}:
                op_sql = op
            elif op in {"like", "ilike"}:
                op_sql = op.upper()
            elif op == "in":
                if isinstance(node.right, Literal) and isinstance(node.right.value, (list, tuple, set)):
                    placeholders: List[str] = []
                    for value in node.right.value:
                        placeholders.append(_new_param(value))
                    combined_refs = left_refs.union(right_refs)
                    return f"(({left_sql}) IN ({', '.join(placeholders)}))", combined_refs, left_fb or right_fb
                _note("warning", "unsupported IN expression → fallback")
                return _fallback_sql(purpose), left_refs.union(right_refs), True
            else:
                _note("warning", f"unsupported binary operator '{node.op}' → fallback")
                return _fallback_sql(purpose), left_refs.union(right_refs), True
            combined = left_refs.union(right_refs)
            fallback = left_fb or right_fb
            if op_sql in {"AND", "OR"}:
                return f"(({left_sql}) {op_sql} ({right_sql}))", combined, fallback
            if op_sql in {"LIKE", "ILIKE"}:
                return f"(({left_sql}) {op_sql} ({right_sql}))", combined, fallback
            return f"(({left_sql}) {op_sql} ({right_sql}))", combined, fallback
        if isinstance(node, CallExpression):
            func_name = ""
            if isinstance(node.function, NameRef):
                func_name = node.function.name
            elif isinstance(node.function, AttributeRef):
                func_name = node.function.attr
            name_lower = str(func_name or "").lower()
            arg_sql: List[str] = []
            arg_refs: Set[str] = set()
            arg_fallback = False
            for argument in node.arguments:
                sql, refs, fallback = _compile_expression(argument, purpose)
                arg_sql.append(sql)
                arg_refs.update(refs)
                arg_fallback = arg_fallback or fallback
            if arg_fallback:
                return _fallback_sql(purpose), arg_refs, True
            if name_lower == "coalesce" and arg_sql:
                return f"COALESCE({', '.join(arg_sql)})", arg_refs, False
            if name_lower in {"lower", "upper", "abs"} and len(arg_sql) == 1:
                return f"{name_lower.upper()}({arg_sql[0]})", arg_refs, False
            if name_lower == "round" and arg_sql:
                return f"ROUND({', '.join(arg_sql)})", arg_refs, False
            if name_lower == "concat" and arg_sql:
                return f"({' || '.join(arg_sql)})", arg_refs, False
            if name_lower == "substring" and arg_sql:
                from_clause = arg_sql[1] if len(arg_sql) > 1 else _new_param(1)
                for_clause = arg_sql[2] if len(arg_sql) > 2 else _new_param(2147483647)
                return f"SUBSTRING({arg_sql[0]} FROM {from_clause} FOR {for_clause})", arg_refs, False
            if name_lower == "date_trunc" and arg_sql:
                precision = arg_sql[0]
                target = arg_sql[1] if len(arg_sql) > 1 else "CURRENT_TIMESTAMP"
                return f"DATE_TRUNC({precision}, {target})", arg_refs, False
            _note("warning", f"unsupported function '{func_name}' → fallback")
            return _fallback_sql(purpose), arg_refs, True
        if isinstance(node, str):  # stringified expression from encoded datasets
            _note("warning", f"unsupported expression string '{node}' → fallback")
            return _fallback_sql(purpose), set(), True
        if node is None:
            return _fallback_sql(purpose), set(), True
        _note("warning", f"unsupported expression {_describe_expression(node)} → fallback")
        return _fallback_sql(purpose), set(), True

    source_type = str(_dataset_attr(dataset, "source_type", "")).lower()
    if source_type not in {"table", "sql"}:
        return {
            "sql": None,
            "params": {},
            "tables": [],
            "columns": [],
            "notes": ["info: non-sql dataset source; skipping SQL compilation"],
            "status": "partial",
        }

    table_name = _dataset_attr(dataset, "source")
    connector = _dataset_attr(dataset, "connector")
    if source_type == "sql" and isinstance(connector, (dict,)):
        table_name = connector.get("options", {}).get("table") or table_name
    if not table_name:
        _note("error", "dataset is missing a physical table reference")
        return {
            "sql": None,
            "params": {},
            "tables": [],
            "columns": [],
            "notes": notes,
            "status": "error",
        }

    base_table = _q_table(table_name)
    base_alias = "t0"
    tables.append(base_table)

    operations = _dataset_attr(dataset, "operations", [])
    pagination = _dataset_attr(dataset, "pagination")

    join_alias_index = 1
    for operation in operations:
        if isinstance(operation, ComputedColumnOp):
            expr_sql, _, fallback = _compile_expression(operation.expression, "computed column")
            if fallback:
                expr_sql = "NULL"
            computed_expressions.append(f"{expr_sql} AS {_q(operation.name)}")
            columns.append(operation.name)
        elif isinstance(operation, FilterOp):
            expr_sql, _, _ = _compile_expression(operation.condition, "filter")
            where_clauses.append(expr_sql)
        elif isinstance(operation, GroupByOp):
            for column in _ensure_list(operation.columns):
                group_by.append(_q(column))
        elif isinstance(operation, OrderByOp):
            for entry in _ensure_list(operation.columns):
                if isinstance(entry, str):
                    parts = entry.split()
                    column_part = parts[0]
                    direction = parts[1].upper() if len(parts) > 1 else "ASC"
                elif isinstance(entry, (list, tuple)) and entry:
                    column_part = entry[0]
                    direction = str(entry[1]).upper() if len(entry) > 1 else "ASC"
                else:
                    _note("warning", "unsupported order_by entry → skipped")
                    continue
                if direction not in {"ASC", "DESC"}:
                    _note("warning", f"unsupported order direction '{direction}' → default ASC")
                    direction = "ASC"
                order_by.append(f"{_q(column_part)} {direction}")
        elif isinstance(operation, WindowOp):
            function_name = str(operation.function or "").upper()
            target_column = operation.target or "*"
            window_parts: List[str] = []
            partitions = [_q(item) for item in _ensure_list(operation.partition_by)]
            orders = [_q(item) for item in _ensure_list(operation.order_by)]
            if partitions:
                window_parts.append(f"PARTITION BY {', '.join(partitions)}")
            if orders:
                window_parts.append(f"ORDER BY {', '.join(orders)}")
            frame_clause = ""
            frame = operation.frame if isinstance(operation.frame, WindowFrame) else None
            if frame and frame.interval_value is not None:
                unit = (frame.interval_unit or "rows").lower()
                if unit in {"row", "rows"}:
                    frame_clause = f"ROWS BETWEEN {int(frame.interval_value)} PRECEDING AND CURRENT ROW"
                elif unit in {"range"}:
                    frame_clause = f"RANGE BETWEEN {int(frame.interval_value)} PRECEDING AND CURRENT ROW"
                else:
                    _note("warning", f"unsupported window frame unit '{unit}' → fallback")
            window_sql = f"{function_name}({_q(target_column)})"
            over_parts = list(window_parts)
            if frame_clause:
                over_parts.append(frame_clause)
            over = f" OVER ({' '.join(over_parts)})" if over_parts else " OVER ()"
            window_expressions.append(f"{window_sql}{over} AS {_q(operation.name)}")
            columns.append(operation.name)
        elif isinstance(operation, JoinOp):
            join_type = str(operation.join_type or "inner").lower()
            if join_type not in {"inner", "left"}:
                _note("warning", f"unsupported join type '{join_type}' → default inner")
                join_type = "inner"
            target_type = str(operation.target_type or "table").lower()
            target_name = getattr(operation, "target_name", None)
            join_table: Any = None
            if target_type == "table":
                join_table = target_name
            elif target_type == "dataset":
                mapping = {}
                if isinstance(metadata, dict):
                    mapping = metadata.get("dataset_tables") or metadata.get("datasets") or {}
                if isinstance(mapping, dict) and target_name in mapping:
                    mapped = mapping.get(target_name)
                    if isinstance(mapped, dict):
                        join_table = mapped.get("table") or mapped.get("name")
                    elif isinstance(mapped, str):
                        join_table = mapped
                if join_table is None:
                    _note("warning", f"join target dataset '{target_name}' has no table mapping → skipped")
                    continue
            else:
                _note("warning", f"unsupported join target type '{target_type}' → skipped")
                continue
            if not join_table:
                _note("warning", "join operation missing target table → skipped")
                continue
            alias = f"t{join_alias_index}"
            join_alias_index += 1
            tables.append(_q_table(join_table))
            join_keyword = "LEFT JOIN" if join_type == "left" else "INNER JOIN"
            if operation.condition is not None:
                cond_sql, _, cond_fb = _compile_expression(operation.condition, "join")
                if cond_fb:
                    _note("warning", f"unsupported join condition for '{join_table}' → skipped")
                    continue
            else:
                _note("warning", f"join '{join_table}' missing condition → skipped")
                continue
            joins.append(f"{join_keyword} {_q_table(join_table)} AS {alias} ON {cond_sql}")
        elif isinstance(operation, dict):
            op_type = str(operation.get("type") or "")
            _note("warning", f"unsupported dataset operation '{op_type}' → skipped")
        else:
            op_name = type(operation).__name__ if operation is not None else "unknown"
            _note("warning", f"unsupported dataset operation '{op_name}' → skipped")

    if not select_expressions:
        select_expressions.append(f"{base_alias}.*")
    select_expressions.extend(computed_expressions)
    select_expressions.extend(window_expressions)

    if not where_clauses:
        where_clauses.append("TRUE")

    limit_clause = ""
    offset_clause = ""
    if isinstance(pagination, PaginationPolicy) and getattr(pagination, "enabled", True):
        page_size = getattr(pagination, "page_size", None)
        if page_size is not None:
            limit_clause = f"LIMIT {_new_param(page_size)}"
            page_index = getattr(pagination, "page_index", 0)
            ctx_page_index = None
            if hasattr(context, "get"):
                ctx_page_index = context.get("page_index")
            if ctx_page_index is None and isinstance(context, dict):
                page_ctx = context.get("pagination")
                if isinstance(page_ctx, dict):
                    entry = page_ctx.get(_dataset_attr(dataset, "name"))
                    if isinstance(entry, dict):
                        ctx_page_index = entry.get("page_index")
            if ctx_page_index is not None:
                page_index = ctx_page_index
            offset_value = (page_index or 0) * page_size
            offset_clause = f"OFFSET {_new_param(offset_value)}"
        else:
            _note("info", "pagination enabled without page_size → LIMIT omitted")

    if _dataset_attr(dataset, "cache_policy"):
        _note("info", "cache policy is a runtime directive; not applied in SQL")
    if _dataset_attr(dataset, "refresh_policy"):
        _note("info", "refresh policy is a runtime directive; not applied in SQL")
    if _dataset_attr(dataset, "streaming"):
        _note("info", "streaming policy is a runtime directive; not applied in SQL")

    sql_lines = [
        f"SELECT {', '.join(select_expressions)}",
        f"FROM {base_table} AS {base_alias}",
    ]
    sql_lines.extend(joins)
    if where_clauses:
        sql_lines.append(f"WHERE {' AND '.join(where_clauses)}")
    if group_by:
        sql_lines.append(f"GROUP BY {', '.join(group_by)}")
    if order_by:
        sql_lines.append(f"ORDER BY {', '.join(order_by)}")
    if limit_clause:
        sql_lines.append(limit_clause)
    if offset_clause:
        sql_lines.append(offset_clause)

    status = "ok"
    if any(note.startswith("warning") for note in notes):
        status = "partial"
    if any(note.startswith("error") for note in notes):
        status = "error"

    return {
        "sql": "\\n".join(sql_lines),
        "params": params,
        "tables": tables,
        "columns": columns or [f"{base_alias}.*"],
        "notes": notes,
        "status": status,
    }


def compile_dataset_insert(
    dataset_name: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compile dataset insert operation to SQL.
    
    Args:
        dataset_name: Name of the dataset
        data: Data to insert
        
    Returns:
        Dictionary with 'query', 'params', and 'select_query'
    """
    table_name = _sanitize_table_name(dataset_name)
    
    # Filter out None values and prepare columns/values
    filtered_data = {k: v for k, v in data.items() if v is not None}
    
    if not filtered_data:
        raise ValueError("No valid data provided for insert")
    
    columns = [_sanitize_column_name(col) for col in filtered_data.keys()]
    placeholders = [f"%({col})s" for col in filtered_data.keys()]
    
    query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        RETURNING id
    """.strip()
    
    # Query to select the created record
    select_query = f"""
        SELECT * FROM {table_name} WHERE id = %(id)s
    """.strip()
    
    return {
        "query": query,
        "params": filtered_data,
        "select_query": select_query,
    }


def compile_dataset_update(
    dataset_name: str,
    record_id: str,
    data: Dict[str, Any],
    primary_key: str = "id",
) -> Dict[str, Any]:
    """
    Compile dataset update operation to SQL.
    
    Args:
        dataset_name: Name of the dataset
        record_id: ID of record to update
        data: Data to update
        primary_key: Primary key column name
        
    Returns:
        Dictionary with 'query', 'params', and 'select_query'
    """
    table_name = _sanitize_table_name(dataset_name)
    pk_column = _sanitize_column_name(primary_key)
    
    # Filter out None values and prepare SET clause
    filtered_data = {k: v for k, v in data.items() if v is not None}
    
    if not filtered_data:
        raise ValueError("No valid data provided for update")
    
    set_clauses = []
    for column in filtered_data.keys():
        safe_column = _sanitize_column_name(column)
        set_clauses.append(f"{safe_column} = %({column})s")
    
    # Add primary key to params
    params = dict(filtered_data)
    params[primary_key] = record_id
    
    query = f"""
        UPDATE {table_name}
        SET {', '.join(set_clauses)}
        WHERE {pk_column} = %({primary_key})s
    """.strip()
    
    # Query to select the updated record
    select_query = f"""
        SELECT * FROM {table_name} WHERE {pk_column} = %({primary_key})s
    """.strip()
    
    return {
        "query": query,
        "params": params,
        "select_query": select_query,
    }


def compile_dataset_delete(
    dataset_name: str,
    record_id: str,
    primary_key: str = "id",
) -> Dict[str, Any]:
    """
    Compile dataset delete operation to SQL.
    
    Args:
        dataset_name: Name of the dataset
        record_id: ID of record to delete
        primary_key: Primary key column name
        
    Returns:
        Dictionary with 'query' and 'params'
    """
    table_name = _sanitize_table_name(dataset_name)
    pk_column = _sanitize_column_name(primary_key)
    
    query = f"""
        DELETE FROM {table_name}
        WHERE {pk_column} = %({primary_key})s
    """.strip()
    
    return {
        "query": query,
        "params": {primary_key: record_id},
    }


def _sanitize_table_name(name: str) -> str:
    """Sanitize table name to prevent SQL injection."""
    # Remove any non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    # Ensure it starts with a letter
    if not sanitized or not sanitized[0].isalpha():
        raise ValueError(f"Invalid table name: {name}")
    
    return sanitized


def _sanitize_column_name(name: str) -> str:
    """Sanitize column name to prevent SQL injection."""
    # Remove any non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    # Ensure it starts with a letter or underscore
    if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == '_'):
        raise ValueError(f"Invalid column name: {name}")
    
    return sanitized
