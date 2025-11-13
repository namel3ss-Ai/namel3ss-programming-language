from __future__ import annotations

from textwrap import dedent

INSIGHTS_SECTION = dedent(
    '''
async def fetch_dataset_rows(
    key: str,
    session: AsyncSession,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    context.setdefault("session", session)
    dataset = DATASETS.get(key)
    if not dataset:
        return []
    resolved_connector = _resolve_connector(dataset, context)
    scope, cache_enabled, ttl = _dataset_cache_settings(dataset, context)
    cache_key = _make_dataset_cache_key(key, scope, context)
    if cache_enabled:
        DATASET_CACHE_INDEX[key].add(cache_key)
    cached = await _cache_get(cache_key) if cache_enabled else None
    if isinstance(cached, list) and cached:
        return _clone_rows(cached)
    source_rows = await _load_dataset_source(dataset, resolved_connector, session, context)
    rows = await _execute_dataset_pipeline(dataset, source_rows, context)
    if cache_enabled:
        await _cache_set(cache_key, rows, ttl)
    else:
        context[cache_key] = _clone_rows(rows)
    slug = context.get("page") if isinstance(context.get("page"), str) else None
    await _broadcast_dataset_refresh(slug, key, rows)
    if dataset.get("refresh_policy"):
        await _schedule_dataset_refresh(key, dataset, session, context)
    return rows


async def _execute_sql(session: Optional[AsyncSession], query: Any) -> Any:
    if session is None:
        raise RuntimeError("SQL execution requires a database session")
    if isinstance(session, AsyncSession):
        return await session.execute(query)
    if Session is not None and isinstance(session, Session):
        return await run_in_threadpool(session.execute, query)
    executor = getattr(session, "execute", None)
    if executor is None:
        raise RuntimeError("Session does not expose an execute method")
    return await run_in_threadpool(executor, query)


def compile_dataset_to_sql(
    dataset: Any,
    metadata: Any,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile a :class:`Dataset` AST into a parameterised SQL statement.

    The compiler aims to be dialect-neutral and conservatively binds every
    literal via ``:pN`` style placeholders. When a feature cannot be expressed
    safely the function records a warning and falls back to a harmless default
    so that the generated SQL remains executable.
    """

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
    else:
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

    def _resolve_context_value(node: ContextValue) -> Tuple[str, Set[str], bool]:
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
        if isinstance(node, str):
            _note("warning", f"unsupported expression string '{node}' → fallback")
            return _fallback_sql(purpose), set(), True
        if node is None:
            return _fallback_sql(purpose), set(), True
        _note("warning", f"unsupported expression {_describe_expression(node)} → fallback")
        return _fallback_sql(purpose), set(), True

    source_type = str(_dataset_attr(dataset, "source_type", "")).lower()
    if source_type not in {"table", "sql"}:
        notes.append("error: non-sql dataset sources cannot be compiled to SQL")
        return {
            "sql": None,
            "params": {},
            "tables": [],
            "columns": [],
            "notes": notes,
            "status": "error",
            "error": "non_sql_source",
            "detail": f"Dataset source type '{source_type}' is not SQL compatible.",
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
            join_table: Optional[str] = None
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

    sql_text = "\\n".join(sql_lines)
    result = {
        "sql": sql_text if status != "error" else None,
        "params": params,
        "tables": tables,
        "columns": columns or [f"{base_alias}.*"],
        "notes": notes,
        "status": status,
    }
    if status == "error":
        detail = "SQL compilation failed"
        for note in notes:
            if note.startswith("error:"):
                detail = note.split(":", 1)[1].strip() or detail
                break
        result["error"] = "sql_compilation_failed"
        result["detail"] = detail
    return result


def evaluate_insights_for_dataset(
    name: str,
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    spec = INSIGHTS.get(name)
    if not spec:
        return {}
    return _run_insight(spec, rows, context)


async def _load_dataset_source(
    dataset: Dict[str, Any],
    connector: Dict[str, Any],
    session: AsyncSession,
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    source_type = str(dataset.get("source_type") or "table").lower()
    source_name = dataset.get("source")
    if source_type == "table":
        query = text(f"SELECT * FROM {source_name}") if source_name else None
        if query is None:
            return []
        try:
            result = await _execute_sql(session, query)
            return [dict(row) for row in result.mappings()]
        except Exception:
            logger.exception("Failed to load table dataset '%s'", source_name)
            return []
    if source_type == "sql":
        connector_name = connector.get("name") if connector else None
        driver = CONNECTOR_DRIVERS.get("sql")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("SQL connector driver '%s' failed", connector_name)
                return []
        query_text = connector.get("options", {}).get("query") if connector else None
        if not query_text:
            return []
        try:
            result = await _execute_sql(session, text(query_text))
            return [dict(row) for row in result.mappings()]
        except Exception:
            logger.exception("Failed to execute SQL query for dataset '%s'", dataset.get("name"))
            return []
    if source_type == "file":
        path = connector.get("name") if connector else source_name
        if not path:
            return []
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [dict(row) for row in reader]
        except FileNotFoundError:
            logger.warning("Dataset file '%s' not found", path)
        except Exception:
            logger.exception("Failed to load dataset file '%s'", path)
        return []
    if source_type == "rest":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = CONNECTOR_DRIVERS.get("rest")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("REST connector driver '%s' failed", connector_name)
        endpoint = connector.get("options", {}).get("endpoint") if connector else None
        if not endpoint:
            return []
        async with _HTTPX_CLIENT_CLS() as client:
            try:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
                rows = _normalize_connector_rows(payload)
                if rows:
                    return rows
            except Exception:
                logger.exception("Failed to fetch REST dataset '%s'", connector_name)
        return []
    if source_type == "graphql":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = CONNECTOR_DRIVERS.get("graphql")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("GraphQL connector driver '%s' failed", connector_name)
        return []
    if source_type == "grpc":
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = CONNECTOR_DRIVERS.get("grpc")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("gRPC connector driver '%s' failed", connector_name)
        return []
    if source_type in {"stream", "streaming"}:
        connector_name = connector.get("name") if connector else dataset.get("name")
        driver = CONNECTOR_DRIVERS.get("stream") or CONNECTOR_DRIVERS.get("streaming")
        if driver:
            try:
                return await driver(connector, context)
            except Exception:
                logger.exception("Streaming connector driver '%s' failed", connector_name)
        return []
    if source_type == "dataset" and source_name:
        target_name = str(source_name)
        if target_name == dataset.get("name"):
            return list(dataset.get("sample_rows", []))
        return await fetch_dataset_rows(target_name, session, context)
    return list(dataset.get("sample_rows", []))


async def _execute_dataset_pipeline(
    dataset: Dict[str, Any],
    rows: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    operations: List[Dict[str, Any]] = list(dataset.get("operations") or [])
    transforms: List[Dict[str, Any]] = list(dataset.get("transforms") or [])
    current_rows = _clone_rows(rows)
    aggregate_ops: List[Tuple[str, str]] = []
    group_by_columns: List[str] = []
    for operation in operations:
        otype = str(operation.get("type") or "").lower()
        if otype == "filter":
            current_rows = _apply_filter(current_rows, operation.get("condition"), context)
        elif otype == "computed_column":
            _apply_computed_column(current_rows, operation.get("name"), operation.get("expression"), context)
        elif otype == "group_by":
            group_by_columns = list(operation.get("columns") or [])
        elif otype == "aggregate":
            aggregate_ops.append((operation.get("function"), operation.get("expression")))
        elif otype == "order_by":
            current_rows = _apply_order(current_rows, operation.get("columns") or [])
        elif otype == "window":
            _apply_window_operation(
                current_rows,
                operation.get("name"),
                operation.get("function"),
                operation.get("target"),
            )
    if aggregate_ops:
        current_rows = _apply_group_aggregate(current_rows, group_by_columns, aggregate_ops, context)
    if transforms:
        current_rows = _apply_transforms(current_rows, transforms, context)
    quality_checks: List[Dict[str, Any]] = list(dataset.get("quality_checks") or [])
    evaluation = _evaluate_quality_checks(current_rows, quality_checks, context)
    if evaluation:
        context.setdefault("quality", {})[dataset.get("name")] = evaluation
    features = dataset.get("features") or []
    targets = dataset.get("targets") or []
    if features:
        context.setdefault("features", {})[dataset.get("name")] = features
    if targets:
        context.setdefault("targets", {})[dataset.get("name")] = targets
    return current_rows



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
    scope["models"] = MODEL_REGISTRY
    scope["predict"] = predict  # Future hook: replace with real inference callable
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


def evaluate_insight(slug: str, context: Optional[Dict[str, Any]] = None) -> InsightResponse:
    spec = INSIGHTS.get(slug)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Insight '{slug}' is not defined")
    ctx = dict(context or build_context(None))
    rows: List[Dict[str, Any]] = []
    result = evaluate_insights_for_dataset(slug, rows, ctx)
    dataset = result.get("dataset") or spec.get("source_dataset") or slug
    return InsightResponse(name=slug, dataset=dataset, result=result)


def run_prediction(model_name: str, payload: Optional[Dict[str, Any]] = None) -> PredictionResponse:
    model_key = str(model_name)
    if model_key not in MODELS and model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not registered")
    request_payload = dict(payload or {})
    result = predict(model_key, request_payload)
    response_payload = {
        "model": result.get("model", model_key),
        "version": result.get("version"),
        "framework": result.get("framework"),
        "input": result.get("input") or {},
        "output": result.get("output") or {},
        "explanations": result.get("explanations") or {},
        "metadata": result.get("spec_metadata") or result.get("metadata") or {},
    }
    return PredictionResponse(**response_payload)


async def predict_model(model_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience helper used by the generated API and tests."""

    response = run_prediction(model_name, payload)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


def run_experiment(slug: str, payload: Optional[Dict[str, Any]] = None) -> ExperimentResult:
    experiment_key = str(slug)
    if experiment_key not in AI_EXPERIMENTS:
        raise HTTPException(status_code=404, detail=f"Experiment '{slug}' is not defined")
    request_payload = dict(payload or {})
    result = evaluate_experiment(experiment_key, request_payload)
    return ExperimentResult(**result)


async def experiment_metrics(slug: str) -> Dict[str, Any]:
    """Return experiment metrics snapshot as a plain dictionary."""

    response = run_experiment(slug, {})
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


async def run_experiment_endpoint(
    slug: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an experiment with an optional payload and return a dict."""

    response = run_experiment(slug, payload or {})
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


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

    '''
).strip()

__all__ = ['INSIGHTS_SECTION']
