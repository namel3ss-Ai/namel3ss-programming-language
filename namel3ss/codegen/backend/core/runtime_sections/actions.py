from __future__ import annotations

from textwrap import dedent

ACTIONS_SECTION = dedent(
    '''
async def _execute_action_operation(
    operation: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
) -> Optional[Dict[str, Any]]:
    otype = str(operation.get("type") or "").lower()
    result: Optional[Dict[str, Any]] = None
    if otype == "toast":
        message = _render_template_value(operation.get("message"), context)
        result = {"type": "toast", "message": message}
    elif otype == "python_call":
        module = operation.get("module")
        method = operation.get("method")
        arguments_raw = operation.get("arguments") or {}
        arguments = _resolve_placeholders(arguments_raw, context)
        payload = call_python_model(module, method, arguments)
        result = {"type": "python_call", "result": payload}
    elif otype == "connector_call":
        name = operation.get("name")
        arguments_raw = operation.get("arguments") or {}
        arguments = _resolve_placeholders(arguments_raw, context)
        payload = call_llm_connector(name, arguments)
        result = {"type": "connector_call", "result": payload}
    elif otype == "prompt_call":
        name = operation.get("prompt") or operation.get("name")
        arguments_raw = operation.get("arguments") or {}
        arguments = _resolve_placeholders(arguments_raw, context)
        payload = run_prompt(name, arguments, context=context)
        result = {"type": "prompt_call", "result": payload}
    elif otype == "chain_run":
        name = operation.get("name")
        inputs_raw = operation.get("inputs") or {}
        inputs = _resolve_placeholders(inputs_raw, context)
        payload = run_chain(name, inputs, context=context)
        result = {"type": "chain_run", "result": payload}
    elif otype == "navigate":
        target_page = operation.get("page_name") or operation.get("page") or operation.get("target")
        resolved = _resolve_page_reference(target_page)
        if resolved is None:
            result = {
                "type": "navigate",
                "status": "not_found",
                "target": target_page,
            }
        else:
            result = {
                "type": "navigate",
                "status": "ok",
                "page": resolved.get("name"),
                "page_slug": resolved.get("slug"),
                "page_route": resolved.get("route"),
            }
    elif otype == "update":
        table = operation.get("table")
        set_expression = operation.get("set_expression")
        where_expression = operation.get("where_expression")
        session: Optional[AsyncSession] = context.get("session")
        if session is None:
            result = {"type": "update", "status": "no_session"}
        else:
            updated = await _execute_update(table, set_expression, where_expression, session, context)
            result = {"type": "update", "status": "ok", "rows": updated}
    else:
        return None

    if result is not None:
        await _handle_post_action_effects(operation, context, result, otype)
    return result


async def _execute_update(
    table: Optional[str],
    set_expression: Optional[str],
    where_expression: Optional[str],
    session: AsyncSession,
    context: Optional[Dict[str, Any]],
) -> int:
    if not table or not set_expression:
        return 0
    ctx = context or {}
    try:
        sanitized_table = _sanitize_table_reference(table)
        assignments, assignment_params = _parse_update_assignments(set_expression, ctx)
        conditions, where_params = _parse_where_expression(where_expression, ctx)
    except ValueError:
        logger.warning("Rejected unsafe update targeting table '%s'", table)
        return 0
    if not assignments:
        return 0
    parameters: Dict[str, Any] = {}
    parameters.update(assignment_params)
    parameters.update(where_params)
    try:
        if _HAS_SQLA_UPDATE and update is not None and sql_table is not None and column is not None and bindparam is not None:  # type: ignore[truthy-function]
            statement = _build_update_statement(sanitized_table, assignments, conditions)
            if statement is None:
                return 0
            result = await session.execute(statement, parameters)
        else:
            query_text = _build_fallback_update_sql(sanitized_table, assignments, conditions)
            result = await session.execute(text(query_text), parameters)
        await session.commit()
        rowcount = getattr(result, "rowcount", 0) or 0
        return rowcount
    except Exception:
        await session.rollback()
        logger.exception("Failed to execute update on table '%s'", sanitized_table)
        return 0


def _build_update_statement(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> Optional[Any]:
    table_clause = _create_table_clause(table_name, assignments, conditions)
    if table_clause is None:
        return None
    stmt = update(table_clause)  # type: ignore[arg-type]
    values_mapping: Dict[str, Any] = {}
    for column_name, param_name, _ in assignments:
        values_mapping[column_name] = bindparam(param_name)  # type: ignore[arg-type]
    if values_mapping:
        stmt = stmt.values(**values_mapping)
    condition_expr: Optional[Any] = None
    for column_name, param_name, _ in conditions:
        clause = table_clause.c[column_name] == bindparam(param_name)  # type: ignore[index,operator]
        condition_expr = clause if condition_expr is None else condition_expr & clause
    if condition_expr is not None:
        stmt = stmt.where(condition_expr)
    return stmt


def _build_fallback_update_sql(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> str:
    set_clause = ", ".join(f"{column} = :{param}" for column, param, _ in assignments)
    if conditions:
        where_clause = " AND ".join(f"{column} = :{param}" for column, param, _ in conditions)
        return f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
    return f"UPDATE {table_name} SET {set_clause}"


def _create_table_clause(
    table_name: str,
    assignments: Sequence[Tuple[str, str, Any]],
    conditions: Sequence[Tuple[str, str, Any]],
) -> Optional[Any]:
    if sql_table is None or column is None:
        return None
    segments = table_name.split(".")
    name = segments[-1]
    schema = ".".join(segments[:-1]) if len(segments) > 1 else None
    seen: Dict[str, None] = {}
    column_order: List[str] = []
    for column_name, _, _ in list(assignments) + list(conditions):
        if column_name not in seen:
            seen[column_name] = None
            column_order.append(column_name)
    columns = [column(column_name) for column_name in column_order]  # type: ignore[arg-type]
    table_clause = sql_table(name, *columns, schema=schema)  # type: ignore[arg-type]
    return table_clause


def _sanitize_table_reference(value: str) -> str:
    return _sanitize_identifier_path(value)


def _sanitize_identifier_path(raw: str) -> str:
    segments = [segment.strip() for segment in (raw or "").split(".") if segment and segment.strip()]
    if not segments:
        raise ValueError("Empty identifier path")
    cleaned: List[str] = []
    for segment in segments:
        if not _IDENTIFIER_RE.match(segment):
            raise ValueError(f"Invalid identifier segment '{segment}'")
        cleaned.append(segment)
    return ".".join(cleaned)


def _parse_update_assignments(expression: str, context: Dict[str, Any]) -> Tuple[List[Tuple[str, str, Any]], Dict[str, Any]]:
    assignments = _split_assignment_list(expression)
    if not assignments:
        raise ValueError("No assignments provided")
    entries: List[Tuple[str, str, Any]] = []
    params: Dict[str, Any] = {}
    for index, assignment in enumerate(assignments):
        match = _UPDATE_ASSIGNMENT_PATTERN.match(assignment)
        if not match:
            raise ValueError(f"Unsupported assignment '{assignment}'")
        column = _sanitize_identifier_path(match.group(1))
        value_expr = match.group(2).strip()
        value = _evaluate_update_value(value_expr, context)
        param_name = f"set_{index}"
        entries.append((column, param_name, value))
        params[param_name] = value
    return entries, params


def _parse_where_expression(expression: Optional[str], context: Dict[str, Any]) -> Tuple[List[Tuple[str, str, Any]], Dict[str, Any]]:
    if not expression:
        return [], {}
    conditions = _split_conditions(str(expression))
    if not conditions:
        raise ValueError("Unable to parse WHERE expression")
    entries: List[Tuple[str, str, Any]] = []
    params: Dict[str, Any] = {}
    for index, condition in enumerate(conditions):
        match = _WHERE_CONDITION_PATTERN.match(condition)
        if not match:
            raise ValueError(f"Unsupported WHERE clause '{condition}'")
        column = _sanitize_identifier_path(match.group(1))
        value_expr = match.group(2).strip()
        value = _evaluate_update_value(value_expr, context)
        param_name = f"where_{index}"
        entries.append((column, param_name, value))
        params[param_name] = value
    return entries, params


def _split_assignment_list(expression: str) -> List[str]:
    if not expression:
        return []
    parts: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escape = False
    depth = 0
    for char in expression:
        if escape:
            current.append(char)
            escape = False
            continue
        if quote:
            current.append(char)
            if char == "\\\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            continue
        if char in ")]}":
            if depth:
                depth -= 1
            current.append(char)
            continue
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_conditions(expression: str) -> List[str]:
    expr = expression.strip()
    if not expr:
        return []
    parts: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    escape = False
    depth = 0
    i = 0
    length = len(expr)
    while i < length:
        char = expr[i]
        if escape:
            current.append(char)
            escape = False
            i += 1
            continue
        if quote:
            current.append(char)
            if char == "\\\\":
                escape = True
            elif char == quote:
                quote = None
            i += 1
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            i += 1
            continue
        if char in "([{":
            depth += 1
            current.append(char)
            i += 1
            continue
        if char in ")]}":
            if depth:
                depth -= 1
            current.append(char)
            i += 1
            continue
        if depth == 0 and expr[i:].lower().startswith("and") and (i == 0 or expr[i - 1].isspace()) and (i + 3 >= length or expr[i + 3].isspace() or expr[i + 3] in ")]}"):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            i += 3
            while i < length and expr[i].isspace():
                i += 1
            continue
        current.append(char)
        i += 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _evaluate_update_value(expression: str, context: Dict[str, Any]) -> Any:
    value_expr = expression.strip()
    if not value_expr:
        return None
    lowered = value_expr.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value_expr)
    except Exception:
        pass
    result = _evaluate_dataset_expression(value_expr, {}, context, [])
    if result is None and lowered not in {"null", "none"}:
        try:
            if "." in value_expr:
                return float(value_expr)
            return int(value_expr)
        except Exception:
            return value_expr
    return result


async def _handle_post_action_effects(
    operation: Dict[str, Any],
    context: Dict[str, Any],
    result: Dict[str, Any],
    operation_type: str,
) -> None:
    if not result:
        return
    status_value = result.get("status")
    if isinstance(status_value, str) and status_value.lower() in {"error", "failed", "no_session"}:
        return
    dataset_targets = _coerce_dataset_names(
        operation.get("invalidate_datasets")
        or operation.get("refresh_datasets")
        or operation.get("datasets")
    )
    if dataset_targets:
        await _invalidate_datasets(dataset_targets)
    event_topic = operation.get("publish_event") or operation.get("emit_event")
    if event_topic:
        event_payload = _resolve_placeholders(operation.get("event_payload"), context)
        message: Dict[str, Any] = {
            "type": "action",
            "operation": operation_type,
            "result": result,
            "datasets": dataset_targets,
        }
        if event_payload is not None:
            message["payload"] = event_payload
        await publish_event(str(event_topic), message)


def _resolve_page_reference(target: Optional[str]) -> Optional[Dict[str, Any]]:
    if not target:
        return None
    lowered = str(target).strip().lower()
    if not lowered:
        return None
    for page in PAGES:
        name = str(page.get("name") or "").strip()
        slug = str(page.get("slug") or "").strip()
        route = str(page.get("route") or "").strip()
        candidates = {
            name.lower(),
            slug.lower(),
            route.lower(),
        }
        if lowered in candidates:
            return {
                "name": name or page.get("name"),
                "slug": slug or page.get("slug"),
                "route": route or page.get("route"),
            }
    return None


def _get_component_spec(slug: str, component_index: int) -> Dict[str, Any]:
    page_spec = PAGE_SPEC_BY_SLUG.get(slug)
    if not page_spec:
        raise KeyError(f"Unknown page '{slug}'")
    components = page_spec.get("components") or []
    if component_index < 0 or component_index >= len(components):
        raise IndexError(f"Component index {component_index} out of range for '{slug}'")
    component = components[component_index]
    if not isinstance(component, dict):
        raise ValueError("Invalid component payload")
    return component


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _validate_form_submission(
    component: Dict[str, Any],
    submitted: Dict[str, Any],
    context: Dict[str, Any],
) -> bool:
    fields = component.get("fields") or []
    has_errors = False
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if not name:
            continue
        required = field.get("required", True)
        if not required:
            continue
        value = submitted.get(name)
        if _is_missing_value(value):
            has_errors = True
            field_label = field.get("label") or name
            _record_runtime_error(
                context,
                code="missing_required_field",
                message="This field is required.",
                scope=f"field:{name}",
                source="form",
                detail=f"Expected value for '{field_label}'.",
            )
    return has_errors


def _partition_interaction_errors(
    slug: str,
    errors: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not errors:
        return [], []
    widget_errors: List[Dict[str, Any]] = []
    page_errors: List[Dict[str, Any]] = []
    slug_value = slug or ""
    slug_lower = slug_value.lower()
    page_scope = f"page:{slug_value}"
    page_scope_lower = page_scope.lower()
    page_dot_scope_lower = f"page.{slug_value}".lower()
    for entry in errors:
        if not isinstance(entry, dict):
            continue
        scope_value = entry.get("scope")
        normalized_scope = str(scope_value).strip().lower() if scope_value is not None else ""
        if normalized_scope in {"", slug_lower, page_scope_lower, page_dot_scope_lower, "page"}:
            entry["scope"] = page_scope
            page_errors.append(entry)
            continue
        if normalized_scope.startswith("page:") or normalized_scope.startswith("page.") or normalized_scope.startswith("app:"):
            page_errors.append(entry)
            continue
        widget_errors.append(entry)
    return widget_errors, page_errors


async def _execute_component_interaction(
    slug: str,
    component_index: int,
    payload: Optional[Dict[str, Any]],
    session: Optional[AsyncSession],
    *,
    kind: str,
) -> Dict[str, Any]:
    component = _get_component_spec(slug, component_index)
    component_type = str(component.get("type") or "").lower()
    if component_type != kind.lower():
        raise ValueError(f"Component {component_index} on '{slug}' is not of type '{kind}'")

    component_payload = component.get("payload") if isinstance(component.get("payload"), dict) else component

    submitted: Dict[str, Any] = dict(payload or {})
    context = build_context(slug)
    if session is not None:
        context["session"] = session
    context.setdefault("vars", {})
    for key, value in submitted.items():
        if isinstance(key, str):
            context["vars"].setdefault(key, value)
    context.setdefault("payload", {}).update(submitted)
    if kind == "form":
        context.setdefault("form", {}).update(submitted)

    scope = ScopeFrame()
    scope.set("context", context)
    scope.bind("payload", submitted)
    if kind == "form":
        scope.bind("form", submitted)

    operations = component_payload.get("operations") or []
    results: List[Dict[str, Any]] = []
    validation_failed = False
    if kind == "form":
        validation_failed = _validate_form_submission(component_payload, submitted, context)
    if not validation_failed:
        for operation in operations:
            outcome = await _execute_action_operation(operation, context, scope)
            if outcome:
                results.append(outcome)

    response: Dict[str, Any] = {
        "status": "ok",
        "slug": slug,
        "component_index": component_index,
        "type": kind,
        "results": results,
    }
    if kind == "form":
        response["accepted"] = submitted
    if results:
        response["effects"] = results
    collected_errors: List[Dict[str, Any]] = []
    if slug:
        collected_errors.extend(_collect_runtime_errors(context, scope=slug))
    collected_errors.extend(_collect_runtime_errors(context))
    widget_errors, page_errors = _partition_interaction_errors(slug, collected_errors)
    if widget_errors:
        response["errors"] = widget_errors
    if page_errors:
        response["page_errors"] = page_errors
        response["pageErrors"] = page_errors
    if widget_errors or page_errors:
        combined = widget_errors + page_errors
        if any((entry.get("severity") or "error").lower() == "error" for entry in combined if isinstance(entry, dict)):
            response["status"] = "error"
    return response


async def submit_form(
    slug: str,
    component_index: int,
    payload: Optional[Dict[str, Any]],
    *,
    session: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    return await _execute_component_interaction(
        slug,
        component_index,
        payload,
        session,
        kind="form",
    )


async def trigger_action(
    slug: str,
    component_index: int,
    payload: Optional[Dict[str, Any]],
    *,
    session: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    return await _execute_component_interaction(
        slug,
        component_index,
        payload,
        session,
        kind="action",
    )

    '''
).strip()

__all__ = ['ACTIONS_SECTION']
