from __future__ import annotations

from textwrap import dedent

RENDERING_SECTION = dedent(
    '''
async def render_statements(
    statements: Optional[Iterable[Dict[str, Any]]],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    return await _render_block(statements, context, scope, session, allow_control=False)


async def _render_block(
    statements: Optional[Iterable[Dict[str, Any]]],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
    *,
    allow_control: bool,
) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    if not statements:
        return components
    for statement in statements:
        try:
            result = await _render_statement(statement, context, scope, session)
        except (BreakFlow, ContinueFlow) as exc:
            if components:
                exc.extend(components)
            if allow_control:
                raise exc


            logger.warning("Loop control statement encountered outside of a loop")
            break
        if result:
            components.extend(result)
    return components


async def _render_statement(
    statement: Dict[str, Any],


    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    stype = statement.get("type")
    if stype == "variable":
        name = statement.get("name")
        expr = statement.get("value_expr")
        raw_value = statement.get("value")
        pipeline_plan = raw_value.get("__frame_pipeline__") if isinstance(raw_value, dict) else None
        if pipeline_plan is not None:
            value = await _resolve_frame_pipeline_value(raw_value, session, context)
        elif expr is not None:
            value = await _evaluate_runtime_expression(expr, context, scope)
        else:
            value = _resolve_placeholders(raw_value, context)
        if name:
            _assign_variable(scope, context, name, value)
        return []
    if stype == "if":
        return await _render_if(statement, context, scope, session)
    if stype == "for_loop":
        return await _render_for_loop(statement, context, scope, session)
    if stype == "while_loop":
        return await _render_while_loop(statement, context, scope, session)
    if stype == "break":
        raise BreakFlow()
    if stype == "continue":
        raise ContinueFlow()

    payload = copy.deepcopy(statement)
    payload_type = payload.pop("type", None)
    component_index = payload.pop("__component_index", None)
    if payload_type == "text":
        rendered_text = _render_template_value(payload.get("text"), context)
        styles = copy.deepcopy(payload.get("styles", {}))
        return [{"type": "text", "text": rendered_text, "styles": styles}]
    if payload_type == "table":
        resolved = _resolve_placeholders(payload, context)
        if "title" in resolved:
            resolved["title"] = _render_template_value(resolved.get("title"), context)
        if "filter" in resolved:
            resolved["filter"] = _render_template_value(resolved.get("filter"), context)
        if "sort" in resolved:
            resolved["sort"] = _render_template_value(resolved.get("sort"), context)
        error_scope = _component_error_scope(resolved)
        component_errors: List[Dict[str, Any]] = []
        if error_scope:
            component_errors.extend(_collect_runtime_errors(context, error_scope))
        insight_ref = resolved.get("insight")
        if isinstance(insight_ref, str):
            component_errors.extend(_collect_runtime_errors(context, insight_ref))
        if component_errors:
            resolved["errors"] = component_errors
        if component_index is not None:
            resolved["component_index"] = component_index
        return [{"type": "table", **resolved}]
    if payload_type == "chart":
        resolved = _resolve_placeholders(payload, context)
        if "heading" in resolved:
            resolved["heading"] = _render_template_value(resolved.get("heading"), context)
        if "title" in resolved:
            resolved["title"] = _render_template_value(resolved.get("title"), context)
        dataset_ref = _component_error_scope(resolved)
        component_errors: List[Dict[str, Any]] = []
        if dataset_ref:
            component_errors.extend(_collect_runtime_errors(context, dataset_ref))
        insight_ref = resolved.get("insight")
        if isinstance(insight_ref, str):
            component_errors.extend(_collect_runtime_errors(context, insight_ref))
        if component_errors:
            resolved["errors"] = component_errors
        if component_index is not None:
            resolved["component_index"] = component_index
        return [{"type": "chart", **resolved}]
    if payload_type == "form":
        resolved = _resolve_placeholders(payload, context)
        if "title" in resolved:


            resolved["title"] = _render_template_value(resolved.get("title"), context)
        if component_index is not None:
            resolved["component_index"] = component_index
        return [{"type": "form", **resolved}]
    if payload_type == "action":
        resolved = _resolve_placeholders(payload, context)
        if "name" in resolved:
            resolved["name"] = _render_template_value(resolved.get("name"), context)
        if "trigger" in resolved:
            resolved["trigger"] = _render_template_value(resolved.get("trigger"), context)
        operations = resolved.get("operations") or []
        results: List[Dict[str, Any]] = []
        for operation in operations:
            outcome = await _execute_action_operation(operation, context, scope)
            if outcome:
                results.append(outcome)
        if component_index is not None:
            resolved["component_index"] = component_index
        if results:
            return results
        return [{"type": "action", **resolved}]
    if payload_type == "predict":
        resolved = _resolve_placeholders(payload, context)
        if component_index is not None:
            resolved["component_index"] = component_index
        return [{"type": "predict", **resolved}]
    if payload_type:
        resolved = _resolve_placeholders(payload, context)
        return [{"type": payload_type, **resolved}]
    return []


async def _render_if(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    condition = statement.get("condition")


    if _runtime_truthy(await _evaluate_runtime_expression(condition, context, scope)):
        return await _render_block(statement.get("body"), context, scope, session, allow_control=True)
    for branch in statement.get("elifs", []) or []:
        branch_condition = branch.get("condition")
        if _runtime_truthy(await _evaluate_runtime_expression(branch_condition, context, scope)):
            return await _render_block(branch.get("body"), context, scope, session, allow_control=True)
    else_body = statement.get("else_body")
    if else_body:
        return await _render_block(else_body, context, scope, session, allow_control=True)
    return []


async def _render_for_loop(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    loop_var = statement.get("loop_var")
    source_kind = statement.get("source_kind")
    source_name = statement.get("source_name")
    if not loop_var or not source_kind or not source_name:
        return []
    items = await _resolve_loop_iterable(source_kind, source_name, context, scope, session)
    if not items:
        return []
    components: List[Dict[str, Any]] = []


    vars_map = context.setdefault("vars", {})
    for item in items:
        previous = vars_map.get(loop_var, _MISSING)
        loop_scope = scope.child()
        _bind_variable(loop_scope, context, loop_var, item)
        try:
            rendered = await _render_block(statement.get("body"), context, loop_scope, session, allow_control=True)
            if rendered:
                components.extend(rendered)
        except ContinueFlow as exc:
            if exc.components:
                components.extend(exc.components)
            continue
        except BreakFlow as exc:
            if exc.components:
                components.extend(exc.components)
            break


        finally:
            _restore_variable(context, loop_var, previous)
    return components


async def _render_while_loop(
    statement: Dict[str, Any],
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    condition = statement.get("condition")
    components: List[Dict[str, Any]] = []
    loop_scope = scope.child()
    iterations = 0
    while _runtime_truthy(await _evaluate_runtime_expression(condition, context, loop_scope)):
        iterations += 1
        if iterations > 1000:
            logger.warning("Aborting while loop after 1000 iterations for safety")
            break
        try:
            rendered = await _render_block(statement.get("body"), context, loop_scope, session, allow_control=True)
            if rendered:
                components.extend(rendered)
        except ContinueFlow as exc:
            if exc.components:
                components.extend(exc.components)
            continue
        except BreakFlow as exc:
            if exc.components:
                components.extend(exc.components)
            break
    return components


async def _resolve_loop_iterable(
    source_kind: str,
    source_name: str,
    context: Dict[str, Any],
    scope: ScopeFrame,
    session: Optional[AsyncSession],
) -> List[Any]:
    if source_kind == "dataset":
        if session is not None:
            try:
                return await fetch_dataset_rows(source_name, session, context)
            except Exception as exc:  # pragma: no cover - runtime fetch failure
                logger.exception("Failed to fetch dataset rows for %s", source_name)
                _record_runtime_error(
                    context,
                    code="dataset_fetch_failed",
                    message=f"Failed to fetch dataset '{source_name}'.",
                    scope=source_name,
                    source="dataset",
                    detail=str(exc),
                )
        dataset_spec = DATASETS.get(source_name)
        if dataset_spec:
            return list(dataset_spec.get("sample_rows", []))


        return []
    if source_kind == "frame":
        try:
            return await fetch_frame_rows(source_name, session, context)
        except Exception as exc:  # pragma: no cover - runtime fetch failure
            logger.exception("Failed to fetch frame rows for %s", source_name)
            _record_runtime_error(
                context,
                code="frame_fetch_failed",
                message=f"Failed to fetch frame '{source_name}'.",
                scope=source_name,
                source="frame",
                detail=str(exc),
            )
        frame_spec = FRAMES.get(source_name)
        if isinstance(frame_spec, dict):
            examples = frame_spec.get("examples")
            if isinstance(examples, list):
                sample_rows = [row for row in examples if isinstance(row, dict)]
                if sample_rows:
                    return _clone_rows(sample_rows)
        return []
    if source_kind == "table":
        tables = context.get("tables")
        if isinstance(tables, dict) and source_name in tables:
            table_value = tables[source_name]
            if isinstance(table_value, list):
                return table_value
        dataset_spec = DATASETS.get(source_name)
        if dataset_spec:
            return list(dataset_spec.get("sample_rows", []))
        return []
    items = scope.get(source_name)
    if isinstance(items, list):
        return items
    return []


async def _evaluate_runtime_expression(
    expression: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    scope: ScopeFrame,
) -> Any:
    if expression is None:
        return None
    etype = expression.get("type")
    if etype == "literal":
        return _resolve_placeholders(expression.get("value"), context)
    if etype == "name":
        name = expression.get("name")
        if not isinstance(name, str):
            return None
        value = scope.get(name, _MISSING)
        if value is not _MISSING:
            return value
        vars_map = context.get("vars", {})
        if isinstance(vars_map, dict) and name in vars_map:
            return vars_map[name]
        if name in _RUNTIME_CALLABLES:
            return _RUNTIME_CALLABLES[name]
        return context.get(name)
    if etype == "attribute":
        path = list(expression.get("path") or [])
        if not path:
            return None
        base_name = path[0]
        base_value = await _evaluate_runtime_expression({"type": "name", "name": base_name}, context, scope)
        if base_value is None:
            return None
        return _traverse_attribute_path(base_value, path[1:])
    if etype == "context":
        return _resolve_context_scope(
            expression.get("scope"),
            expression.get("path", []),
            context,
            expression.get("default"),
        )
    if etype == "binary":
        op = expression.get("op")
        left = await _evaluate_runtime_expression(expression.get("left"), context, scope)
        right = await _evaluate_runtime_expression(expression.get("right"), context, scope)
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right if right not in (0, None) else None
            if op == "and":
                return _runtime_truthy(left) and _runtime_truthy(right)
            if op == "or":
                return _runtime_truthy(left) or _runtime_truthy(right)
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == "in":
                return left in right if right is not None else False
            if op == "not in":
                return left not in right if right is not None else True
        except Exception as exc:  # pragma: no cover - operator failure
            logger.exception("Failed to evaluate binary operation '%s'", op)
            _record_runtime_error(
                context,
                code="runtime_expression_failed",
                message=f"Binary operation '{op}' failed during evaluation.",
                scope=context.get("page") if isinstance(context, dict) else None,
                source="page",
                detail=str(exc),
            )
            return None
        return None
    if etype == "unary":
        operand = await _evaluate_runtime_expression(expression.get("operand"), context, scope)
        op = expression.get("op")
        try:
            if op == "not":
                return not _runtime_truthy(operand)
            if op == "-":
                return -operand
            if op == "+":
                return +operand
        except Exception as exc:  # pragma: no cover - unary failure
            logger.exception("Failed to evaluate unary operation '%s'", op)
            _record_runtime_error(
                context,
                code="runtime_expression_failed",
                message=f"Unary operation '{op}' failed during evaluation.",
                scope=context.get("page") if isinstance(context, dict) else None,
                source="page",
                detail=str(exc),
            )
            return None
        return None
    if etype == "call":
        function_expr = expression.get("function")
        func = await _evaluate_runtime_expression(function_expr, context, scope)
        if func is None:
            return None
        arguments = []
        for arg in expression.get("arguments", []) or []:
            arguments.append(await _evaluate_runtime_expression(arg, context, scope))
        try:
            if inspect.iscoroutinefunction(func):
                return await func(*arguments)
            result = func(*arguments) if callable(func) else None
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:  # pragma: no cover - call failure
            logger.exception("Failed to execute call expression")
            _record_runtime_error(
                context,
                code="runtime_expression_failed",
                message="Runtime call expression failed during execution.",
                scope=context.get("page") if isinstance(context, dict) else None,
                source="page",
                detail=str(exc),
            )
            return None
    return None


def _traverse_attribute_path(value: Any, path: Iterable[str]) -> Any:
    current = value
    for segment in path:
        if current is None:
            return None
        current = _resolve_path_segment(current, segment)
    return current


def _resolve_path_segment(value: Any, segment: str) -> Any:
    if isinstance(value, dict):
        return value.get(segment)
    if isinstance(value, (list, tuple)):
        try:
            index = int(segment)
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(value):
            return value[index]
        return None
    if hasattr(value, segment):
        return getattr(value, segment)
    return None


async def prepare_page_components(
    page_meta: Dict[str, Any],
    components: List[Dict[str, Any]],
    context: Dict[str, Any],
    session: Optional[AsyncSession],
) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {}
    base_path = page_meta.get('api_path') or '/api/pages'
    for order, component in enumerate(components or []):
        if not isinstance(component, dict):
            continue
        clone = copy.deepcopy(component)
        ctype = str(clone.get('type') or 'component')
        counters[ctype] = counters.get(ctype, 0) + 1
        clone.setdefault('id', f"{ctype}-{counters[ctype]}")
        clone['order'] = order
        component_index = clone.get('component_index')
        if ctype == 'table':
            await _hydrate_table_component(clone, context, session)
            if component_index is not None:
                clone.setdefault('endpoint', f"{base_path}/tables/{component_index}")
        elif ctype == 'chart':
            await _hydrate_chart_component(clone, context, session)
            if component_index is not None:
                clone.setdefault('endpoint', f"{base_path}/charts/{component_index}")
        elif ctype == 'form' and component_index is not None:
            clone.setdefault('submit_url', f"{base_path}/forms/{component_index}")
        elif ctype == 'action' and component_index is not None:
            clone.setdefault('action_url', f"{base_path}/actions/{component_index}")
        hydrated.append(clone)
    return hydrated


async def _hydrate_table_component(
    component: Dict[str, Any],
    context: Dict[str, Any],
    session: Optional[AsyncSession],
) -> None:
    source_type, dataset_name = _component_source_info(component)
    rows: List[Dict[str, Any]] = []
    if dataset_name:
        if source_type == 'frame':
            try:
                rows = await fetch_frame_rows(dataset_name, session, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to fetch rows for frame '%s'", dataset_name)
                _record_runtime_error(
                    context,
                    code="frame_fetch_failed",
                    message=f"Failed to fetch frame '{dataset_name}'.",
                    scope=dataset_name,
                    source="frame",
                    detail=str(exc),
                )
            if not rows:
                frame_spec = FRAMES.get(dataset_name) if isinstance(FRAMES, dict) else None
                if isinstance(frame_spec, dict):
                    examples = frame_spec.get('examples')
                    if isinstance(examples, list):
                        sample_rows = [row for row in examples if isinstance(row, dict)]
                        if sample_rows:
                            rows = _clone_rows(sample_rows)
        else:
            try:
                rows = await fetch_dataset_rows(dataset_name, session, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to fetch rows for dataset '%s'", dataset_name)
                _record_runtime_error(
                    context,
                    code="dataset_fetch_failed",
                    message=f"Failed to fetch dataset '{dataset_name}'.",
                    scope=dataset_name,
                    source="dataset",
                    detail=str(exc),
                )
            if not rows:
                dataset_spec = DATASETS.get(dataset_name) or {}
                sample_rows = dataset_spec.get('sample_rows') if isinstance(dataset_spec, dict) else None
                if isinstance(sample_rows, list):
                    rows = _clone_rows(sample_rows)
    if rows:
        rows = _clone_rows(rows)
    else:
        rows = []
    limited_rows = rows[:50]
    component['rows'] = limited_rows
    columns = component.get('columns')
    if not columns:
        if limited_rows:
            columns = list(limited_rows[0].keys())
        elif source_type == 'frame' and dataset_name:
            frame_spec = FRAMES.get(dataset_name) if isinstance(FRAMES, dict) else None
            if isinstance(frame_spec, dict):
                frame_columns = frame_spec.get('columns') or []
                if isinstance(frame_columns, list):
                    candidate = [col.get('name') for col in frame_columns if isinstance(col, dict) and col.get('name')]
                    if candidate:
                        columns = candidate
        columns = columns or []
    component['columns'] = columns


async def _hydrate_chart_component(
    component: Dict[str, Any],
    context: Dict[str, Any],
    session: Optional[AsyncSession],
) -> None:
    source_type, dataset_name = _component_source_info(component)
    rows: List[Dict[str, Any]] = []
    if dataset_name:
        if source_type == 'frame':
            try:
                rows = await fetch_frame_rows(dataset_name, session, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to fetch chart rows for frame '%s'", dataset_name)
                _record_runtime_error(
                    context,
                    code="frame_fetch_failed",
                    message=f"Failed to fetch frame '{dataset_name}' for chart.",
                    scope=dataset_name,
                    source="frame",
                    detail=str(exc),
                )
            if not rows:
                frame_spec = FRAMES.get(dataset_name) if isinstance(FRAMES, dict) else None
                if isinstance(frame_spec, dict):
                    examples = frame_spec.get('examples')
                    if isinstance(examples, list):
                        dict_rows = [row for row in examples if isinstance(row, dict)]
                        if dict_rows:
                            rows = _clone_rows(dict_rows)
        else:
            try:
                rows = await fetch_dataset_rows(dataset_name, session, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to fetch chart rows for dataset '%s'", dataset_name)
                _record_runtime_error(
                    context,
                    code="dataset_fetch_failed",
                    message=f"Failed to fetch dataset '{dataset_name}' for chart.",
                    scope=dataset_name,
                    source="dataset",
                    detail=str(exc),
                )
            if not rows:
                dataset_spec = DATASETS.get(dataset_name) or {}
                sample_rows = dataset_spec.get('sample_rows') if isinstance(dataset_spec, dict) else None
                if isinstance(sample_rows, list):
                    rows = _clone_rows(sample_rows)
    if rows:
        rows = _clone_rows(rows)
    else:
        rows = []
    x_key = component.get('x')
    y_key = component.get('y')
    labels: List[Any] = []
    values: List[float] = []
    limited_rows = rows[:50]
    for idx, row in enumerate(limited_rows, start=1):
        if not isinstance(row, dict):
            continue
        label_value = row.get(x_key) if x_key else None
        if label_value is None:
            label_value = row.get('label') or row.get('name') or idx
        labels.append(label_value)
        raw_value = row.get(y_key) if y_key else row.get('value')
        values.append(_coerce_number(raw_value))
    if not labels and limited_rows:
        labels = list(range(1, len(limited_rows) + 1))
    if not values:
        values = [0 for _ in labels] or [0]
    component['labels'] = labels
    component['series'] = [{
        'label': component.get('title') or component.get('heading') or 'Series',
        'data': values,
    }]
    component['rows'] = limited_rows


def _component_source_info(component: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    source_type = str(component.get('source_type') or '').lower()
    source = component.get('source')
    source_name: Optional[str] = None
    if isinstance(source, str):
        source_name = source
    elif isinstance(source, dict):
        name = source.get('name')
        if isinstance(name, str):
            source_name = name
        if not source_type:
            inline_type = source.get('type')
            if isinstance(inline_type, str):
                source_type = inline_type.lower()
    if not source_type and source_name:
        source_type = 'dataset'
    return source_type, source_name


def _component_dataset_name(component: Dict[str, Any]) -> Optional[str]:
    _, source_name = _component_source_info(component)
    return source_name


def _component_error_scope(component: Dict[str, Any]) -> Optional[str]:
    source_type, source_name = _component_source_info(component)
    if source_name and source_type in {'dataset', 'table', 'frame'}:
        return source_name
    return None


def _coerce_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

    '''
).strip()

__all__ = ['RENDERING_SECTION']
