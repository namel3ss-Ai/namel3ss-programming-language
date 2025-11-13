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
