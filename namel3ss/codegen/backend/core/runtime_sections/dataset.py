from __future__ import annotations

from textwrap import dedent

DATASET_SECTION = dedent(
    '''
def _parse_aggregate_expression(expression: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if expression is None:
        return None, None
    text = expression.strip()
    if not text:
        return None, None
    parts = _AGGREGATE_ALIAS_PATTERN.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip() or None
    return text, None


def _aggregate_result_key(function: str, expression: Optional[str], alias: Optional[str]) -> str:
    if alias:
        return alias
    base = f"{function}_{expression or 'value'}"
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_")
    return sanitized or f"{function}_value"


def _evaluate_dataset_expression(
    expression: Optional[str],
    row: Dict[str, Any],
    context: Dict[str, Any],
    rows: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    if expression is None:
        return None
    expr = expression.strip()
    if not expr:
        return None
    scope: Dict[str, Any] = {
        "row": row,
        "rows": rows or [],
        "context": context,
    }
    scope.update(row)
    scope.setdefault("math", math)
    scope.setdefault("len", len)
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "round": round,
    }
    try:
        compiled = compile(expr, "<dataset_expr>", "eval")
        return eval(compiled, {"__builtins__": safe_builtins}, scope)
    except Exception:
        logger.debug("Failed to evaluate dataset expression '%s'", expression)
        return None


def _apply_filter(rows: List[Dict[str, Any]], condition: Optional[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not rows or not condition:
        return rows
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        value = _evaluate_dataset_expression(condition, row, context, rows)
        if _runtime_truthy(value):
            filtered.append(row)
    return filtered


def _apply_computed_column(
    rows: List[Dict[str, Any]],
    name: str,
    expression: Optional[str],
    context: Dict[str, Any],
) -> None:
    if not name:
        return
    for row in rows:
        row[name] = _evaluate_dataset_expression(expression, row, context, rows)


def _apply_group_aggregate(
    rows: List[Dict[str, Any]],
    columns: Sequence[str],
    aggregates: Sequence[Tuple[str, str]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not columns:
        key = tuple()
        grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {key: rows}
    else:
        grouped = defaultdict(list)
        for row in rows:
            key = tuple(row.get(column) for column in columns)
            grouped[key].append(row)
    results: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        base: Dict[str, Any] = {}
        for idx, column in enumerate(columns):
            base[column] = key[idx] if idx < len(key) else None
        for function, expression in aggregates:
            expr_source, alias = _parse_aggregate_expression(expression)
            result_key = _aggregate_result_key(function or "agg", expr_source, alias)
            values = []
            if expr_source:
                for row in items:
                    values.append(_evaluate_dataset_expression(expr_source, row, context, items))
            numeric_values = [_ensure_numeric(value) for value in values if value is not None]
            func_lower = str(function or "").lower()
            if func_lower == "sum":
                base[result_key] = sum(numeric_values) if numeric_values else 0
            elif func_lower == "count":
                if values:
                    base[result_key] = sum(1 for value in values if _runtime_truthy(value))
                else:
                    base[result_key] = len(items)
            elif func_lower == "avg":
                base[result_key] = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            elif func_lower == "min":
                base[result_key] = min(numeric_values) if numeric_values else None
            elif func_lower == "max":
                base[result_key] = max(numeric_values) if numeric_values else None
            else:
                base[result_key] = values
        results.append(base)
    return results


def _apply_order(rows: List[Dict[str, Any]], columns: Sequence[str]) -> List[Dict[str, Any]]:
    if not columns:
        return rows

    def sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        values: List[Any] = []
        for column in columns:
            column_name = column.lstrip("-")
            values.append(row.get(column_name))
        return tuple(values)

    descending_flags = [col.startswith("-") for col in columns]
    ordered = sorted(rows, key=sort_key, reverse=descending_flags[0] if descending_flags else False)
    if any(descending_flags[1:]):
        for index in range(1, len(columns)):
            column = columns[index]
            if not column.startswith("-"):
                continue
            column_name = column.lstrip("-")
            ordered = sorted(ordered, key=lambda item: item.get(column_name), reverse=True)
    return ordered


def _apply_window_operation(
    rows: List[Dict[str, Any]],
    name: str,
    function: str,
    target: Optional[str],
) -> None:
    if not name:
        return
    func_lower = (function or "").lower()
    values = [row.get(target) if target else None for row in rows]
    if func_lower == "rank":
        sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
        ranks = {index: rank + 1 for rank, (index, _) in enumerate(sorted_pairs)}
        for idx, row in enumerate(rows):
            row[name] = ranks.get(idx, idx + 1)
        return
    running_total = 0.0
    for index, row in enumerate(rows):
        value = _ensure_numeric(row.get(target)) if target else 0.0
        if func_lower in {"cumsum", "running_sum", "sum"}:
            running_total += value
            row[name] = running_total
        elif func_lower == "avg":
            running_total += value
            row[name] = running_total / (index + 1)
        else:
            row[name] = value


def _apply_transforms(
    rows: List[Dict[str, Any]],
    transforms: Sequence[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    current = _clone_rows(rows)
    for step in transforms:
        transform_type = str(step.get("type") or "").lower()
        handler = DATASET_TRANSFORMS.get(transform_type)
        if handler is None:
            continue
        options = _resolve_option_dict(step.get("options") if isinstance(step, dict) else {})
        try:
            current = handler(current, options, context)
        except Exception:
            logger.exception("Dataset transform '%s' failed", transform_type)
    return current


def _evaluate_quality_checks(
    rows: List[Dict[str, Any]],
    checks: Sequence[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for check in checks:
        name = check.get("name")
        condition = check.get("condition")
        metric = check.get("metric")
        threshold = check.get("threshold")
        severity = check.get("severity", "error")
        message = check.get("message")
        extras = dict(check.get("extras") or {})
        passed = True
        metric_value = None
        if metric:
            metric_lower = metric.lower()
            if metric_lower == "row_count":
                metric_value = len(rows)
            elif metric_lower == "null_ratio" and threshold:
                target_column = extras.get("column")
                if target_column:
                    nulls = sum(1 for row in rows if row.get(target_column) in {None, ""})
                    metric_value = nulls / max(len(rows), 1)
            else:
                metric_value = sum(_ensure_numeric(row.get(metric)) for row in rows)
        if condition:
            violations = [
                row for row in rows
                if not _runtime_truthy(_evaluate_dataset_expression(condition, row, context, rows))
            ]
            passed = not violations
        elif isinstance(threshold, (int, float)) and metric_value is not None:
            passed = metric_value >= threshold
        results.append(
            {
                "name": name,
                "passed": passed,
                "severity": severity,
                "message": message,
                "metric": metric,
                "value": metric_value,
                "threshold": threshold,
                "extras": extras,
            }
        )
    return results


def _dataset_cache_settings(dataset: Dict[str, Any], context: Dict[str, Any]) -> Tuple[str, bool, Optional[int]]:
    policy = dataset.get("cache_policy") or {}
    if not isinstance(policy, dict):
        policy = {}
    runtime_cache_toggle = _runtime_setting("cache_enabled")
    dataset_runtime_settings = _runtime_setting("datasets")
    dataset_runtime_cache: Optional[bool] = None
    default_scope = context.get("page") or "global"
    default_ttl: Optional[int] = None
    if isinstance(dataset_runtime_settings, dict):
        dataset_runtime_cache = dataset_runtime_settings.get("cache_enabled") if isinstance(dataset_runtime_settings.get("cache_enabled"), bool) else None
        runtime_scope = dataset_runtime_settings.get("default_scope")
        if isinstance(runtime_scope, str) and runtime_scope.strip():
            default_scope = runtime_scope.strip()
        runtime_ttl = dataset_runtime_settings.get("ttl")
        if isinstance(runtime_ttl, int) and runtime_ttl > 0:
            default_ttl = runtime_ttl
    strategy = str(policy.get("strategy") or "auto").lower()
    enabled = strategy not in {"none", "off", "disabled"}
    ttl_value = policy.get("ttl_seconds")
    ttl = ttl_value if isinstance(ttl_value, int) and ttl_value > 0 else default_ttl
    scope = policy.get("scope") or default_scope
    if isinstance(dataset_runtime_cache, bool):
        enabled = dataset_runtime_cache
    if isinstance(runtime_cache_toggle, bool):
        enabled = enabled and runtime_cache_toggle
    return scope, enabled, ttl


def _make_dataset_cache_key(dataset_name: str, scope: str, context: Dict[str, Any]) -> str:
    vars_snapshot = context.get("vars") if isinstance(context.get("vars"), dict) else {}
    try:
        serialised = json.dumps(vars_snapshot, sort_keys=True, default=str)
    except TypeError:
        safe_snapshot = {key: str(value) for key, value in vars_snapshot.items()}
        serialised = json.dumps(safe_snapshot, sort_keys=True)
    digest = hashlib.sha1(serialised.encode("utf-8")).hexdigest()
    return f"dataset::{scope}::{dataset_name}::{digest}"


def _coerce_dataset_names(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        names: List[str] = []
        for key, enabled in value.items():
            if enabled:
                names.append(str(key))
        return names
    if isinstance(value, (list, tuple, set)):
        names = [str(item) for item in value if isinstance(item, str) and item]
        return names
    return []


async def _invalidate_dataset(dataset_name: str) -> None:
    if not dataset_name:
        return
    tracked_keys = list(DATASET_CACHE_INDEX.get(dataset_name, set()))
    if tracked_keys:
        for cache_key in tracked_keys:
            try:
                await _cache_delete(cache_key)
            except Exception:
                logger.exception("Failed to delete cache entry '%s' for dataset '%s'", cache_key, dataset_name)
    DATASET_CACHE_INDEX.pop(dataset_name, None)
    STREAM_CONFIG.setdefault(dataset_name, {})["last_invalidate_ts"] = time.time()
    message = {"type": "dataset.invalidate", "dataset": dataset_name}
    await publish_dataset_event(dataset_name, message)
    if REALTIME_ENABLED:
        timestamped = _with_timestamp(dict(message))
        subscribers = list(DATASET_SUBSCRIBERS.get(dataset_name, set()))
        if subscribers:
            for slug in subscribers:
                payload = dict(timestamped)
                payload["slug"] = slug
                await BROADCAST.broadcast(slug, payload)
        else:
            await BROADCAST.broadcast(dataset_name, timestamped)


async def _invalidate_datasets(dataset_names: Iterable[str]) -> None:
    unique = []
    seen: Set[str] = set()
    for name in dataset_names:
        if not name or name in seen:
            continue
        if name not in DATASETS:
            continue
        seen.add(name)
        unique.append(name)
    if not unique:
        return
    await asyncio.gather(*[_invalidate_dataset(name) for name in unique])


async def _broadcast_dataset_refresh(slug: Optional[str], dataset_name: str, payload: List[Dict[str, Any]]) -> None:
    message = {
        "type": "dataset",
        "slug": slug,
        "dataset": dataset_name,
        "rows": payload,
        "meta": _page_meta(slug) if slug else {},
    }
    await publish_dataset_event(dataset_name, message)
    if REALTIME_ENABLED:
        await BROADCAST.broadcast(slug or dataset_name, _with_timestamp(dict(message)))


async def _schedule_dataset_refresh(dataset_name: str, dataset: Dict[str, Any], session: Optional[AsyncSession], context: Dict[str, Any]) -> None:
    policy = dataset.get("refresh_policy") or {}
    if not isinstance(policy, dict):
        return
    interval = policy.get("interval_seconds") or policy.get("interval")
    try:
        interval_value = int(interval)
    except (TypeError, ValueError):
        return
    if interval_value <= 0:
        return
    slug = context.get("page") if isinstance(context.get("page"), str) else None
    if slug:
        DATASET_SUBSCRIBERS[dataset_name].add(slug)
    STREAM_CONFIG.setdefault(dataset_name, {})["refresh_interval"] = interval_value
    context.setdefault("refresh_schedule", {})[dataset_name] = interval_value
    policy_message = {
        "type": "refresh_policy",
        "slug": slug,
        "dataset": dataset_name,
        "interval": interval_value,
    }
    await publish_dataset_event(dataset_name, policy_message)
    if REALTIME_ENABLED and slug:
        await BROADCAST.broadcast(slug, _with_timestamp(dict(policy_message)))

    '''
).strip()

__all__ = ['DATASET_SECTION']
