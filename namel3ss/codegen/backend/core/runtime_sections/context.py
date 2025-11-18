from __future__ import annotations

from textwrap import dedent

CONTEXT_SECTION = dedent(
    '''


def _record_runtime_error(
    context: Dict[str, Any],
    *,
    code: str,
    message: str,
    scope: Optional[str] = None,
    source: str = "runtime",
    detail: Optional[str] = None,
    severity: str = "error",
) -> Dict[str, Any]:
    severity_value = severity if severity in {"debug", "info", "warning", "error"} else "error"
    error_entry = {
        "code": code,
        "message": message,
        "scope": scope,
        "source": source,
        "detail": detail,
        "severity": severity_value,
    }
    context.setdefault("errors", []).append(error_entry)
    return error_entry


def _observability_setting_from(config: Any, channel: str) -> Optional[bool]:
    if not isinstance(config, dict):
        return None
    if "enabled" in config:
        enabled_flag = bool(config["enabled"])
        if not enabled_flag:
            return False
        default_flag: Optional[bool] = True
    else:
        default_flag = None
    if channel in config:
        return bool(config[channel])
    return default_flag


def _observability_enabled(context: Optional[Dict[str, Any]], channel: str) -> bool:
    result: Optional[bool] = None
    runtime_config = RUNTIME_SETTINGS.get("observability") if "observability" in RUNTIME_SETTINGS else None
    if isinstance(runtime_config, dict):
        flag = _observability_setting_from(runtime_config, channel)
        if flag is not None:
            result = flag
    if isinstance(context, dict):
        context_config = context.get("observability")
        if isinstance(context_config, dict):
            flag = _observability_setting_from(context_config, channel)
            if flag is not None:
                result = flag
    if result is None:
        return True
    return result


def _record_runtime_metric(
    context: Optional[Dict[str, Any]],
    *,
    name: str,
    value: Any,
    unit: str = "count",
    tags: Optional[Dict[str, Any]] = None,
    scope: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    if not _observability_enabled(context, "metrics"):
        return {}
    try:
        numeric_value: Any
        if isinstance(value, (int, float)):
            numeric_value = float(value)
        else:
            numeric_value = value
    except Exception:
        numeric_value = value
    entry = {
        "name": str(name),
        "value": numeric_value,
        "unit": str(unit) if unit is not None else "count",
        "tags": {str(key): val for key, val in (tags or {}).items()},
        "scope": scope,
        "timestamp": float(timestamp if timestamp is not None else time.time()),
    }
    telemetry = context.setdefault("telemetry", {})
    metrics = telemetry.setdefault("metrics", [])
    metrics.append(entry)
    return entry


def _record_runtime_event(
    context: Optional[Dict[str, Any]],
    *,
    event: str,
    level: str = "info",
    message: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None,
    log: bool = True,
) -> Dict[str, Any]:
    entry = {
        "event": str(event),
        "level": str(level or "info").lower(),
        "message": str(message or event),
        "timestamp": float(timestamp if timestamp is not None else time.time()),
        "data": dict(data or {}),
    }
    if isinstance(context, dict) and _observability_enabled(context, "events"):
        telemetry = context.setdefault("telemetry", {})
        events = telemetry.setdefault("events", [])
        events.append(entry)
    if log and _observability_enabled(context, "events"):
        level_name = entry["level"].upper()
        numeric_level = getattr(logging, level_name, logging.INFO)
        try:
            logger.log(
                numeric_level,
                entry["message"],
                extra={
                    "namel3ss_event": entry["event"],
                    "namel3ss_data": entry["data"],
                },
            )
        except Exception:
            logger.log(numeric_level, entry["message"])
    return entry


def _collect_runtime_errors(
    context: Dict[str, Any],
    scope: Optional[str] = None,
    *,
    consume: bool = True,
) -> List[Dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    errors = [entry for entry in context.get("errors", []) if isinstance(entry, dict)]
    if not errors:
        return []
    if scope is None:
        selected = list(errors)
        if consume:
            context.pop("errors", None)
        return selected
    selected: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for entry in errors:
        if entry.get("scope") == scope:
            selected.append(entry)
        else:
            remaining.append(entry)
    if consume:
        if remaining:
            context["errors"] = remaining
        else:
            context.pop("errors", None)
    return selected


_GLOBAL_MEMORY_STORE: Dict[str, List[Dict[str, Any]]] = {}


def _build_memory_runtime_state(
    existing_state: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    definitions = globals().get("AI_MEMORIES", {})
    state: Dict[str, Dict[str, Any]] = {}
    for name, spec in definitions.items():
        scope = str(spec.get("scope") or "session").lower()
        kind = str(spec.get("kind") or "list").lower()
        max_items_raw = spec.get("max_items")
        try:
            max_items = int(max_items_raw) if max_items_raw is not None else None
            if max_items is not None and max_items <= 0:
                max_items = None
        except Exception:
            max_items = None
        config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
        if scope == "global":
            entries = _GLOBAL_MEMORY_STORE.setdefault(name, [])
        else:
            existing_entries = []
            if existing_state and name in existing_state:
                existing_entries = list(existing_state[name].get("entries") or [])
            entries = existing_entries
        state[name] = {
            "scope": scope,
            "kind": kind,
            "max_items": max_items,
            "config": dict(config),
            "entries": entries,
        }
    return state


def _ensure_memory_state(context: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(context, dict):
        return _build_memory_runtime_state()
    memory_state = context.get("memory_state")
    if isinstance(memory_state, dict):
        return memory_state
    memory_state = _build_memory_runtime_state()
    context["memory_state"] = memory_state
    return memory_state


def _memory_snapshot(
    memory_state: Dict[str, Dict[str, Any]],
    names: Iterable[str],
) -> Dict[str, List[Dict[str, Any]]]:
    snapshot: Dict[str, List[Dict[str, Any]]] = {}
    for name in names:
        store = memory_state.get(name)
        if not store:
            continue
        snapshot[name] = copy.deepcopy(store.get("entries") or [])
    return snapshot


def _normalize_memory_entry(kind: str, value: Any) -> Dict[str, Any]:
    timestamp = time.time()
    if kind == "conversation":
        if isinstance(value, dict):
            entry = dict(value)
        else:
            entry = {"content": value}
        entry.setdefault("role", entry.get("role") or "system")
        entry["timestamp"] = timestamp
        return entry
    if kind == "key_value":
        entry: Dict[str, Any] = {}
        if isinstance(value, dict):
            if "key" in value:
                entry["key"] = value["key"]
            entry["value"] = value.get("value", value)
        else:
            entry["value"] = value
        entry["timestamp"] = timestamp
        return entry
    return {"value": value, "timestamp": timestamp}


def _write_memory_entries(
    memory_state: Dict[str, Dict[str, Any]],
    names: Iterable[str],
    value: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> None:
    for name in names:
        store = memory_state.get(name)
        if not store:
            if context is not None:
                _record_runtime_error(
                    context,
                    code="memory.undefined",
                    message=f"Memory '{name}' is not defined",
                    scope="memory",
                    detail=source,
                )
            else:
                logger.warning("Memory '%s' is not defined", name)
            continue
        kind = str(store.get("kind") or "list").lower()
        if kind == "vector":
            if context is not None:
                _record_runtime_error(
                    context,
                    code="memory.unsupported_kind",
                    message=f"Memory '{name}' uses kind 'vector' without a configured backend",
                    scope="memory",
                    detail=source,
                )
            else:
                logger.error("Memory '%s' uses kind 'vector' without a configured backend", name)
            raise ValueError(f"Memory '{name}' uses unsupported kind 'vector' without a configured backend")
        entries = store.setdefault("entries", [])
        entries.append(_normalize_memory_entry(kind, value))
        max_items = store.get("max_items")
        if isinstance(max_items, int) and max_items > 0:
            while len(entries) > max_items:
                entries.pop(0)


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
    context.setdefault("models", MODEL_REGISTRY)
    context.setdefault("connectors", AI_CONNECTORS)
    context.setdefault("ai_models", AI_MODELS)
    context.setdefault("prompts", AI_PROMPTS)
    context.setdefault("templates", AI_TEMPLATES)
    context.setdefault("chains", AI_CHAINS)
    context.setdefault("experiments", AI_EXPERIMENTS)
    memory_state = _ensure_memory_state(context)
    context.setdefault("memory_state", memory_state)
    context.setdefault("memory", memory_state)
    context.setdefault("memory_definitions", AI_MEMORIES)
    context.setdefault("call_python_model", call_python_model)
    context.setdefault("call_llm_connector", call_llm_connector)
    context.setdefault("run_prompt", run_prompt)
    context.setdefault("run_chain", run_chain)
    context.setdefault("evaluate_experiment", evaluate_experiment)
    context.setdefault("predict", predict)
    context.setdefault("datasets", DATASETS)
    context.setdefault("datasets_data", {})
    context.setdefault("frames", FRAMES)
    if AI_CONNECTORS:
        tool_instances: Dict[str, Any] = {}
        for connector_name, spec in AI_CONNECTORS.items():
            if not isinstance(spec, dict):
                continue
            tool_instances[connector_name] = _instantiate_tool_plugin(connector_name, spec, context)
        if tool_instances:
            context.setdefault("tools", {}).update(tool_instances)
    evaluator_instances: Dict[str, Any] = {}
    for evaluator_name, spec in EVALUATORS.items():
        if not isinstance(spec, dict):
            continue
        evaluator_instances[evaluator_name] = _instantiate_evaluator_plugin(evaluator_name, spec)
    if evaluator_instances:
        context.setdefault("evaluators", {}).update(evaluator_instances)
    else:
        context.setdefault("evaluators", {})
    context.setdefault("guardrails", GUARDRAILS)
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
            resolved = _resolve_context_path(context, [token], None)
            if resolved is not None:
                return str(resolved)
            vars_context = context.get("vars") if isinstance(context.get("vars"), dict) else None
            if vars_context is not None:
                value = _resolve_context_path(vars_context, [token], None)
                if value is not None:
                    return str(value)
            final_value = _resolve_context_path(context, [token], "")
            return "" if final_value is None else str(final_value)
        return TEMPLATE_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]
    if isinstance(value, dict):
        return {
            key: _render_template_value(item, context) for key, item in value.items()
        }
    return value


def register_connector_driver(
    connector_type: str,
    handler: Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Any]],
) -> None:
    if not connector_type or handler is None:
        return
    CONNECTOR_DRIVERS[connector_type.lower()] = handler


def register_dataset_transform(
    transform_type: str,
    handler: Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]],
) -> None:
    if not transform_type or handler is None:
        return
    DATASET_TRANSFORMS[transform_type.lower()] = handler


_PLUGIN_CATEGORY_ALIASES: Dict[str, str] = {
    "llm": "llm_provider",
    "llm_provider": "llm_provider",
    "vector": "vector_store",
    "vector_store": "vector_store",
    "embedding": "embedding_provider",
    "embedding_provider": "embedding_provider",
    "graph": "graph_db",
    "graph_db": "graph_db",
    "custom": "custom_tool",
    "custom_tool": "custom_tool",
    "tool": "custom_tool",
}


def _determine_connector_category(spec: Dict[str, Any]) -> str:
    raw = spec.get("category") or spec.get("type")
    key = str(raw or "").strip().lower()
    if not key:
        return ""
    return _PLUGIN_CATEGORY_ALIASES.get(key, key)


def _instantiate_tool_plugin(name: str, spec: Dict[str, Any], context: Dict[str, Any]) -> Any:
    category = _determine_connector_category(spec)
    provider_raw = spec.get("provider") or (spec.get("config") or {}).get("provider")
    provider = str(provider_raw or "").strip()
    if not category or not provider:
        raise RuntimeError(f"Connector '{name}' is missing a plugin category or provider.")
    try:
        plugin_cls = get_plugin(category, provider)
    except PluginRegistryError as exc:
        message = f"Connector '{name}' references unknown plugin '{provider}' in category '{category}'."
        logger.error(message)
        raise RuntimeError(message) from exc
    plugin = plugin_cls()
    config_raw = spec.get("config", {})
    config_resolved = _resolve_placeholders(config_raw, context)
    if not isinstance(config_resolved, dict):
        config_resolved = config_raw if isinstance(config_raw, dict) else {}
    try:
        plugin.configure(dict(config_resolved))
    except Exception as exc:
        message = f"Failed to configure connector '{name}': {exc}"
        logger.error(message)
        raise RuntimeError(message) from exc
    return plugin


def _instantiate_evaluator_plugin(name: str, spec: Dict[str, Any]) -> Any:
    provider = str(spec.get("provider") or "").strip()
    if not provider:
        raise RuntimeError(f"Evaluator '{name}' is missing a provider.")
    try:
        plugin_cls = get_plugin(PLUGIN_CATEGORY_EVALUATOR, provider)
    except PluginRegistryError as exc:
        message = f"Evaluator '{name}' references unknown plugin '{provider}'."
        logger.error(message)
        raise RuntimeError(message) from exc
    plugin = plugin_cls()
    config_raw = spec.get("config", {})
    config_payload = dict(config_raw) if isinstance(config_raw, dict) else {}
    try:
        plugin.configure(config_payload)
    except Exception as exc:
        message = f"Failed to configure evaluator '{name}': {exc}"
        logger.error(message)
        raise RuntimeError(message) from exc
    return plugin


class BreakFlow(Exception):
    """Internal signal to indicate a ``break`` statement was encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.components: List[Dict[str, Any]] = []

    def extend(self, items: Iterable[Dict[str, Any]]) -> None:
        if not items:
            return
        self.components.extend(items)


class ContinueFlow(Exception):
    """Internal signal to indicate a ``continue`` statement was encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.components: List[Dict[str, Any]] = []

    def extend(self, items: Iterable[Dict[str, Any]]) -> None:
        if not items:
            return
        self.components.extend(items)


class ScopeFrame:
    """Hierarchical scope for storing variables during page rendering."""

    def __init__(self, parent: Optional['ScopeFrame'] = None) -> None:
        self.parent = parent
        self._values: Dict[str, Any] = {}

    def child(self) -> 'ScopeFrame':
        return ScopeFrame(self)

    def contains(self, name: str) -> bool:
        if name in self._values:
            return True
        if self.parent is not None:
            return self.parent.contains(name)
        return False

    def get(self, name: str, default: Any = None) -> Any:
        if name in self._values:
            return self._values[name]
        if self.parent is not None:
            return self.parent.get(name, default)
        return default

    def assign(self, name: str, value: Any) -> None:
        if name in self._values:
            self._values[name] = value
            return
        if self.parent is not None and self.parent.contains(name):
            self.parent.assign(name, value)
            return
        self._values[name] = value

    def set(self, name: str, value: Any) -> None:
        self.assign(name, value)

    def bind(self, name: str, value: Any) -> None:
        self._values[name] = value

    def remove_local(self, name: str) -> None:
        self._values.pop(name, None)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.parent is not None:
            data.update(self.parent.to_dict())
        data.update(self._values)
        return data


_MISSING = object()

_RUNTIME_CALLABLES: Dict[str, Callable[..., Any]] = {
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "abs": abs,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "range": range,
}


def _assign_variable(scope: ScopeFrame, context: Dict[str, Any], name: str, value: Any) -> None:
    scope.assign(name, value)
    context.setdefault("vars", {})[name] = value


def _bind_variable(scope: ScopeFrame, context: Dict[str, Any], name: str, value: Any) -> None:
    scope.bind(name, value)
    context.setdefault("vars", {})[name] = value


def _restore_variable(context: Dict[str, Any], name: str, previous: Any) -> None:
    vars_map = context.setdefault("vars", {})
    if previous is _MISSING:
        vars_map.pop(name, None)
    else:
        vars_map[name] = previous


def _runtime_truthy(value: Any) -> bool:
    return bool(value)


def _clone_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _ensure_numeric(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _resolve_option_dict(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(raw or {})

    '''
).strip()

__all__ = ['CONTEXT_SECTION']
