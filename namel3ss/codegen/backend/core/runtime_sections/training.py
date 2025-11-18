from __future__ import annotations

from textwrap import dedent

TRAINING_SECTION = dedent(
    '''

def list_training_jobs() -> List[str]:
    return sorted(TRAINING_JOBS.keys())


def available_training_backends() -> List[str]:
    return list(_registered_training_backends())


def get_training_job(name: str) -> Optional[Dict[str, Any]]:
    key = str(name)
    spec = TRAINING_JOBS.get(key)
    if not spec:
        return None
    return copy.deepcopy(spec)


def resolve_training_job_plan(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    key = str(name)
    spec = TRAINING_JOBS.get(key)
    if not spec:
        raise ValueError(f"Training job '{name}' is not defined")
    env_map = _training_environment(context)
    payload_map = dict(payload or {})
    overrides_map = dict(overrides or {})
    return _resolve_training_plan_impl(spec, payload_map, overrides_map, env=env_map)


def training_job_history(name: str, limit: int = 20) -> List[Dict[str, Any]]:
    history = TRAINING_JOB_HISTORY.get(str(name), [])
    if limit is not None and limit > 0:
        selected = history[-limit:]
    else:
        selected = history
    return [copy.deepcopy(entry) for entry in selected]


def run_training_job(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    key = str(name)
    spec = TRAINING_JOBS.get(key)
    runtime_context = context if isinstance(context, dict) else build_context(None)
    if not spec:
        return {
            "status": "error",
            "error": "training_job_not_found",
            "detail": f"Training job '{name}' is not defined",
            "job": key,
        }
    payload_map = dict(payload or {})
    overrides_map = dict(overrides or {})
    try:
        plan = resolve_training_job_plan(key, payload_map, overrides_map, context=runtime_context)
    except Exception as exc:
        logger.exception("Failed to resolve training plan for job %s", key)
        return {
            "status": "error",
            "error": "training_plan_failed",
            "detail": str(exc),
            "job": key,
        }

    backend_context = _prepare_training_context(spec, runtime_context)
    backend_context.setdefault("payload", payload_map)
    backend_context.setdefault("overrides", overrides_map)
    backend_context.setdefault("job", spec.get("name", key))

    started = time.time()
    backend_name = plan.get("backend")
    try:
        backend = _resolve_training_backend_impl(backend_name)
        result_payload = backend.run(plan, backend_context)
    except Exception as exc:
        logger.exception("Training backend '%s' failed for job '%s'", backend_name, key)
        return {
            "status": "error",
            "error": "training_backend_failed",
            "detail": str(exc),
            "job": spec.get("name", key),
            "backend": backend_name,
        }

    result = dict(result_payload or {}) if isinstance(result_payload, dict) else {}
    result.setdefault("status", "ok")
    result.setdefault("job", spec.get("name", key))
    result.setdefault("backend", backend_name or "local")
    result.setdefault("model", spec.get("model"))
    result.setdefault("dataset", spec.get("dataset"))
    result.setdefault("hyperparameters", plan.get("hyperparameters", {}))
    result.setdefault("resources", plan.get("resources", {}))
    result.setdefault("metadata", plan.get("metadata", {}))

    duration_ms = float(round((time.time() - started) * 1000.0, 3))
    _record_training_history(key, result, duration_ms)
    _record_runtime_event(
        runtime_context,
        event="training_job.complete",
        level="info" if result.get("status") == "ok" else "warning",
        data={
            "job": key,
            "backend": result.get("backend"),
            "status": result.get("status"),
            "duration_ms": duration_ms,
        },
    )
    return result


def _training_environment(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    if isinstance(context, dict):
        context_env = context.get("env") or {}
        if isinstance(context_env, dict):
            for key, value in context_env.items():
                if value is not None:
                    env[str(key)] = str(value)
    for key in ENV_KEYS:
        if key not in env:
            value = os.getenv(key)
            if value is not None:
                env[key] = value
    return env


def _prepare_training_context(
    spec: Dict[str, Any],
    runtime_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    dataset_snapshot = _training_dataset_snapshot(spec.get("dataset"))
    if dataset_snapshot:
        context.update(dataset_snapshot)
    if runtime_context:
        context.setdefault("context", runtime_context)
    return context


def _training_dataset_snapshot(dataset_name: Optional[str]) -> Dict[str, Any]:
    if not dataset_name:
        return {}
    dataset_spec = DATASETS.get(dataset_name) or FRAMES.get(dataset_name)
    if not isinstance(dataset_spec, dict):
        return {}
    snapshot: Dict[str, Any] = {"dataset_name": dataset_name}
    sample_rows: Optional[List[Dict[str, Any]]] = None
    rows_field = dataset_spec.get("sample_rows")
    if isinstance(rows_field, list):
        sample_rows = rows_field
    elif isinstance(dataset_spec.get("examples"), list):
        sample_rows = dataset_spec.get("examples")
    if sample_rows:
        snapshot["dataset_rows"] = _clone_rows(sample_rows)
        snapshot["dataset_size"] = len(snapshot["dataset_rows"])
    profile = dataset_spec.get("profile")
    if isinstance(profile, dict):
        snapshot["dataset_profile"] = dict(profile)
        snapshot.setdefault("dataset_size", profile.get("row_count"))
    schema = dataset_spec.get("schema")
    if isinstance(schema, list):
        snapshot["dataset_schema"] = copy.deepcopy(schema)
    elif isinstance(dataset_spec.get("columns"), list):
        snapshot["dataset_schema"] = copy.deepcopy(dataset_spec.get("columns"))
    return snapshot


def _record_training_history(name: str, result: Dict[str, Any], duration_ms: float) -> None:
    entry = copy.deepcopy(result)
    entry.setdefault("ts", time.time())
    entry.setdefault("duration_ms", duration_ms)
    history = TRAINING_JOB_HISTORY.setdefault(name, [])
    history.append(entry)
    if len(history) > 50:
        del history[:-50]

'''
).strip()

__all__ = ["TRAINING_SECTION"]
