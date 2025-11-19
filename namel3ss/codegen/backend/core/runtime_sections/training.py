from __future__ import annotations

from textwrap import dedent

TRAINING_SECTION = dedent(
    '''

def list_training_jobs() -> List[str]:
    """List all available training job names."""
    return sorted(TRAINING_JOBS.keys())


def list_tuning_jobs() -> List[str]:
    """List all available tuning job names."""
    return sorted(TUNING_JOBS.keys())


def available_training_backends() -> List[str]:
    """List registered training backend names."""
    return list(_registered_training_backends())


def get_training_job(name: str) -> Optional[Dict[str, Any]]:
    """Get training job spec by name."""
    key = str(name)
    spec = TRAINING_JOBS.get(key)
    if not spec:
        return None
    return copy.deepcopy(spec)


def get_tuning_job(name: str) -> Optional[Dict[str, Any]]:
    """Get tuning job spec by name."""
    key = str(name)
    spec = TUNING_JOBS.get(key)
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
    """Resolve a training plan with payload and overrides."""
    key = str(name)
    spec = TRAINING_JOBS.get(key)
    if not spec:
        raise ValueError(f"Training job '{name}' is not defined")
    env_map = _training_environment(context)
    payload_map = dict(payload or {})
    overrides_map = dict(overrides or {})
    return _resolve_training_plan_impl(spec, payload_map, overrides_map, env=env_map)


def training_job_history(name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get training job execution history."""
    history = TRAINING_JOB_HISTORY.get(str(name), [])
    if limit is not None and limit > 0:
        selected = history[-limit:]
    else:
        selected = history
    return [copy.deepcopy(entry) for entry in selected]


def tuning_job_history(name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get tuning job execution history."""
    history = TUNING_JOB_HISTORY.get(str(name), [])
    if limit is not None and limit > 0:
        selected = history[-limit:]
    else:
        selected = history
    return [copy.deepcopy(entry) for entry in selected]


async def run_training_job(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a training job with full dataset loading, training, and model registry persistence."""
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

    # Load dataset
    dataset_name = spec.get("dataset")
    if not dataset_name:
        return {
            "status": "error",
            "error": "missing_dataset",
            "detail": "Training job must specify a dataset",
            "job": key,
        }
    
    try:
        dataset_context = await _load_training_dataset(dataset_name, spec, session, runtime_context)
    except Exception as exc:
        logger.exception("Failed to load dataset for training job %s", key)
        return {
            "status": "error",
            "error": "dataset_load_failed",
            "detail": str(exc),
            "job": key,
        }

    # Prepare backend context
    backend_context = _prepare_training_context(spec, runtime_context, dataset_context)
    backend_context.setdefault("payload", payload_map)
    backend_context.setdefault("overrides", overrides_map)
    backend_context.setdefault("job", spec.get("name", key))

    started = time.time()
    backend_name = plan.get("backend", "local")
    framework = spec.get("framework", backend_name)
    
    try:
        backend = _resolve_training_backend_impl(framework or backend_name)
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
    
    # Persist trained model to registry if training succeeded
    if result.get("status") == "ok":
        try:
            registry_key = await _persist_trained_model(spec, result, session)
            result["registry_key"] = registry_key
        except Exception as exc:
            logger.warning("Failed to persist trained model to registry: %s", exc)
            result["registry_warning"] = str(exc)
    
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


async def run_tuning_job(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a hyperparameter tuning job with trial tracking and best model selection."""
    key = str(name)
    spec = TUNING_JOBS.get(key)
    runtime_context = context if isinstance(context, dict) else build_context(None)
    
    if not spec:
        return {
            "status": "error",
            "error": "tuning_job_not_found",
            "detail": f"Tuning job '{name}' is not defined",
            "job": key,
        }
    
    # Get base training job
    training_job_name = spec.get("training_job")
    if not training_job_name or training_job_name not in TRAINING_JOBS:
        return {
            "status": "error",
            "error": "invalid_training_job",
            "detail": f"Tuning job '{name}' references invalid training job '{training_job_name}'",
            "job": key,
        }
    
    started = time.time()
    strategy = spec.get("strategy", "grid")
    max_trials = spec.get("max_trials", 10)
    parallel_trials = spec.get("parallel_trials", 1)
    search_space = spec.get("search_space", {})
    objective_metric = spec.get("objective_metric", "accuracy")
    early_stopping = spec.get("early_stopping")
    
    try:
        trials = await _execute_tuning_trials(
            training_job_name=training_job_name,
            search_space=search_space,
            strategy=strategy,
            max_trials=max_trials,
            parallel_trials=parallel_trials,
            objective_metric=objective_metric,
            early_stopping=early_stopping,
            payload=payload,
            overrides=overrides,
            context=runtime_context,
            session=session,
        )
    except Exception as exc:
        logger.exception("Tuning job '%s' failed", key)
        return {
            "status": "error",
            "error": "tuning_execution_failed",
            "detail": str(exc),
            "job": key,
        }
    
    # Find best trial
    best_trial = _find_best_trial(trials, objective_metric)
    
    duration_ms = float(round((time.time() - started) * 1000.0, 3))
    
    result = {
        "status": "ok",
        "job": key,
        "training_job": training_job_name,
        "strategy": strategy,
        "trials": trials,
        "total_trials": len(trials),
        "best_trial": best_trial,
        "best_hyperparameters": best_trial.get("hyperparameters", {}) if best_trial else {},
        "best_metrics": best_trial.get("metrics", {}) if best_trial else {},
        "objective_metric": objective_metric,
        "duration_ms": duration_ms,
        "metadata": spec.get("metadata", {}),
    }
    
    # Persist best model to registry
    if best_trial and best_trial.get("status") == "ok":
        try:
            training_spec = TRAINING_JOBS.get(training_job_name, {})
            registry_key = await _persist_trained_model(training_spec, best_trial, session)
            result["best_model_registry_key"] = registry_key
        except Exception as exc:
            logger.warning("Failed to persist best tuned model to registry: %s", exc)
            result["registry_warning"] = str(exc)
    
    _record_tuning_history(key, result, duration_ms)
    _record_runtime_event(
        runtime_context,
        event="tuning_job.complete",
        level="info",
        data={
            "job": key,
            "total_trials": len(trials),
            "duration_ms": duration_ms,
        },
    )
    return result


async def _load_training_dataset(
    dataset_name: str,
    training_spec: Dict[str, Any],
    session: Optional[Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Load dataset for training with proper async handling."""
    dataset_spec = DATASETS.get(dataset_name) or FRAMES.get(dataset_name)
    if not dataset_spec:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    # Use existing dataset loading infrastructure
    if dataset_name in DATASETS:
        rows = await load_dataset(dataset_name, session=session, context=context)
    else:
        rows = await load_frame(dataset_name, session=session, context=context)
    
    return {
        "dataset_name": dataset_name,
        "dataset_rows": rows,
        "dataset_size": len(rows) if rows else 0,
        "dataset_schema": dataset_spec.get("schema", []),
    }


async def _execute_tuning_trials(
    training_job_name: str,
    search_space: Dict[str, Any],
    strategy: str,
    max_trials: int,
    parallel_trials: int,
    objective_metric: str,
    early_stopping: Optional[Dict[str, Any]],
    payload: Optional[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]],
    context: Dict[str, Any],
    session: Optional[Any],
) -> List[Dict[str, Any]]:
    """Execute tuning trials with specified search strategy."""
    trials = []
    
    # Generate trial configurations
    trial_configs = _generate_trial_configs(search_space, strategy, max_trials)
    
    # Execute trials (sequential for now; parallel execution would require asyncio.gather)
    for trial_idx, trial_hyperparams in enumerate(trial_configs):
        trial_overrides = dict(overrides or {})
        trial_overrides["hyperparameters"] = trial_hyperparams
        
        logger.info("Executing tuning trial %d/%d with hyperparameters: %s", 
                   trial_idx + 1, len(trial_configs), trial_hyperparams)
        
        trial_result = await run_training_job(
            name=training_job_name,
            payload=payload,
            overrides=trial_overrides,
            context=context,
            session=session,
        )
        
        trial_result["trial_index"] = trial_idx
        trial_result["hyperparameters"] = trial_hyperparams
        trials.append(trial_result)
        
        # Check early stopping
        if early_stopping and _should_stop_early(trials, early_stopping, objective_metric):
            logger.info("Early stopping triggered after %d trials", len(trials))
            break
    
    return trials


def _generate_trial_configs(
    search_space: Dict[str, Any],
    strategy: str,
    max_trials: int,
) -> List[Dict[str, Any]]:
    """Generate hyperparameter configurations for trials based on strategy."""
    import random
    
    if strategy == "grid":
        # Grid search: all combinations
        return _grid_search_configs(search_space, max_trials)
    elif strategy == "random":
        # Random search: random sampling
        return _random_search_configs(search_space, max_trials)
    else:
        # Default to random
        return _random_search_configs(search_space, max_trials)


def _grid_search_configs(search_space: Dict[str, Any], max_trials: int) -> List[Dict[str, Any]]:
    """Generate grid search configurations."""
    import itertools
    
    param_names = list(search_space.keys())
    param_values = []
    
    for param_name in param_names:
        spec = search_space[param_name]
        if isinstance(spec, dict):
            if "values" in spec:
                param_values.append(spec["values"])
            elif "min" in spec and "max" in spec:
                # Sample uniformly
                min_val = spec["min"]
                max_val = spec["max"]
                step = spec.get("step", (max_val - min_val) / 5.0)
                values = []
                current = min_val
                while current <= max_val:
                    values.append(current)
                    current += step
                param_values.append(values)
            else:
                param_values.append([None])
        else:
            param_values.append([spec])
    
    # Generate all combinations
    all_configs = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        all_configs.append(config)
        if len(all_configs) >= max_trials:
            break
    
    return all_configs[:max_trials]


def _random_search_configs(search_space: Dict[str, Any], max_trials: int) -> List[Dict[str, Any]]:
    """Generate random search configurations."""
    import random
    
    configs = []
    for _ in range(max_trials):
        config = {}
        for param_name, spec in search_space.items():
            if isinstance(spec, dict):
                if "values" in spec:
                    config[param_name] = random.choice(spec["values"])
                elif "min" in spec and "max" in spec:
                    min_val = spec["min"]
                    max_val = spec["max"]
                    if spec.get("log", False):
                        # Log-uniform sampling
                        import math
                        log_min = math.log(max(min_val, 1e-10))
                        log_max = math.log(max_val)
                        config[param_name] = math.exp(random.uniform(log_min, log_max))
                    else:
                        # Uniform sampling
                        config[param_name] = random.uniform(min_val, max_val)
                else:
                    config[param_name] = None
            else:
                config[param_name] = spec
        configs.append(config)
    
    return configs


def _should_stop_early(
    trials: List[Dict[str, Any]],
    early_stopping: Dict[str, Any],
    objective_metric: str,
) -> bool:
    """Check if early stopping criteria are met."""
    patience = early_stopping.get("patience", 5)
    min_delta = early_stopping.get("min_delta", 0.0)
    mode = early_stopping.get("mode", "max")
    
    if len(trials) < patience:
        return False
    
    # Get recent metric values
    recent_metrics = []
    for trial in trials[-patience:]:
        metrics = trial.get("metrics", {})
        if objective_metric in metrics:
            recent_metrics.append(metrics[objective_metric])
    
    if len(recent_metrics) < patience:
        return False
    
    # Check if no improvement
    if mode == "max":
        best_recent = max(recent_metrics)
        is_improving = any(m >= best_recent + min_delta for m in recent_metrics[1:])
    else:
        best_recent = min(recent_metrics)
        is_improving = any(m <= best_recent - min_delta for m in recent_metrics[1:])
    
    return not is_improving


def _find_best_trial(trials: List[Dict[str, Any]], objective_metric: str) -> Optional[Dict[str, Any]]:
    """Find the best trial based on objective metric."""
    valid_trials = [t for t in trials if t.get("status") == "ok" and objective_metric in t.get("metrics", {})]
    if not valid_trials:
        return None
    
    # Assume higher is better for accuracy-like metrics, lower for loss-like metrics
    if "loss" in objective_metric.lower() or "error" in objective_metric.lower():
        return min(valid_trials, key=lambda t: t["metrics"][objective_metric])
    else:
        return max(valid_trials, key=lambda t: t["metrics"][objective_metric])


async def _persist_trained_model(
    training_spec: Dict[str, Any],
    result: Dict[str, Any],
    session: Optional[Any],
) -> str:
    """Persist trained model to the model registry."""
    model_name = training_spec.get("model", "unnamed_model")
    job_name = training_spec.get("name", "unnamed_job")
    
    # Generate registry key
    import time
    timestamp = int(time.time())
    version = f"v{timestamp}"
    registry_key = f"{model_name}_{job_name}_{version}"
    
    # Prepare registry entry
    artifacts = result.get("artifacts", {})
    metrics = result.get("metrics", {})
    metadata = result.get("metadata", {})
    
    registry_entry = {
        "type": metadata.get("model_type", "sklearn"),
        "framework": metadata.get("framework", training_spec.get("framework", "sklearn")),
        "version": version,
        "metrics": metrics,
        "metadata": {
            **metadata,
            "training_job": job_name,
            "dataset": training_spec.get("dataset"),
            "target": training_spec.get("target"),
            "features": training_spec.get("features", []),
            "hyperparameters": result.get("hyperparameters", {}),
            "trained_at": timestamp,
        },
    }
    
    # Add model artifacts if available
    if "model_object" in artifacts:
        registry_entry["metadata"]["model_object"] = artifacts["model_object"]
    if "feature_names" in artifacts:
        registry_entry["metadata"]["feature_names"] = artifacts["feature_names"]
    if "target_name" in artifacts:
        registry_entry["metadata"]["target_name"] = artifacts["target_name"]
    
    # Store in MODEL_REGISTRY
    MODEL_REGISTRY[registry_key] = registry_entry
    
    logger.info("Persisted trained model to registry: %s", registry_key)
    return registry_key


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
    dataset_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    
    # Add dataset context
    if dataset_context:
        context.update(dataset_context)
    
    # Add training-specific metadata
    context["metadata"] = {
        **spec.get("metadata", {}),
        "target": spec.get("target"),
        "features": spec.get("features", []),
        "split": spec.get("split", {}),
        "model_type": spec.get("metadata", {}).get("model_type", "RandomForestClassifier"),
    }
    
    if runtime_context:
        context.setdefault("context", runtime_context)
    
    return context


def _record_training_history(name: str, result: Dict[str, Any], duration_ms: float) -> None:
    entry = copy.deepcopy(result)
    entry.setdefault("ts", time.time())
    entry.setdefault("duration_ms", duration_ms)
    history = TRAINING_JOB_HISTORY.setdefault(name, [])
    history.append(entry)
    if len(history) > 50:
        del history[:-50]


def _record_tuning_history(name: str, result: Dict[str, Any], duration_ms: float) -> None:
    entry = copy.deepcopy(result)
    entry.setdefault("ts", time.time())
    entry.setdefault("duration_ms", duration_ms)
    history = TUNING_JOB_HISTORY.setdefault(name, [])
    history.append(entry)
    if len(history) > 50:
        del history[:-50]

'''
).strip()

__all__ = ["TRAINING_SECTION"]
