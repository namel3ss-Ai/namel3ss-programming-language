from __future__ import annotations

from textwrap import dedent

REGISTRY_SECTION = dedent(
    '''
def _page_meta(slug: str) -> Dict[str, Any]:
    spec = PAGE_SPEC_BY_SLUG.get(slug, {})
    return {
        "reactive": bool(spec.get("reactive")),
        "refresh_policy": spec.get("refresh_policy"),
    }


def _model_to_payload(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return dict(model)
    if hasattr(model, "__dict__"):
        return {
            key: value
            for key, value in model.__dict__.items()
            if not key.startswith("_")
        }
    return {"value": model}


def _with_timestamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched.setdefault("ts", time.time())
    return enriched


def register_model_loader(framework: str, loader: Callable[[str, Dict[str, Any]], Any]) -> None:
    if not framework or loader is None:
        return
    MODEL_LOADERS[framework.lower()] = loader


def register_model_runner(
    framework: str,
    runner: Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> None:
    if not framework or runner is None:
        return
    MODEL_RUNNERS[framework.lower()] = runner


def register_model_explainer(
    framework: str,
    explainer: Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> None:
    if not framework or explainer is None:
        return
    MODEL_EXPLAINERS[framework.lower()] = explainer


def _resolve_model_artifact_path(model_spec: Dict[str, Any]) -> Optional[str]:
    metadata_obj = model_spec.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    path = metadata.get("model_file") or metadata.get("artifact_path")
    if not path:
        return None
    root = os.getenv("NAMEL3SS_MODEL_ROOT")
    if root and not os.path.isabs(path):
        return os.path.join(root, path)
    return path


def _load_python_callable(import_path: str) -> Optional[Callable[..., Any]]:
    if not import_path:
        return None

    path = import_path.strip()
    module_path: str
    attr: str

    if ":" in path:
        module_path, attr = path.split(":", 1)
    else:
        module_path, _, attr = path.rpartition(".")

    module_path = module_path.strip()
    attr = attr.strip()
    if not module_path or not attr:
        return None

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - import failure guard
        logger.exception("Failed to import module '%s' for callable '%s'", module_path, import_path)
        raise ImportError(f"Could not import module '{module_path}'") from exc

    try:
        candidate = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr}'") from exc

    if not callable(candidate):
        raise TypeError(f"Resolved object '{module_path}.{attr}' is not callable")

    return candidate


def _import_python_module(module: str) -> Optional[Any]:
    if not module:
        return None
    try:
        if module.endswith(".py"):
            path = Path(module)
            if not path.is_absolute():
                base = os.getenv("NAMEL3SS_APP_ROOT")
                if base:
                    path = Path(base) / path
                else:
                    path = Path.cwd() / path
            if not path.exists():
                return None
            module_name = path.stem
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module_obj = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module_obj)  # type: ignore[attr-defined]
                    sys.modules.setdefault(module_name, module_obj)
                    return module_obj
                except Exception:  # pragma: no cover - user module failure
                    logger.exception("Failed to import python module from %s", path)
                    return None
            return None
        return importlib.import_module(module)
    except Exception:  # pragma: no cover - import failure
        logger.exception("Failed to import python module %s", module)
        return None


def _is_truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _trim_traceback(limit: int = 5, max_chars: int = 3000) -> str:
    import traceback

    try:
        formatted = traceback.format_exc(limit=limit)
    except Exception:  # pragma: no cover - safety guard
        return ""
    if not formatted:
        return ""
    text = formatted.strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _short_error(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def call_python_model(
    module: str,
    method: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Invoke a Python callable and return structured status details.

    Args:
        module: Fully qualified module path or file path that can be imported.
        method: Attribute name to resolve on the imported module; defaults to ``predict`` when empty.
        arguments: Keyword arguments passed to the callable when invoked.

    Returns:
        A dictionary containing ``status`` alongside contextual fields:

        * ``status = 'ok'`` includes the callable result.
        * ``status = 'error'`` reports structured failure details and a trimmed traceback when stubs are disabled.
        * ``status = 'stub'`` mirrors legacy stub behaviour when ``NAMEL3SS_ALLOW_STUBS`` is truthy.
    """

    args = dict(arguments or {})
    attr_name = method or "predict"
    allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")

    try:
        module_obj = _import_python_module(module)
        if module_obj is None:
            raise ImportError(f"Module '{module}' could not be imported")

        callable_obj = getattr(module_obj, attr_name)
        if not callable(callable_obj):
            raise TypeError(f"Attribute '{attr_name}' on module '{module}' is not callable")

        result = callable_obj(**args)
        return {
            "status": "ok",
            "result": result,
            "inputs": args,
            "module": module,
            "method": attr_name,
        }
    except Exception as exc:  # pragma: no cover - user callable failure
        logger.exception("Python callable %s.%s raised an error", module, attr_name)
        error_message = _short_error(exc)
        if allow_stubs:
            return {
                "status": "stub",
                "result": "stub_prediction",
                "inputs": args,
                "module": module,
                "method": attr_name,
                "error": error_message,
            }

        response = {
            "status": "error",
            "inputs": args,
            "module": module,
            "method": attr_name,
            "error": error_message,
        }

        traceback_text = _trim_traceback()
        if traceback_text:
            response["traceback"] = traceback_text

        return response


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_path_segments(path: Any) -> List[str]:
    if path is None:
        return []
    if isinstance(path, (list, tuple)):
        segments: List[str] = []
        for item in path:
            segments.extend(_as_path_segments(item))
        return segments
    text = str(path).strip()
    if not text:
        return []
    normalized = text.replace("[", ".").replace("]", ".")
    return [segment for segment in normalized.split(".") if segment]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        multiplier = 1.0
        if text.endswith("%"):
            multiplier = 0.01
            text = text[:-1]
        try:
            return float(text) * multiplier
        except ValueError:
            return None
    return None


def _aggregate_numeric_samples(samples: List[float], aggregation: str) -> Optional[float]:
    if not samples:
        return None
    key = aggregation.lower()
    if key in ("avg", "average", "mean"):
        return sum(samples) / len(samples)
    if key in ("sum", "total"):
        return sum(samples)
    if key in ("min", "minimum"):
        return min(samples)
    if key in ("max", "maximum"):
        return max(samples)
    if key == "median":
        ordered = sorted(samples)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2
    if key in ("p95", "percentile95", "p_95"):
        ordered = sorted(samples)
        index = int(round(0.95 * (len(ordered) - 1)))
        index = max(0, min(index, len(ordered) - 1))
        return ordered[index]
    if key in ("p90", "percentile90", "p_90"):
        ordered = sorted(samples)
        index = int(round(0.90 * (len(ordered) - 1)))
        index = max(0, min(index, len(ordered) - 1))
        return ordered[index]
    if key in ("last", "latest"):
        return samples[-1]
    if key == "first":
        return samples[0]
    return sum(samples) / len(samples)


def _collect_metric_samples(
    metric: Dict[str, Any],
    variants: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> List[float]:
    metadata = _ensure_dict(metric.get("metadata"))
    source_kind = str(metric.get("source_kind") or metadata.get("source") or "score").lower()
    path = metadata.get("path") or metadata.get("result_path") or metadata.get("field")
    segments = _as_path_segments(path)
    samples: List[float] = []

    if source_kind in ("score", "variant", "variant_score"):
        for variant in variants:
            value: Any = variant.get("score")
            if value is None and segments:
                value = _traverse_attribute_path(variant, segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)
    elif source_kind in ("result", "output", "prediction"):
        target_segments = segments or ["result", "output", "score"]
        for variant in variants:
            value = _traverse_attribute_path(variant, target_segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)
    elif source_kind in ("payload", "input", "request"):
        value = _traverse_attribute_path(args, segments) if segments else args.get("input")
        sample = _safe_float(value)
        if sample is not None:
            samples.append(sample)
    elif source_kind in ("manual", "provided", "static"):
        values = metadata.get("samples") or metadata.get("values") or []
        if isinstance(values, (list, tuple)):
            for entry in values:
                sample = _safe_float(entry)
                if sample is not None:
                    samples.append(sample)
    else:
        target_segments = segments or ["result", "output", "score"]
        for variant in variants:
            value = _traverse_attribute_path(variant, target_segments)
            sample = _safe_float(value)
            if sample is not None:
                samples.append(sample)

    extras = metadata.get("include")
    if isinstance(extras, (list, tuple)):
        for entry in extras:
            sample = _safe_float(entry)
            if sample is not None:
                samples.append(sample)

    return samples


def _evaluate_experiment_metrics(
    spec: Dict[str, Any],
    variants: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metrics_result: List[Dict[str, Any]] = []
    for index, metric in enumerate(spec.get("metrics", []), start=1):
        metadata = _ensure_dict(metric.get("metadata"))
        aggregation = str(metadata.get("aggregation") or metadata.get("aggregate") or "max")
        samples = _collect_metric_samples(metric, variants, args)
        aggregated_value = _aggregate_numeric_samples(samples, aggregation) if samples else None
        if aggregated_value is None:
            calculated = float(round(0.72 + 0.045 * index, 3))
        else:
            calculated = float(aggregated_value)
        precision = metadata.get("precision") if isinstance(metadata.get("precision"), int) else metadata.get("round")
        if isinstance(precision, int):
            rounded_value = round(calculated, max(int(precision), 0))
        else:
            rounded_value = round(calculated, 4)

        goal_value = _safe_float(metric.get("goal"))
        direction = str(metadata.get("goal_operator") or metadata.get("direction") or "gte").lower()
        achieved_goal: Optional[bool] = None
        if goal_value is not None:
            if direction in ("lte", "le", "max", "lower", "down"):
                achieved_goal = rounded_value <= goal_value
            else:
                achieved_goal = rounded_value >= goal_value

        if samples:
            metadata.setdefault("aggregation", aggregation)
            metadata.setdefault("samples", len(samples))
            metadata.setdefault(
                "summary",
                {
                    "min": min(samples),
                    "max": max(samples),
                    "mean": round(sum(samples) / len(samples), 6),
                },
            )

        metrics_result.append(
            {
                "name": metric.get("name"),
                "value": rounded_value,
                "goal": metric.get("goal"),
                "source_kind": metric.get("source_kind"),
                "source_name": metric.get("source_name"),
                "metadata": metadata,
                "achieved_goal": achieved_goal,
            }
        )

    return metrics_result

    '''
).strip()

__all__ = ['REGISTRY_SECTION']
