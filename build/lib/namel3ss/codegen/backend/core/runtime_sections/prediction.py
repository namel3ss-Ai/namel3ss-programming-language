from __future__ import annotations

from textwrap import dedent

PREDICTION_SECTION = dedent(
    '''
def get_model_spec(model_name: str) -> Dict[str, Any]:
    base = copy.deepcopy(MODEL_REGISTRY.get(model_name) or {})
    generated = MODELS.get(model_name) or {}
    registry_info = generated.get("registry") or {}

    if not base:
        base = {
            "type": generated.get("type", "unknown"),
            "framework": generated.get("engine") or generated.get("framework") or "unknown",
            "version": registry_info.get("version", "v1"),
            "metrics": registry_info.get("metrics", {}),
            "metadata": registry_info.get("metadata", {}),
        }
    else:
        base.setdefault("type", generated.get("type") or base.get("type") or "unknown")
        base.setdefault("framework", generated.get("engine") or base.get("framework") or "unknown")
        base.setdefault("version", registry_info.get("version") or base.get("version") or "v1")
        merged_metrics = dict(base.get("metrics") or {})
        merged_metrics.update(registry_info.get("metrics") or {})
        base["metrics"] = merged_metrics
        merged_metadata = dict(base.get("metadata") or {})
        merged_metadata.update(registry_info.get("metadata") or {})
        base["metadata"] = merged_metadata

    base.setdefault("metrics", {})
    base.setdefault("metadata", {})
    base["type"] = base.get("type") or generated.get("type") or "custom"
    base["framework"] = base.get("framework") or "unknown"
    base["version"] = base.get("version") or "v1"
    return base


def _load_model_instance(model_name: str, model_spec: Dict[str, Any]) -> Any:
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    framework = str(model_spec.get("framework") or "").lower()
    model_type = str(model_spec.get("type") or "").lower()
    metadata = model_spec.get("metadata") or {}
    custom_loader = None
    loader_path = metadata.get("loader")
    if loader_path:
        try:
            custom_loader = _load_python_callable(loader_path)
        except Exception:  # pragma: no cover - loader import failure
            logger.exception("Failed to import custom loader for %s", model_name)
            custom_loader = None
    loader = None
    loader = (
        custom_loader
        or MODEL_LOADERS.get(framework)
        or MODEL_LOADERS.get(model_type)
        or MODEL_LOADERS.get("generic")
    )
    instance = None
    if loader:
        try:
            instance = loader(model_name, model_spec)
        except Exception:  # pragma: no cover - loader failure
            logger.exception("Model loader failed for %s", model_name)
            instance = None
    MODEL_CACHE[model_name] = instance
    return instance


def _default_explanations(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "global_importances": {"feature_a": 0.7, "feature_b": 0.3},
        "local_explanations": [
            {"feature": "feature_a", "impact": +0.2},
        ],
        "visualizations": {
            "saliency_map": "base64://dummy_image_data",
            "attention": "base64://placeholder_heatmap",
        },
    }


def explain_prediction(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    framework = str(prediction.get("framework") or "").lower()
    model_name_key = str(prediction.get("model") or "").lower()
    metadata = prediction.get("spec_metadata") or {}
    custom_explainer = None
    explainer_path = metadata.get("explainer") if isinstance(metadata, dict) else None
    if explainer_path:
        try:
            custom_explainer = _load_python_callable(explainer_path)
        except Exception:  # pragma: no cover - explainer import failure
            logger.exception("Failed to import custom explainer for %s", model_name)
            custom_explainer = None
    explainer = (
        custom_explainer
        or MODEL_EXPLAINERS.get(framework)
        or MODEL_EXPLAINERS.get(model_name_key)
        or MODEL_EXPLAINERS.get("generic")
    )
    if explainer:
        try:
            value = explainer(model_name, payload, prediction)
            if isinstance(value, dict) and value:
                return value
        except Exception:  # pragma: no cover - explainer failure
            logger.exception("Model explainer failed for %s", model_name)
    return _default_explanations(model_name, payload, prediction)


def predict(model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a prediction using registered model loaders and runners."""

    model_spec = get_model_spec(model_name)
    framework = model_spec.get("framework", "unknown")
    version = model_spec.get("version", "v1")
    model_instance = _load_model_instance(model_name, model_spec)
    framework_key = framework.lower()
    model_type = str(model_spec.get("type") or "").lower()
    metadata = model_spec.get("metadata") or {}
    runner_callable = None
    runner_path = metadata.get("runner")
    if runner_path:
        try:
            runner_callable = _load_python_callable(runner_path)
        except Exception:  # pragma: no cover - runner import failure
            logger.exception("Failed to import custom runner for %s", model_name)
            runner_callable = None
    runner = (
        runner_callable
        or MODEL_RUNNERS.get(framework_key)
        or MODEL_RUNNERS.get(model_type)
        or MODEL_RUNNERS.get("generic")
    )
    output: Optional[Dict[str, Any]] = None
    if model_instance is not None and runner:
        try:
            output = runner(model_name, model_instance, payload, model_spec)
        except Exception:  # pragma: no cover - runner failure
            logger.exception("Model runner failed for %s", model_name)
            output = None
    if not isinstance(output, dict) or "score" not in output:
        output = {"score": 0.42, "label": "Positive"}
    result = {
        "model": model_name,
        "version": version,
        "framework": framework,
        "input": payload,
        "output": output,
        "spec_metadata": metadata,
    }
    result["explanations"] = explain_prediction(model_name, payload, result)
    return result


async def broadcast_page_snapshot(slug: str, payload: Dict[str, Any]) -> None:
    if not REALTIME_ENABLED:
        return
    message = {
        "type": "snapshot",
        "slug": slug,
        "payload": payload,
        "meta": _page_meta(slug),
    }
    await BROADCAST.broadcast(slug, _with_timestamp(message))


async def broadcast_component_update(
    slug: str,
    component_type: str,
    component_index: int,
    model: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if not REALTIME_ENABLED:
        return
    payload = _model_to_payload(model)
    message: Dict[str, Any] = {
        "type": "component",
        "slug": slug,
        "component_type": component_type,
        "component_index": component_index,
        "payload": payload,
        "meta": {"page": _page_meta(slug)},
    }
    if meta:
        message["meta"].update(meta)
    await BROADCAST.broadcast(slug, _with_timestamp(message))


async def broadcast_rollback(slug: str, component_index: int) -> None:
    if not REALTIME_ENABLED:
        return
    message = {
        "type": "rollback",
        "slug": slug,
        "component_index": component_index,
        "meta": {"page": _page_meta(slug)},
    }
    await BROADCAST.broadcast(slug, _with_timestamp(message))

    '''
).strip()

__all__ = ['PREDICTION_SECTION']
