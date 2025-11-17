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
        except Exception as exc:  # pragma: no cover - loader import failure
            logger.exception("Failed to import custom loader for %s", model_name)
            raise RuntimeError(f"Custom loader for '{model_name}' could not be imported") from exc
    loader = (
        custom_loader
        or MODEL_LOADERS.get(framework)
        or MODEL_LOADERS.get(model_type)
        or MODEL_LOADERS.get("generic")
    )
    if loader is None:
        raise RuntimeError(
            f"No loader registered for model '{model_name}' (framework='{framework or 'unknown'}', type='{model_type or 'unknown'}')"
        )
    def _invoke_loader(func: Callable[..., Any]) -> Any:
        attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = [
            ((model_name, model_spec), {}),
            ((model_spec,), {}),
        ]
        if isinstance(metadata, dict) and metadata:
            attempts.append(((metadata,), {}))
            attempts.append(((), {"config": metadata}))
        attempts.append(((), {}))
        attempts.append(((), {"model_name": model_name, "model_spec": model_spec}))

        last_signature_error: Optional[BaseException] = None
        for args, kwargs in attempts:
            try:
                return func(*args, **kwargs)
            except TypeError as exc:
                message = str(exc).lower()
                signature_issue = any(token in message for token in ("positional", "keyword", "argument"))
                if signature_issue:
                    last_signature_error = exc
                    continue
                raise
        if last_signature_error is not None:
            raise last_signature_error
        raise RuntimeError(f"Loader for '{model_name}' could not be invoked with available signatures")

    try:
        instance = _invoke_loader(loader)
    except Exception as exc:  # pragma: no cover - loader failure
        logger.exception("Model loader failed for %s", model_name)
        raise RuntimeError(f"Loader for '{model_name}' raised an error") from exc
    if isinstance(instance, dict) and instance.get("status") == "error":
        error_detail = instance.get("detail") or instance.get("error") or "unknown loader error"
        raise RuntimeError(f"Loader for '{model_name}' reported an error: {error_detail}")
    MODEL_CACHE[model_name] = instance
    return instance


def _coerce_output(raw_output: Any) -> Dict[str, Any]:
    """Normalise a raw runner payload into the canonical prediction schema."""

    result: Dict[str, Any] = {
        "score": None,
        "label": None,
        "scores": None,
        "raw": raw_output,
        "status": "ok",
        "error": None,
    }

    if raw_output is None:
        result["status"] = "partial"
        return result

    if isinstance(raw_output, Exception):
        result["status"] = "error"
        result["error"] = f"{type(raw_output).__name__}: {raw_output}"
        return result

    if isinstance(raw_output, dict):
        status_value = raw_output.get("status")
        if isinstance(status_value, str):
            normalized = status_value.lower()
            if normalized in {"error", "failed", "failure"}:
                result["status"] = "error"
            elif normalized in {"partial", "incomplete"}:
                result["status"] = "partial"
        error_value = raw_output.get("error") or raw_output.get("detail") or raw_output.get("message")
        if error_value:
            result["error"] = str(error_value)
            if result["status"] == "ok":
                result["status"] = "partial"

        candidates: List[Dict[str, Any]] = [raw_output]
        output_section = raw_output.get("output")
        if isinstance(output_section, dict):
            candidates.append(output_section)
        prediction_section = raw_output.get("prediction")
        if isinstance(prediction_section, dict):
            candidates.append(prediction_section)

        for candidate in candidates:
            if result["score"] is None:
                for key in ("score", "prob", "probability", "probabilities", "proba", "confidence"):
                    value = candidate.get(key)
                    coerced = _to_float(value)
                    if coerced is not None:
                        result["score"] = coerced
                        break
            if result["label"] is None:
                for key in ("label", "prediction", "class", "value"):
                    if key in candidate:
                        value = candidate[key]
                        if isinstance(value, (str, int, float)):
                            result["label"] = value
                            break
                        if isinstance(value, (list, tuple)) and value:
                            first = value[0]
                            if isinstance(first, (str, int, float)):
                                result["label"] = first
                                break
            if result["scores"] is None:
                for key in ("scores", "probabilities", "logits", "distribution"):
                    value = candidate.get(key)
                    if isinstance(value, (dict, list, tuple)):
                        result["scores"] = value
                        break

        if result["score"] is None and isinstance(result["scores"], dict):
            for key, value in result["scores"].items():
                numeric = _to_float(value)
                if numeric is not None:
                    result["score"] = numeric
                    if result["label"] is None and isinstance(key, (str, int)):
                        result["label"] = key
                    break

        if result["score"] is None and isinstance(result["scores"], (list, tuple)):
            for value in result["scores"]:
                numeric = _to_float(value)
                if numeric is not None:
                    result["score"] = numeric
                    break

        if result["label"] is None:
            label_candidate = raw_output.get("label") if isinstance(raw_output, dict) else None
            if isinstance(label_candidate, (str, int, float)):
                result["label"] = label_candidate

        if result["score"] is None and result["label"] is None and result["scores"] is None and result["error"] is None:
            result["status"] = "partial"

        return result

    if isinstance(raw_output, (list, tuple)):
        numeric_values = [_to_float(value) for value in raw_output]
        filtered = [value for value in numeric_values if value is not None]
        if filtered:
            result["score"] = filtered[0]
            result["scores"] = filtered
        else:
            result["scores"] = list(raw_output)
        result["status"] = "partial" if result["score"] is None else "ok"
        return result

    if isinstance(raw_output, (int, float)):
        result["score"] = float(raw_output)
        return result

    if isinstance(raw_output, str):
        result["label"] = raw_output
        result["status"] = "partial"
        return result

    result["status"] = "partial"
    return result


def _safe_numeric_explanations(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Generate lightweight perturbation-based explanations when possible.

    The helper inspects numeric features in the input payload, perturbs each by
    ±ε (where ε = max(|value| * 0.01, 0.01)) and measures the local impact on
    the primary scalar prediction. It falls back gracefully when inputs are not
    numeric or when the runtime cannot be re-executed.
    """

    if not isinstance(payload, dict) or not payload:
        return None

    context = prediction.get("_explanation_context") if isinstance(prediction, dict) else None
    if not isinstance(context, dict):
        return None

    runner = context.get("runner")
    model_instance = context.get("model_instance")
    model_spec = context.get("model_spec")
    coerce = context.get("coerce")
    base_output = prediction.get("output") if isinstance(prediction, dict) else None

    if runner is None or model_instance is None or model_spec is None or coerce is None:
        return None
    if not isinstance(base_output, dict):
        return None

    numeric_features: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            numeric_features[key] = float(value)
    if not numeric_features:
        return None

    feature_names = sorted(numeric_features.keys())[:16]

    reference = _select_scalar_reference(base_output)
    if reference is None:
        return None

    local_impacts: List[Dict[str, Any]] = []
    global_importances: Dict[str, float] = {}

    def _invoke(adjusted_payload: Dict[str, Any]) -> Optional[float]:
        try:
            raw = runner(model_name, model_instance, adjusted_payload, model_spec)
        except Exception:
            logger.exception("Model runner failed during explanation for %s", model_name)
            return None
        coerced = coerce(raw)
        if coerced.get("status") == "error":
            return None
        return _extract_scalar_from_output(coerced, reference)

    for feature in feature_names:
        base_value = numeric_features[feature]
        epsilon = max(abs(base_value) * 0.01, 0.01)

        decreased_payload = dict(payload)
        increased_payload = dict(payload)
        decreased_payload[feature] = base_value - epsilon
        increased_payload[feature] = base_value + epsilon

        lower = _invoke(decreased_payload)
        upper = _invoke(increased_payload)
        if lower is None or upper is None:
            continue

        impact = (upper - lower) / (2 * epsilon)
        local_impacts.append({"feature": feature, "impact": impact})
        global_importances[feature] = abs(impact)

    if not local_impacts:
        return None

    return {
        "global_importances": global_importances,
        "local_explanations": local_impacts,
        "visualizations": {},
    }


def _default_stub_explanations(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    output = prediction.get("output") if isinstance(prediction, dict) else None
    features: Dict[str, float] = {}
    if isinstance(output, dict):
        feature_map = output.get("features")
        if isinstance(feature_map, dict):
            for key, value in feature_map.items():
                if isinstance(value, (int, float)):
                    features[str(key)] = float(value)
    if not features and isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                features[str(key)] = float(value)
    if not features:
        features = {"feature_a": 0.5, "feature_b": 0.25}
    total = sum(abs(value) for value in features.values()) or 1.0
    global_importances = {
        key: round(abs(value) / total, 4)
        for key, value in features.items()
    }
    local_explanations = [
        {"feature": key, "impact": round(value, 4)}
        for key, value in features.items()
    ]
    return {
        "global_importances": global_importances,
        "local_explanations": local_explanations,
        "visualizations": {
            "saliency_map": "base64://image_classifier_saliency",
            "grad_cam": "base64://image_classifier_grad_cam",
            "attention": "base64://demo_attention_heatmap",
        },
    }


def _select_scalar_reference(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    score = _to_float(output.get("score"))
    if score is not None:
        return {"kind": "score"}
    scores = output.get("scores")
    if isinstance(scores, dict):
        numeric_items = [
            (key, _to_float(value))
            for key, value in scores.items()
            if _to_float(value) is not None
        ]
        if numeric_items:
            key, _ = max(numeric_items, key=lambda item: item[1])
            return {"kind": "scores_dict", "key": key}
    if isinstance(scores, (list, tuple)):
        numeric_values = [_to_float(value) for value in scores]
        filtered = [value for value in numeric_values if value is not None]
        if filtered:
            index = numeric_values.index(filtered[0])
            return {"kind": "scores_index", "index": index}
    label = output.get("label")
    if isinstance(label, (str, int, float)):
        return {"kind": "label"}
    return None


def _extract_scalar_from_output(output: Dict[str, Any], reference: Dict[str, Any]) -> Optional[float]:
    kind = reference.get("kind") if isinstance(reference, dict) else None
    if kind == "score":
        return _to_float(output.get("score"))
    if kind == "scores_dict":
        scores = output.get("scores")
        key = reference.get("key")
        if isinstance(scores, dict) and key in scores:
            return _to_float(scores.get(key))
    if kind == "scores_index":
        scores = output.get("scores")
        index = reference.get("index")
        if isinstance(scores, (list, tuple)) and isinstance(index, int) and 0 <= index < len(scores):
            return _to_float(scores[index])
    if kind == "label":
        label_value = output.get("label")
        return _to_float(label_value)
    return _to_float(output.get("score"))


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _deterministic_stub_output(
    model_name: str,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Dict[str, Any]:
    numeric_inputs = {
        str(key): float(value)
        for key, value in (payload or {}).items()
        if isinstance(value, (int, float))
    }
    if not numeric_inputs:
        numeric_inputs = {"feature_a": 0.5, "feature_b": 0.25}
    weights = [0.3, 0.15, 0.1, 0.05, 0.03]
    baseline = 0.75
    score = baseline
    for index, (key, value) in enumerate(sorted(numeric_inputs.items())):
        weight = weights[index % len(weights)]
        score += value * weight
    score = round(score, 2)
    label = "Positive" if score >= 0 else "Negative"
    features = {key: round(value, 4) for key, value in numeric_inputs.items()}
    confidence = round(min(max(score / (abs(score) + 1.0), 0.0), 1.0), 4)
    return {
        "score": score,
        "label": label,
        "features": features,
        "confidence": confidence,
        "status": "ok",
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
    fallback = _safe_numeric_explanations(model_name, payload, prediction)
    if fallback:
        return fallback
    try:
        allow_stubs = _is_truthy_env("NAMEL3SS_ALLOW_STUBS")  # type: ignore[name-defined]
    except NameError:  # pragma: no cover - defensive when section loaded independently
        allow_stubs = False
    if allow_stubs:
        return _default_stub_explanations(model_name, payload, prediction)
    return None


def predict(model_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute a model runner, normalise its output, and attach viable explanations.

    The function preserves the public signature while ensuring that no synthetic
    predictions are fabricated. Runner outputs are coerced into a canonical
    structure, and perturbation-based explanations are only produced for numeric
    inputs when no custom explainer is available.
    """

    from time import perf_counter

    inputs: Dict[str, Any]
    if isinstance(payload, dict):
        inputs = dict(payload)
    elif payload is None:
        inputs = {}
    else:
        inputs = {"value": payload}

    model_spec = get_model_spec(model_name)
    framework = model_spec.get("framework", "unknown")
    version = model_spec.get("version", "v1")

    metadata_payload: Dict[str, Any] = {
        "framework": framework,
        "version": version,
    }

    try:
        model_instance = _load_model_instance(model_name, model_spec)
    except Exception as exc:
        logger.exception("Failed to load model instance for %s", model_name)
        error_message = f"{type(exc).__name__}: {exc}"
        return {
            "model": model_name,
            "version": version,
            "framework": framework,
            "inputs": inputs,
            "input": inputs,
            "output": {
                "score": None,
                "label": None,
                "scores": None,
                "raw": None,
                "status": "error",
                "error": error_message,
            },
            "spec_metadata": model_spec.get("metadata") or {},
            "metadata": metadata_payload,
            "status": "error",
        }
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

    runner_name = getattr(runner, "__name__", None) if runner else None
    if not runner_name and runner_callable is not None:
        runner_name = getattr(runner_callable, "__name__", None)
    if runner_name:
        metadata_payload["runner"] = runner_name

    overall_status = "ok"
    raw_output: Any = None
    runner_error: Optional[BaseException] = None
    timing_ms: Optional[float] = None

    if runner is None:
        overall_status = "error"
        output = {
            "score": None,
            "label": None,
            "scores": None,
            "raw": None,
            "status": "error",
            "error": "Runner not registered for model",
        }
    elif model_instance is None:
        overall_status = "error"
        output = {
            "score": None,
            "label": None,
            "scores": None,
            "raw": None,
            "status": "error",
            "error": "Model loader returned no instance",
        }
    else:
        start = perf_counter()
        try:
            raw_output = runner(model_name, model_instance, inputs, model_spec)
        except Exception as exc:  # pragma: no cover - runner failure guard
            runner_error = exc
            logger.exception("Model runner failed for %s", model_name)
        finally:
            elapsed = perf_counter() - start
            timing_ms = round(elapsed * 1000, 4)
            metadata_payload["timing_ms"] = timing_ms

        if runner_error is not None:
            overall_status = "error"
            output = {
                "score": None,
                "label": None,
                "scores": None,
                "raw": None,
                "status": "error",
                "error": f"{type(runner_error).__name__}: {runner_error}",
            }
        else:
            output = _coerce_output(raw_output)
            overall_status = output.get("status", "ok")

    explanations = None
    if isinstance(raw_output, dict):
        for container in (raw_output, raw_output.get("output") if isinstance(raw_output.get("output"), dict) else None):
            if not isinstance(container, dict):
                continue
            candidate = container.get("explanations")
            if isinstance(candidate, dict) and candidate:
                explanations = candidate
                break
            visuals = container.get("visualizations")
            if isinstance(visuals, dict) and visuals:
                explanations = {"visualizations": visuals}
                break

    result = {
        "model": model_name,
        "version": version,
        "framework": framework,
        "inputs": inputs,
        "input": inputs,
        "output": output,
        "spec_metadata": metadata,
        "metadata": {key: value for key, value in metadata_payload.items() if value is not None},
        "status": overall_status,
    }

    if explanations is None and runner_error is None and runner is not None and overall_status != "error":
        result["_explanation_context"] = {
            "runner": runner,
            "model_instance": model_instance,
            "model_spec": model_spec,
            "coerce": _coerce_output,
        }
        custom_explanations = explain_prediction(model_name, inputs, result)
        result.pop("_explanation_context", None)
        if isinstance(custom_explanations, dict) and custom_explanations:
            explanations = custom_explanations
    else:
        result.pop("_explanation_context", None)

    if explanations:
        result["explanations"] = explanations

    if overall_status == "error" and output.get("error") is None:
        output["error"] = "Prediction failed"

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
