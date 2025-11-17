from __future__ import annotations

from textwrap import dedent

MODELS_SECTION = dedent(
    '''
def evaluate_experiment(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate an experiment by executing each variant and computing metrics.

    Parameters
    ----------
    name:
        Experiment identifier registered in ``AI_EXPERIMENTS``.
    payload:
        Optional dictionary containing request inputs. Should include prediction
        payloads and, when required, ground-truth targets (for example
        ``{"y_true": [...], "y_pred": [...]}`` or ``{"examples": [{"y_true": .., "y_pred": ..}]}``).

    Returns
    -------
    Dict[str, Any]
        Structured evaluation report containing keys ``experiment``,
        ``variants`` (per-variant outputs and metric values), ``metrics``
        (experiment-level summaries), ``metric_definitions`` (metric settings),
        ``leaderboard`` (sorted variant summaries), ``winner`` (top variant
        name), ``inputs`` (echoed payload), ``metadata`` and a ``status`` flag
        (``"ok"`` or ``"not_found"``).

        Variants include a ``status`` field set to ``"ok"`` on success or
        ``"error"`` when execution failed. When a variant fails, the error is
        captured without aborting the experiment.
    """

    spec = AI_EXPERIMENTS.get(name)
    args = dict(payload or {})
    if not spec:
        return {
            "status": "error",
            "error": "experiment_not_found",
            "detail": f"Experiment '{name}' is not registered.",
            "experiment": name,
            "variants": [],
            "metrics": [],
            "metric_definitions": [],
            "leaderboard": [],
            "winner": None,
            "inputs": args,
            "metadata": {},
        }

    metric_configs = _normalise_metric_configs(spec.get("metrics", []))
    primary_metric = _resolve_primary_metric(metric_configs)

    variants_result: List[Dict[str, Any]] = []
    for variant in spec.get("variants", []):
        variant_result = _evaluate_experiment_variant(variant, args)
        if metric_configs:
            variant_metrics = _compute_variant_metrics(variant_result, metric_configs, args)
            variant_result["metrics"] = variant_metrics
        variants_result.append(variant_result)

    experiment_metrics = _summarise_experiment_metrics(metric_configs, variants_result)
    leaderboard, winner = _build_experiment_leaderboard(variants_result, primary_metric)

    status = "ok"
    if variants_result and all(entry.get("status") == "error" for entry in variants_result):
        status = "error"

    result = {
        "experiment": spec.get("name", name),
        "slug": spec.get("slug") or spec.get("name", name),
        "variants": variants_result,
        "metrics": experiment_metrics,
        "metric_definitions": metric_configs,
        "leaderboard": leaderboard,
        "winner": winner,
        "inputs": args,
        "metadata": spec.get("metadata", {}),
        "status": status,
    }
    if status == "error":
        result.setdefault("error", "experiment_failed")
        result.setdefault("detail", "All experiment variants failed to execute.")
    return result


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


def _normalise_metric_configs(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    explicit_primary = False
    for metric in metrics or []:
        metadata_raw = metric.get("metadata") or {}
        metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {"value": metadata_raw}
        entry: Dict[str, Any] = {
            "name": metric.get("name"),
            "slug": metric.get("slug"),
            "goal": metric.get("goal"),
            "source_kind": metric.get("source_kind"),
            "source_name": metric.get("source_name"),
            "direction": _determine_metric_direction(metric.get("name"), metadata),
            "metadata": metadata,
        }
        primary_marker = metadata.get("primary") if isinstance(metadata, dict) else None
        if primary_marker is None:
            primary_marker = metric.get("primary")
        if isinstance(primary_marker, str):
            primary_marker = primary_marker.lower() in {"true", "1", "yes"}
        if isinstance(primary_marker, bool) and primary_marker:
            entry["primary"] = True
            explicit_primary = True
        configs.append(entry)
    if configs and not explicit_primary:
        configs[0]["primary"] = True
    return configs


def _determine_metric_direction(name: Optional[str], metadata: Dict[str, Any]) -> str:
    direction_value = metadata.get("direction") or metadata.get("goal_direction")
    if direction_value is None:
        direction_value = metadata.get("optimize") or metadata.get("optimise")
    if isinstance(direction_value, str):
        normalized = direction_value.lower()
        if normalized in {"max", "maximize", "maximise", "higher", "increase", "asc"}:
            return "maximize"
        if normalized in {"min", "minimize", "minimise", "lower", "decrease", "desc"}:
            return "minimize"
    goal_operator = metadata.get("goal_operator") or metadata.get("operator")
    if isinstance(goal_operator, str):
        text = goal_operator.strip().lower()
        if any(symbol in text for symbol in (">", "gt")):
            return "maximize"
        if any(symbol in text for symbol in ("<", "lt")):
            return "minimize"
    name_text = str(name or "").lower()
    if any(keyword in name_text for keyword in ("latency", "delay", "duration", "time", "error", "loss", "mse", "rmse", "mae", "cost")):
        return "minimize"
    return "maximize"


def _resolve_primary_metric(metrics: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metrics:
        return None
    for metric in metrics:
        marker = metric.get("primary")
        if isinstance(marker, str):
            marker = marker.lower() in {"true", "1", "yes"}
        if marker:
            return metric
    return metrics[0]


def _evaluate_experiment_variant(variant: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    target_type = str(variant.get("target_type") or "model").lower()
    target_name = str(variant.get("target_name") or "")
    config = variant.get("config") or {}
    result: Dict[str, Any] = {
        "name": variant.get("name"),
        "slug": variant.get("slug"),
        "target_type": target_type,
        "target_name": target_name,
        "config": config,
        "status": "ok",
        "raw_output": None,
        "result": None,
        "metrics": {},
    }
    try:
        if target_type == "model" and target_name:
            model_input = _resolve_variant_input(args, config)
            payload = predict(target_name, model_input)
        elif target_type == "chain" and target_name:
            chain_args = _resolve_chain_arguments(args, config)
            payload = run_chain(target_name, chain_args)
        elif target_type == "python":
            module_name = config.get("module") or target_name
            method_name = config.get("method") or config.get("callable") or "predict"
            python_args = _resolve_python_arguments(args, config)
            if not module_name:
                raise ValueError("python variant requires a module in 'target_name' or config['module']")
            payload = call_python_model(module_name, method_name, python_args)
        else:
            raise ValueError(f"Unsupported target type '{target_type}' for variant '{variant.get('name')}'")
        result["raw_output"] = payload
        result["result"] = payload
        status_value = _extract_status_from_payload(payload)
        if status_value == "error":
            result["status"] = "error"
            error_text = _extract_error_from_payload(payload)
            if error_text:
                result["error"] = error_text
        elif status_value == "partial":
            result["status"] = "partial"
    except Exception as exc:  # pragma: no cover - defensive guard
        result["status"] = "error"
        result["error"] = str(exc)
    return result


def _extract_status_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        status_value = payload.get("status")
        if isinstance(status_value, str):
            normalized = status_value.lower()
            if normalized in {"ok", "success"}:
                return "ok"
            if normalized in {"error", "failed", "failure"}:
                return "error"
            if normalized in {"partial"}:
                return "partial"
    return "ok"


def _extract_error_from_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if value:
                return str(value)
    return None


def _resolve_variant_input(args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config, dict):
        override = config.get("payload") or config.get("input") or config.get("data")
        if isinstance(override, dict):
            return dict(override)
        if isinstance(override, str) and override in args:
            candidate = args.get(override)
            if isinstance(candidate, dict):
                return dict(candidate)
            if candidate is not None:
                return {"value": candidate}
    model_input = args.get("input") or args.get("payload") or {}
    if isinstance(model_input, dict):
        return dict(model_input)
    if isinstance(model_input, (list, tuple)):
        return {"values": list(model_input)}
    if model_input is not None:
        return {"value": model_input}
    return {}


def _resolve_chain_arguments(args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    chain_args = dict(args)
    if isinstance(config, dict):
        overrides = config.get("inputs") or config.get("arguments") or {}
        if isinstance(overrides, dict):
            chain_args.update(overrides)
        input_override = config.get("input")
        if isinstance(input_override, dict):
            chain_args["input"] = dict(input_override)
        elif isinstance(input_override, str) and input_override in args:
            chain_args["input"] = args[input_override]
    chain_args.setdefault("input", args.get("input", args))
    return chain_args


def _resolve_python_arguments(args: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    python_args: Dict[str, Any] = {}
    if isinstance(config, dict):
        for key in ("arguments", "inputs", "kwargs"):
            value = config.get(key)
            if isinstance(value, dict):
                python_args.update(value)
    if not python_args and isinstance(args, dict):
        python_args = {"payload": args}
    return python_args


def _compute_variant_metrics(
    variant_result: Dict[str, Any],
    metric_configs: List[Dict[str, Any]],
    args: Dict[str, Any],
) -> Dict[str, float]:
    if variant_result.get("status") != "ok":
        return {}
    metric_inputs = _resolve_metric_inputs(variant_result, args)
    metrics: Dict[str, float] = {}
    for metric in metric_configs:
        value = _evaluate_metric_value(metric, metric_inputs)
        if value is not None:
            metrics[metric["name"]] = value
    return metrics


def _summarise_experiment_metrics(
    metric_configs: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for index, config in enumerate(metric_configs, start=1):
        name = config.get("name")
        metadata_source = config.get("metadata") or {}
        metadata = dict(metadata_source) if isinstance(metadata_source, dict) else {}
        values: List[float] = []
        if name:
            for variant in variants:
                variant_metrics = variant.get("metrics") or {}
                number = _to_float(variant_metrics.get(name))
                if number is not None:
                    values.append(number)
        if not values:
            for variant in variants:
                number = _extract_score_from_variant(variant)
                if number is not None:
                    values.append(number)
        aggregation = metadata.get("aggregation") or metadata.get("aggregate") or "mean"
        aggregation_text = str(aggregation).lower()
        aggregated_value = _aggregate_metric_values(values, aggregation_text)
        if aggregated_value is None:
            if not values:
                continue
            aggregated_value = float(sum(values) / len(values))
        else:
            aggregated_value = float(aggregated_value)
        metadata["aggregation"] = aggregation_text
        samples = len(values)
        metadata["samples"] = samples
        if values:
            metadata["summary"] = {
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 6),
            }
        precision = metadata.get("precision") if isinstance(metadata.get("precision"), int) else metadata.get("round")
        if isinstance(precision, int):
            aggregated_value = round(aggregated_value, max(int(precision), 0))
        else:
            aggregated_value = round(aggregated_value, 4)
        goal_value = _to_float(config.get("goal"))
        direction = config.get("direction")
        achieved_goal: Optional[bool] = None
        if goal_value is not None:
            minimize = str(direction).lower() == "minimize"
            achieved_goal = aggregated_value <= goal_value if minimize else aggregated_value >= goal_value
        summaries.append(
            {
                "name": name,
                "value": aggregated_value,
                "goal": config.get("goal"),
                "direction": direction,
                "source_kind": config.get("source_kind"),
                "source_name": config.get("source_name"),
                "metadata": metadata,
                "achieved_goal": achieved_goal,
                "primary": bool(config.get("primary")),
            }
        )
    return summaries


def _aggregate_metric_values(values: List[float], aggregation: Any) -> Optional[float]:
    if not values:
        return None
    key = str(aggregation or "mean").lower()
    if key in {"max", "maximum", "highest"}:
        return max(values)
    if key in {"min", "minimum", "lowest"}:
        return min(values)
    if key in {"sum", "total"}:
        return sum(values)
    if key in {"avg", "average", "mean"}:
        return sum(values) / len(values)
    if key in {"median", "p50"}:
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 0:
            return (ordered[mid - 1] + ordered[mid]) / 2.0
        return ordered[mid]
    if key in {"p95", "percentile95"}:
        return _percentile(values, 95)
    if key in {"p90", "percentile90"}:
        return _percentile(values, 90)
    if key in {"p80", "percentile80"}:
        return _percentile(values, 80)
    if key in {"p20", "percentile20"}:
        return _percentile(values, 20)
    if key in {"p10", "percentile10"}:
        return _percentile(values, 10)
    return sum(values) / len(values)


def _percentile(values: List[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = (percentile / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _resolve_metric_inputs(variant_result: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Optional[List[Any]]]:
    y_true = _normalise_sequence(args.get("y_true"))
    y_pred = _normalise_sequence(args.get("y_pred"))
    y_score = _normalise_numeric_series(args.get("y_score"))

    examples = args.get("examples")
    if isinstance(examples, list):
        if not y_true:
            y_true = [item.get("y_true") for item in examples if isinstance(item, dict) and item.get("y_true") is not None]
        if not y_pred:
            y_pred = [item.get("y_pred") for item in examples if isinstance(item, dict) and item.get("y_pred") is not None]

    if not y_true:
        for key in ("labels", "targets", "ground_truth", "actuals"):
            series = _normalise_sequence(args.get(key))
            if series:
                y_true = series
                break

    raw_output = variant_result.get("raw_output")
    if not y_pred:
        y_pred = _extract_predictions_from_raw(raw_output)
    if not y_true:
        y_true = _extract_ground_truth(args)
    if not y_score:
        y_score = _extract_numeric_scores(raw_output)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def _normalise_sequence(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict) and "values" in value:
        values = value.get("values")
        if isinstance(values, (list, tuple)):
            return list(values)
    return [value]


def _normalise_numeric_series(value: Any) -> Optional[List[float]]:
    sequence = _normalise_sequence(value)
    if not sequence:
        return None
    numeric: List[float] = []
    for item in sequence:
        number = _to_float(item)
        if number is None:
            return None
        numeric.append(number)
    return numeric


def _extract_predictions_from_raw(payload: Any) -> Optional[List[Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("predictions"), (list, tuple)):
            return list(payload["predictions"])
        output = payload.get("output")
        if isinstance(output, dict):
            if isinstance(output.get("predictions"), (list, tuple)):
                return list(output["predictions"])
            if isinstance(output.get("labels"), (list, tuple)):
                return list(output["labels"])
            if "label" in output:
                return [output.get("label")]
        if "result" in payload:
            result_value = payload.get("result")
            if isinstance(result_value, (list, tuple)):
                return list(result_value)
            return [result_value]
        if "value" in payload and not isinstance(payload.get("value"), dict):
            return [payload.get("value")]
    if isinstance(payload, (list, tuple)):
        return list(payload)
    if payload is not None:
        return [payload]
    return None


def _extract_numeric_scores(payload: Any) -> Optional[List[float]]:
    if isinstance(payload, dict):
        for key in ("score", "value"):
            number = _to_float(payload.get(key))
            if number is not None:
                return [number]
        output = payload.get("output")
        if isinstance(output, dict):
            number = _to_float(output.get("score"))
            if number is not None:
                return [number]
            if isinstance(output.get("scores"), (list, tuple)):
                numeric = [_to_float(item) for item in output.get("scores", [])]
                filtered = [item for item in numeric if item is not None]
                if filtered:
                    return [float(item) for item in filtered]
        result_value = payload.get("result")
        if isinstance(result_value, dict):
            numeric_nested = _extract_numeric_scores(result_value)
            if numeric_nested:
                return numeric_nested
        number = _to_float(payload.get("result"))
        if number is not None:
            return [number]
    if isinstance(payload, (list, tuple)):
        numeric = [_to_float(item) for item in payload]
        filtered = [item for item in numeric if item is not None]
        if filtered:
            return [float(item) for item in filtered]
    number = _to_float(payload)
    if number is not None:
        return [number]
    return None


def _extract_ground_truth(args: Dict[str, Any]) -> Optional[List[Any]]:
    for key in ("ground_truth", "truth", "labels", "targets", "actuals", "y_true"):
        series = _normalise_sequence(args.get(key))
        if series:
            return series
    examples = args.get("examples")
    if isinstance(examples, list):
        values = [item.get("y_true") for item in examples if isinstance(item, dict) and item.get("y_true") is not None]
        if values:
            return values
    return None


def _evaluate_metric_value(metric: Dict[str, Any], inputs: Dict[str, Optional[List[Any]]]) -> Optional[float]:
    name = str(metric.get("name") or "").lower()
    y_true = inputs.get("y_true")
    y_pred = inputs.get("y_pred")
    y_score = inputs.get("y_score")
    metadata = metric.get("metadata") or {}

    if name in {"accuracy", "acc"}:
        pairs = _label_pairs(y_true, y_pred)
        return _compute_accuracy(pairs)
    if name in {"precision"}:
        pairs = _label_pairs(y_true, y_pred)
        return _compute_precision(pairs, _safe_positive_label(metadata, pairs))
    if name in {"recall"}:
        pairs = _label_pairs(y_true, y_pred)
        return _compute_recall(pairs, _safe_positive_label(metadata, pairs))
    if name in {"f1", "f1_score"}:
        pairs = _label_pairs(y_true, y_pred)
        positive_label = _safe_positive_label(metadata, pairs)
        return _compute_f1(pairs, positive_label)
    if name in {"mse", "mean_squared_error"}:
        pairs = _numeric_pairs(y_true, y_pred or y_score)
        return _compute_mse(pairs)
    if name in {"rmse"}:
        pairs = _numeric_pairs(y_true, y_pred or y_score)
        return _compute_rmse(pairs)
    if name in {"mae", "mean_absolute_error"}:
        pairs = _numeric_pairs(y_true, y_pred or y_score)
        return _compute_mae(pairs)
    return None


def _label_pairs(y_true: Optional[List[Any]], y_pred: Optional[List[Any]]) -> List[Any]:
    if not y_true or not y_pred:
        return []
    pairs: List[Any] = []
    for truth, pred in zip(y_true, y_pred):
        if truth is None or pred is None:
            continue
        pairs.append((truth, pred))
    return pairs


def _numeric_pairs(y_true: Optional[List[Any]], y_pred: Optional[List[Any]]) -> List[Any]:
    if not y_true or not y_pred:
        return []
    pairs: List[Any] = []
    for truth, pred in zip(y_true, y_pred):
        truth_val = _to_float(truth)
        pred_val = _to_float(pred)
        if truth_val is None or pred_val is None:
            continue
        pairs.append((truth_val, pred_val))
    return pairs


def _labels_equal(left: Any, right: Any) -> bool:
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except Exception:
        pass
    return str(left).strip().lower() == str(right).strip().lower()


def _labels_match(value: Any, positive_label: Any) -> bool:
    if isinstance(positive_label, bool):
        return bool(value) is bool(positive_label) and bool(value) == positive_label
    if isinstance(positive_label, (int, float)) and not isinstance(positive_label, bool):
        try:
            return float(value) == float(positive_label)
        except Exception:
            return False
    return str(value).strip().lower() == str(positive_label).strip().lower()


def _safe_positive_label(metadata: Dict[str, Any], pairs: List[Any]) -> Any:
    label = metadata.get("positive_label") or metadata.get("positive_class")
    if label is not None:
        return label
    candidates = [truth for truth, _ in pairs if truth is not None]
    for candidate in candidates:
        text = str(candidate).strip().lower()
        if text in {"positive", "pos"}:
            return candidate
    for candidate in candidates:
        if isinstance(candidate, bool):
            return True
    for candidate in candidates:
        numeric = _to_float(candidate)
        if numeric == 1.0:
            return candidate
    return True


def _compute_accuracy(pairs: List[Any]) -> Optional[float]:
    if not pairs:
        return None
    matches = sum(1 for truth, pred in pairs if _labels_equal(truth, pred))
    return matches / len(pairs)


def _compute_precision(pairs: List[Any], positive_label: Any) -> Optional[float]:
    if not pairs:
        return None
    tp = fp = 0
    for truth, pred in pairs:
        if _labels_match(pred, positive_label):
            if _labels_match(truth, positive_label):
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def _compute_recall(pairs: List[Any], positive_label: Any) -> Optional[float]:
    if not pairs:
        return None
    tp = fn = 0
    for truth, pred in pairs:
        if _labels_match(truth, positive_label):
            if _labels_match(pred, positive_label):
                tp += 1
            else:
                fn += 1
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def _compute_f1(pairs: List[Any], positive_label: Any) -> Optional[float]:
    if not pairs:
        return None
    precision = _compute_precision(pairs, positive_label) or 0.0
    recall = _compute_recall(pairs, positive_label) or 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_mse(pairs: List[Any]) -> Optional[float]:
    if not pairs:
        return None
    errors = [(pred - truth) ** 2 for truth, pred in pairs]
    return sum(errors) / len(errors)


def _compute_rmse(pairs: List[Any]) -> Optional[float]:
    mse = _compute_mse(pairs)
    if mse is None:
        return None
    return mse ** 0.5


def _compute_mae(pairs: List[Any]) -> Optional[float]:
    if not pairs:
        return None
    errors = [abs(pred - truth) for truth, pred in pairs]
    return sum(errors) / len(errors)


def _build_experiment_leaderboard(
    variants: List[Dict[str, Any]],
    primary_metric: Optional[Dict[str, Any]],
):
    if not variants:
        return [], None
    metric_name = (primary_metric or {}).get("name")
    direction = (primary_metric or {}).get("direction", "maximize")
    maximize = str(direction).lower() != "minimize"

    entries: List[Dict[str, Any]] = []
    for variant in variants:
        metrics = variant.get("metrics") or {}
        score_value: Optional[float] = None
        if metric_name and metric_name in metrics:
            score_value = _to_float(metrics.get(metric_name))
        if score_value is None:
            score_value = _extract_score_from_variant(variant)
        variant["score"] = score_value
        if score_value is not None:
            entries.append({"name": variant.get("name"), "score": float(score_value)})

    if not entries:
        return [], None

    def sort_key(item: Dict[str, Any]) -> float:
        value = item.get("score")
        return -value if maximize else value

    leaderboard = sorted(entries, key=sort_key)
    winner = leaderboard[0]["name"] if leaderboard else None
    return leaderboard, winner


def _extract_score_from_variant(variant: Dict[str, Any]) -> Optional[float]:
    direct = _to_float(variant.get("score"))
    if direct is not None:
        return direct
    raw_output = variant.get("raw_output") or variant.get("result")
    if isinstance(raw_output, dict):
        for key in ("score",):
            number = _to_float(raw_output.get(key))
            if number is not None:
                return number
        output = raw_output.get("output")
        if isinstance(output, dict):
            number = _to_float(output.get("score"))
            if number is not None:
                return number
    return None


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

def _generic_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    metadata_obj = model_spec.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
    loader_path = metadata.get("loader")
    if not loader_path:
        return None
    try:
        callable_loader = _load_python_callable(loader_path)
        if callable_loader is None:
            return None
        return callable_loader(model_name, model_spec)
    except Exception:  # pragma: no cover - user loader failure
        logger.exception("Generic loader failed for model %s", model_name)
        return None


def _pytorch_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("PyTorch not available for model %s", model_name)
        return None
    try:
        if path.endswith((".pt", ".pth")):
            try:
                return torch.jit.load(path, map_location="cpu")
            except Exception:
                return torch.load(path, map_location="cpu")
        return torch.load(path, map_location="cpu")
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load PyTorch model %s from %s", model_name, path)
        return None


def _tensorflow_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import tensorflow as tf  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("TensorFlow not available for model %s", model_name)
        return None
    try:
        return tf.saved_model.load(path)
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load TensorFlow model %s from %s", model_name, path)
        return None


def _onnx_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path or not Path(path).exists():
        return None
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.debug("ONNX Runtime not available for model %s", model_name)
        return None
    try:
        return ort.InferenceSession(path)
    except Exception:  # pragma: no cover - IO/runtime failure
        logger.exception("Failed to load ONNX model %s from %s", model_name, path)
        return None


def _sklearn_loader(model_name: str, model_spec: Dict[str, Any]) -> Any:
    path = _resolve_model_artifact_path(model_spec)
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        logger.debug("scikit-learn artifact for %s not found at %s", model_name, file_path)
        return None
    try:
        try:
            import joblib  # type: ignore
        except Exception:
            joblib = None  # type: ignore
        if joblib is not None:
            return joblib.load(file_path)
    except Exception:
        logger.exception("joblib failed to load model %s", model_name)
    try:
        with file_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        logger.exception("Failed to load pickled model %s from %s", model_name, file_path)
    return None


def _coerce_numeric_payload(payload: Any) -> Optional[List[float]]:
    if isinstance(payload, dict):
        values: List[float] = []
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, (list, tuple)):
                try:
                    values.extend(float(item) for item in value)
                except Exception:
                    return None
            else:
                return None
        return values
    if isinstance(payload, (list, tuple)):
        try:
            return [float(item) for item in payload]
        except Exception:
            return None
    if isinstance(payload, (int, float)):
        return [float(payload)]
    return None


def _generic_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    metadata = model_spec.get("metadata") or {}
    runner_path = metadata.get("runner")
    if runner_path:
        try:
            callable_runner = _load_python_callable(runner_path)
            if callable_runner is not None:
                result = callable_runner(model_instance, payload, model_spec)
                if isinstance(result, dict):
                    return result
        except Exception:  # pragma: no cover - user runner failure
            logger.exception("Custom runner failed for model %s", model_name)
    if callable(model_instance):
        try:
            result = model_instance(payload)
            if isinstance(result, dict):
                return result
            if isinstance(result, (list, tuple)) and result:
                score = float(result[0])
                label = "Positive" if score >= 0 else "Negative"
                return {"score": score, "label": label}
        except Exception:
            logger.debug("Callable runner failed for model %s", model_name)
    return None


def _pytorch_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    if model_instance is None:
        return None
    try:
        if hasattr(model_instance, "eval"):
            model_instance.eval()
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_tensor = torch.tensor(values, dtype=torch.float32)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model_instance(input_tensor)
        if isinstance(output, torch.Tensor):
            flattened = output.detach().cpu().view(-1).tolist()
            score = float(flattened[0]) if flattened else 0.0
        elif isinstance(output, (list, tuple)) and output:
            score = float(output[0])
        else:
            return None
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("PyTorch runner failed for model %s", model_name)
        return None


def _tensorflow_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    if model_instance is None:
        return None
    try:
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_tensor = tf.convert_to_tensor([values], dtype=tf.float32)
        output = model_instance(input_tensor)
        if hasattr(output, "numpy"):
            score = float(output.numpy().reshape(-1)[0])
        elif isinstance(output, (list, tuple)) and output:
            first = output[0]
            if hasattr(first, "numpy"):
                score = float(first.numpy().reshape(-1)[0])
            else:
                score = float(first)
        else:
            return None
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("TensorFlow runner failed for model %s", model_name)
        return None


def _onnx_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if model_instance is None:
        return None
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    try:
        values = _coerce_numeric_payload(payload)
        if values is None:
            return None
        input_name = model_instance.get_inputs()[0].name  # type: ignore[attr-defined]
        array = np.array([values], dtype=np.float32)
        output = model_instance.run(None, {input_name: array})  # type: ignore[call-arg]
        if not output:
            return None
        first = output[0]
        if hasattr(first, "reshape"):
            score = float(first.reshape(-1)[0])
        elif isinstance(first, (list, tuple)) and first:
            score = float(first[0])
        else:
            score = float(first)
        label = "Positive" if score >= 0 else "Negative"
        return {"score": score, "label": label}
    except Exception:  # pragma: no cover - runtime failure
        logger.exception("ONNX runner failed for model %s", model_name)
        return None


def _sklearn_runner(
    model_name: str,
    model_instance: Any,
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if model_instance is None:
        return None
    values = _coerce_numeric_payload(payload)
    if values is None:
        return None
    sample = [values]
    try:
        score: Optional[float] = None
        if hasattr(model_instance, "predict_proba"):
            probabilities = model_instance.predict_proba(sample)  # type: ignore[attr-defined]
            if hasattr(probabilities, "tolist"):
                probabilities = probabilities.tolist()
            if isinstance(probabilities, list) and probabilities:
                first = probabilities[0]
                if isinstance(first, list) and first:
                    score = float(first[-1])
                elif isinstance(first, (int, float)):
                    score = float(first)
        if score is None and hasattr(model_instance, "predict"):
            prediction = model_instance.predict(sample)  # type: ignore[attr-defined]
            if hasattr(prediction, "tolist"):
                prediction = prediction.tolist()
            if isinstance(prediction, list) and prediction:
                first_value = prediction[0]
                score = float(first_value)
            elif isinstance(prediction, (int, float)):
                score = float(prediction)
        if score is None:
            return None
        metadata = model_spec.get("metadata") or {}
        threshold = float(metadata.get("threshold", 0.5))
        label = "Positive" if score >= threshold else "Negative"
        return {"score": score, "label": label}
    except Exception:
        logger.exception("scikit-learn runner failed for model %s", model_name)
        return None


def _default_explainer(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    return _default_explanations(model_name, payload, prediction)


def _register_default_model_hooks() -> None:
    register_model_loader("generic", _generic_loader)
    register_model_loader("python", _generic_loader)
    register_model_loader("pytorch", _pytorch_loader)
    register_model_loader("torch", _pytorch_loader)
    register_model_loader("sklearn", _sklearn_loader)
    register_model_loader("scikit-learn", _sklearn_loader)
    register_model_loader("tensorflow", _tensorflow_loader)
    register_model_loader("tf", _tensorflow_loader)
    register_model_loader("onnx", _onnx_loader)
    register_model_loader("onnxruntime", _onnx_loader)

    register_model_runner("generic", _generic_runner)
    register_model_runner("callable", _generic_runner)
    register_model_runner("pytorch", _pytorch_runner)
    register_model_runner("torch", _pytorch_runner)
    register_model_runner("sklearn", _sklearn_runner)
    register_model_runner("scikit-learn", _sklearn_runner)
    register_model_runner("tensorflow", _tensorflow_runner)
    register_model_runner("tf", _tensorflow_runner)
    register_model_runner("onnx", _onnx_runner)
    register_model_runner("onnxruntime", _onnx_runner)

    register_model_explainer("generic", _default_explainer)
    register_model_explainer("pytorch", _default_explainer)
    register_model_explainer("torch", _default_explainer)
    register_model_explainer("tensorflow", _default_explainer)
    register_model_explainer("tf", _default_explainer)
    register_model_explainer("onnx", _default_explainer)
    register_model_explainer("onnxruntime", _default_explainer)


_register_default_model_hooks()

    '''
).strip()

__all__ = ['MODELS_SECTION']
