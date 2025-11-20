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
            if isinstance(result_value, dict):
                if isinstance(result_value.get("predictions"), (list, tuple)):
                    return list(result_value["predictions"])
                if "label" in result_value:
                    return [result_value.get("label")]
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
        metadata = raw_output.get("metadata")
        if isinstance(metadata, dict):
            for meta_key in ("score", "value", "quality", "metric", "elapsed_ms"):
                number = _to_float(metadata.get(meta_key))
                if number is not None:
                    return number
        result_section = raw_output.get("result")
        if isinstance(result_section, dict):
            metadata = result_section.get("metadata")
            if isinstance(metadata, dict):
                for meta_key in ("score", "value", "quality", "metric", "elapsed_ms"):
                    number = _to_float(metadata.get(meta_key))
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