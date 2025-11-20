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

    data_config = spec.get("data_config")
    if isinstance(data_config, dict) and data_config:
        try:
            dataset_payload = _prepare_experiment_dataset(data_config, args)
        except Exception as exc:
            logger.exception("Failed to prepare dataset for experiment %s", name)
            return {
                "status": "error",
                "error": "experiment_data_error",
                "detail": str(exc),
                "experiment": spec.get("name", name),
                "slug": spec.get("slug") or spec.get("name", name),
                "variants": [],
                "metrics": [],
                "metric_definitions": metric_configs,
                "leaderboard": [],
                "winner": None,
                "inputs": args,
                "metadata": spec.get("metadata", {}),
            }
        _ensure_dataset_inputs(args, dataset_payload)

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