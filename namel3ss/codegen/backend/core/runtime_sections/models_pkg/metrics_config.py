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