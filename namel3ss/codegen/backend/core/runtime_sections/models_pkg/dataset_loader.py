def _ensure_dataset_inputs(args: Dict[str, Any], dataset_payload: Dict[str, Any]) -> None:
    if not isinstance(args, dict):
        return
    args.setdefault("dataset", dataset_payload)
    if dataset_payload.get("rows"):
        args.setdefault("examples", dataset_payload["rows"])
    if dataset_payload.get("y") and not args.get("y_true"):
        args["y_true"] = list(dataset_payload["y"])
    if dataset_payload.get("splits") and not args.get("splits"):
        args["splits"] = dataset_payload["splits"]
    existing_input = args.get("input")
    if isinstance(existing_input, dict):
        existing_input.setdefault("dataset", dataset_payload)
    elif existing_input is None:
        args["input"] = {"dataset": dataset_payload}


def _prepare_experiment_dataset(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict) or not config:
        raise ValueError("Experiment data configuration must be a non-empty dictionary.")
    context = build_context(None)
    frame = _await_frame_task(_load_experiment_frame(config, context))
    if frame is None:
        raise ValueError("Experiment data source did not return a frame.")
    columns = frame.spec.get("columns") or []
    column_index = {column.get("name"): column for column in columns if column.get("name")}
    frame_name = frame.name or config.get("frame") or "frame"

    target_column = _resolve_target_column(config, columns, column_index, frame_name)
    feature_columns = _resolve_feature_columns(config, columns, column_index, target_column, frame_name)
    time_column = _resolve_time_column(config, columns, column_index)
    group_columns = _resolve_group_columns(config, columns, column_index)
    weight_column = _resolve_weight_column(config, columns, column_index)

    rows = _clone_rows(frame.rows)
    if not rows:
        raise ValueError(f"Frame '{frame_name}' did not yield any rows for experiment data.")

    dataset_payload: Dict[str, Any] = {
        "frame": frame_name,
        "schema": frame.schema_payload(),
        "features": feature_columns,
        "target": target_column,
        "time_column": time_column,
        "group_columns": group_columns,
        "weight_column": weight_column,
        "rows": rows,
        "X": _build_feature_matrix(rows, feature_columns),
        "y": _build_target_series(rows, target_column),
    }
    if weight_column:
        dataset_payload["weights"] = _build_target_series(rows, weight_column)
    split_spec = config.get("splits") or frame.spec.get("splits") or {}
    splits_payload = _materialise_split_payload(
        rows,
        split_spec,
        feature_columns,
        target_column,
        weight_column,
        time_column,
        frame.spec.get("key"),
    )
    if splits_payload:
        dataset_payload["splits"] = splits_payload
    return dataset_payload


async def _load_experiment_frame(config: Dict[str, Any], context: Dict[str, Any]) -> Any:
    pipeline_value = config.get("pipeline")
    if isinstance(pipeline_value, dict) and "__frame_pipeline__" in pipeline_value:
        pipeline_value = pipeline_value["__frame_pipeline__"]
    if isinstance(pipeline_value, dict) and pipeline_value.get("root"):
        return await _evaluate_frame_pipeline(pipeline_value, None, context, set())
    frame_name = config.get("frame")
    if not frame_name:
        raise ValueError("Experiment data config requires a 'frame' or 'pipeline'.")
    return await _resolve_frame_runtime(frame_name, None, context, set())


def _await_frame_task(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder: Dict[str, Any] = {}
    error: List[BaseException] = []
    done = threading.Event()

    def _runner() -> None:
        try:
            holder["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            error.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    done.wait()
    if error:
        raise error[0]
    return holder.get("value")


def _resolve_target_column(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
    frame_name: str,
) -> str:
    target_value = config.get("target") or config.get("label")
    if isinstance(target_value, dict):
        target_value = target_value.get("name")
    if target_value:
        target_name = str(target_value)
        if target_name not in column_index:
            raise ValueError(f"Frame '{frame_name}' does not define target column '{target_name}'.")
        return target_name
    for column in columns:
        role = str(column.get("role") or "").lower()
        if role in {"target", "label"} and column.get("name"):
            return column["name"]
    raise ValueError(f"Frame '{frame_name}' does not define a target column or role.")


def _resolve_feature_columns(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
    target_column: str,
    frame_name: str,
) -> List[str]:
    explicit = _coerce_column_list(config.get("features") or config.get("feature_columns"))
    if explicit:
        _validate_columns(explicit, column_index, frame_name)
        return _deduplicate(explicit)
    inferred: List[str] = []
    for column in columns:
        name = column.get("name")
        if not name or name == target_column:
            continue
        role = str(column.get("role") or "").lower()
        if role in {"feature", "input"}:
            inferred.append(name)
    if not inferred:
        for column in columns:
            name = column.get("name")
            if not name or name == target_column:
                continue
            inferred.append(name)
            if len(inferred) >= 4:
                break
    if not inferred:
        raise ValueError(f"Frame '{frame_name}' does not provide feature candidates.")
    return _deduplicate(inferred)


def _resolve_time_column(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    time_value = config.get("time") or config.get("time_column") or config.get("timestamp")
    if isinstance(time_value, dict):
        time_value = time_value.get("name")
    if time_value:
        name = str(time_value)
        if name not in column_index:
            raise ValueError(f"Frame does not define time column '{name}'.")
        return name
    for column in columns:
        role = str(column.get("role") or "").lower()
        if role == "time" and column.get("name"):
            return column["name"]
    return None


def _resolve_group_columns(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
) -> List[str]:
    explicit = _coerce_column_list(config.get("group_columns") or config.get("groups"))
    if explicit:
        _validate_columns(explicit, column_index, None)
        return _deduplicate(explicit)
    inferred: List[str] = []
    for column in columns:
        role = str(column.get("role") or "").lower()
        if role in {"group", "id"} and column.get("name"):
            inferred.append(column["name"])
    return _deduplicate(inferred)


def _resolve_weight_column(
    config: Dict[str, Any],
    columns: List[Dict[str, Any]],
    column_index: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    weight_value = config.get("weight") or config.get("weight_column") or config.get("sample_weight")
    if isinstance(weight_value, dict):
        weight_value = weight_value.get("name")
    if weight_value:
        candidate = str(weight_value)
        if candidate not in column_index:
            raise ValueError(f"Frame does not define weight column '{candidate}'.")
        return candidate
    for column in columns:
        role = str(column.get("role") or "").lower()
        if role == "weight" and column.get("name"):
            return column["name"]
    return None


def _coerce_column_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        parts = [segment.strip() for segment in value.split(",") if segment.strip()]
        return parts or [value.strip()]
    if isinstance(value, dict):
        if "name" in value:
            return [str(value["name"])]
        value = value.get("columns") or value.get("names")
    result: List[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and item.get("name"):
                result.append(str(item["name"]))
    return result


def _validate_columns(columns: List[str], column_index: Dict[str, Dict[str, Any]], frame_name: Optional[str]) -> None:
    missing = [name for name in columns if name not in column_index]
    if missing:
        joined = ", ".join(missing)
        if frame_name:
            raise ValueError(f"Frame '{frame_name}' does not define columns: {joined}")
        raise ValueError(f"Unknown columns referenced in experiment data config: {joined}")


def _deduplicate(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _materialise_split_payload(
    rows: List[Dict[str, Any]],
    split_spec: Dict[str, Any],
    feature_columns: List[str],
    target_column: str,
    weight_column: Optional[str],
    time_column: Optional[str],
    key_columns: Optional[Sequence[str]],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(split_spec, dict) or not split_spec:
        return {}
    normalized: List[Tuple[str, float]] = []
    for name, value in split_spec.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        normalized.append((str(name), weight))
    if not normalized:
        return {}
    order_source = list(rows)
    if time_column and all(time_column in row for row in order_source):
        order_source.sort(key=lambda row: row.get(time_column))
    elif key_columns:
        order_source.sort(key=lambda row: tuple(row.get(col) for col in key_columns))
    total_rows = len(order_source)
    total_weight = sum(weight for _, weight in normalized)
    splits: Dict[str, Dict[str, Any]] = {}
    offset = 0
    for index, (name, weight) in enumerate(normalized):
        if index == len(normalized) - 1:
            subset = order_source[offset:]
        else:
            span = int(round((weight / total_weight) * total_rows))
            if span <= 0 and total_rows > len(splits):
                span = 1
            subset = order_source[offset : offset + span]
            offset += span
        if not subset:
            continue
        splits[name] = _build_split_record(subset, feature_columns, target_column, weight_column)
    return splits


def _build_split_record(
    subset: List[Dict[str, Any]],
    feature_columns: List[str],
    target_column: str,
    weight_column: Optional[str],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "rows": list(subset),
        "X": _build_feature_matrix(subset, feature_columns),
        "y": _build_target_series(subset, target_column),
        "size": len(subset),
    }
    if weight_column:
        record["weights"] = _build_target_series(subset, weight_column)
    return record


def _build_feature_matrix(rows: List[Dict[str, Any]], feature_columns: List[str]) -> List[Dict[str, Any]]:
    matrix: List[Dict[str, Any]] = []
    for row in rows:
        matrix.append({column: row.get(column) for column in feature_columns})
    return matrix


def _build_target_series(rows: List[Dict[str, Any]], column: Optional[str]) -> List[Any]:
    if not column:
        return []
    return [row.get(column) for row in rows]