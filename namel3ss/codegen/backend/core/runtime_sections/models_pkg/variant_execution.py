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