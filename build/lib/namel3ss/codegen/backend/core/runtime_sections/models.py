from __future__ import annotations

from textwrap import dedent

MODELS_SECTION = dedent(
    '''
def evaluate_experiment(
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec = AI_EXPERIMENTS.get(name)
    args = dict(payload or {})
    if not spec:
        return {
            "experiment": name,
            "variants": [],
            "metrics": [],
            "leaderboard": [],
            "winner": None,
            "inputs": args,
            "metadata": {},
            "status": "stub",
        }

    variants_result: List[Dict[str, Any]] = []
    for index, variant in enumerate(spec.get("variants", []), start=1):
        target_type = str(variant.get("target_type") or "model").lower()
        target_name = str(variant.get("target_name") or "")
        config = variant.get("config") or {}
        result_payload: Dict[str, Any]
        if target_type == "model" and target_name:
            model_input = args.get("input") or args.get("payload") or {}
            if not isinstance(model_input, dict):
                model_input = {"value": model_input}
            result_payload = predict(target_name, model_input)
        elif target_type == "chain" and target_name:
            chain_args = dict(args)
            chain_args.setdefault("input", args.get("input", args))
            result_payload = run_chain(target_name, chain_args)
        else:
            result_payload = {
                "status": "stub",
                "detail": f"Unsupported target '{target_type}' for variant '{variant.get('name')}'",
            }
        score = round(0.6 + 0.075 * index, 3)
        variants_result.append(
            {
                "name": variant.get("name"),
                "target_type": target_type,
                "target_name": target_name,
                "config": config,
                "score": score,
                "result": result_payload,
            }
        )

    metrics_result = _evaluate_experiment_metrics(spec, variants_result, args)

    leaderboard = sorted(
        variants_result,
        key=lambda item: item.get("score", 0),
        reverse=True,
    )
    winner = leaderboard[0]["name"] if leaderboard else None

    return {
        "experiment": spec.get("name", name),
        "slug": spec.get("slug", name),
        "variants": variants_result,
        "metrics": metrics_result,
        "leaderboard": [entry.get("name") for entry in leaderboard if entry.get("name")],
        "winner": winner,
        "inputs": args,
        "metadata": spec.get("metadata", {}),
        "status": "ok",
    }


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
