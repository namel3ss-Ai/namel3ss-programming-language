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