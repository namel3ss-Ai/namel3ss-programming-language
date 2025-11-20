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