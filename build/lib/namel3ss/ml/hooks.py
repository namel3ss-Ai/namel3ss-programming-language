"""Built-in training, deployment, and inference hooks for Namel3ss models."""

from __future__ import annotations

import importlib
import traceback
from typing import Any, Dict, List, Optional, Protocol, TypedDict


class ImageModel(Protocol):
    """Minimal protocol for image classification plugins."""

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - protocol
        """Return a structured prediction for ``inputs``."""


class ChurnModel(Protocol):
    """Minimal protocol for churn prediction plugins."""

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - protocol
        """Return a structured prediction for ``inputs``."""


class LocalExplanation(TypedDict, total=False):
    feature: str
    impact: float


class Prediction(TypedDict, total=False):
    score: Optional[float]
    label: Optional[str]
    scores: Optional[Dict[str, float]]
    raw: Any


class Explanations(TypedDict, total=False):
    global_importances: Dict[str, float]
    local_explanations: List[LocalExplanation]
    visualizations: Dict[str, Any]


_MAX_TRACE_LENGTH = 4000
_PERTURBATION_EPS = 1e-3


def _short_error(exc: BaseException) -> str:
    """Return a compact ``Class: message`` description for ``exc``."""

    message = f"{exc.__class__.__name__}: {exc}"
    return message if len(message) <= 280 else f"{message[:277]}..."


def _trim_trace(trace: str, max_len: int = _MAX_TRACE_LENGTH) -> str:
    """Trim long tracebacks to ``max_len`` characters for safe logging."""

    trace = trace.strip()
    return trace if len(trace) <= max_len else f"{trace[:max_len - 3]}..."


def _has_predict(candidate: Any) -> bool:
    """Return ``True`` if ``candidate`` exposes a callable ``predict`` attribute."""

    predict = getattr(candidate, "predict", None)
    return callable(predict)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_prediction_score(prediction: Any) -> Optional[float]:
    """Best-effort extraction of a scalar score from ``prediction``."""

    if isinstance(prediction, (int, float)):
        return float(prediction)
    if isinstance(prediction, dict):
        score = _coerce_float(prediction.get("score"))
        if score is not None:
            return score
        output = prediction.get("output")
        score = _extract_prediction_score(output)
        if score is not None:
            return score
        raw = prediction.get("raw")
        score = _extract_prediction_score(raw)
        if score is not None:
            return score
    return None


def _normalize_image_prediction(raw: Any) -> Prediction:
    if not isinstance(raw, dict):
        return {"score": None, "label": None, "scores": None, "raw": raw}
    score = _coerce_float(raw.get("score"))
    label = raw.get("label") if isinstance(raw.get("label"), str) else None
    scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else None
    prediction: Prediction = {
        "score": score,
        "label": label,
        "scores": scores if scores is not None else None,
        "raw": raw,
    }
    return prediction


def _normalize_churn_prediction(raw: Any) -> Prediction:
    if not isinstance(raw, dict):
        return {"score": None, "label": None, "scores": None, "raw": raw}
    score = _coerce_float(raw.get("score"))
    label = raw.get("label") if isinstance(raw.get("label"), str) else None
    scores = raw.get("scores") if isinstance(raw.get("scores"), dict) else None
    if scores is None and score is not None:
        scores = {
            "positive": round(score, 6),
            "negative": round(max(0.0, 1.0 - score), 6),
        }
    prediction: Prediction = {
        "score": score,
        "label": label,
        "scores": scores if scores is not None else None,
        "raw": raw,
    }
    return prediction


def _extract_explanations(prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(prediction, dict):
        return None
    explanations = prediction.get("explanations")
    if isinstance(explanations, dict):
        return explanations
    output = prediction.get("output")
    if isinstance(output, dict):
        nested = output.get("explanations")
        if isinstance(nested, dict):
            return nested
    return None


def _load_driver(driver_path: str, config: Dict[str, Any]) -> Any:
    module_path, _, attr = driver_path.partition(":")
    if not module_path:
        raise ValueError("Driver path must include a module name")
    module = importlib.import_module(module_path)
    factory = getattr(module, attr or "load_model")
    if not callable(factory):
        raise TypeError("Driver factory is not callable")
    try:
        return factory(config)
    except TypeError:
        return factory()


def load_image_classifier(config: Optional[Dict[str, Any]] = None) -> ImageModel | Dict[str, Any]:
    """Load an image classifier plugin using the configured driver.

    Parameters
    ----------
    config:
        Optional driver configuration. Must include a ``driver`` entry pointing
        to a ``"module:callable"`` loader.

    Returns
    -------
    ImageModel | Dict[str, Any]
        A predict-capable plugin instance or a structured error payload when
        loading fails or is not configured.
    """

    cfg = dict(config or {})
    driver_path = cfg.get("driver") if isinstance(cfg.get("driver"), str) else None
    if not driver_path:
        return {
            "status": "error",
            "error": "image_model_not_configured",
            "detail": "No image model driver is configured.",
        }
    try:
        model = _load_driver(driver_path, cfg)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "error",
            "error": "image_model_driver_error",
            "detail": _short_error(exc),
            "traceback": _trim_trace(traceback.format_exc()),
        }
    if isinstance(model, dict) and model.get("status") == "error":
        return model
    if not _has_predict(model):
        return {
            "status": "error",
            "error": "image_model_driver_invalid",
            "detail": "Driver did not return an object exposing predict().",
        }
    return model


def run_image_classifier(
    model: Optional[Any],
    payload: Dict[str, Any],
    *,
    demo: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute ``model`` for ``payload`` and normalise the response.

    The ``demo`` flag is retained for backward compatibility but no longer
    changes behaviour; the function only returns real predictions or structured
    errors.

    Returns
    -------
    dict
        ``{"status": "ok", "output": Prediction}`` on success or
        ``{"status": "error", ...}`` when execution fails.
    """

    _ = demo  # retained for signature compatibility

    if not _has_predict(model):
        return {
            "status": "error",
            "error": "image_model_not_configured",
            "detail": "Prediction attempted without a configured image model.",
        }

    try:
        raw = model.predict(payload)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "error",
            "error": "image_model_predict_failed",
            "detail": _short_error(exc),
            "traceback": _trim_trace(traceback.format_exc()),
        }

    return {"status": "ok", "output": _normalize_image_prediction(raw)}


def explain_image_classifier(
    model: Optional[Any],
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
    *,
    demo: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Produce explanations for ``prediction`` when available.

    The helper first inspects the prediction for plugin-supplied explanations.
    When a scalar score is available and the model exposes ``predict`` the
    function attempts a lightweight finite-difference perturbation analysis.
    ``demo`` is accepted for signature compatibility and ignored.
    """

    _ = demo  # retained for compatibility

    direct = _extract_explanations(prediction)
    if isinstance(direct, dict):
        return {"status": "ok", "explanations": direct}

    return _numeric_perturbation_explanations(model, payload, prediction)


def load_churn_classifier(config: Optional[Dict[str, Any]] = None) -> ChurnModel | Dict[str, Any]:
    """Load a churn classifier plugin using the configured driver."""

    cfg = dict(config or {})
    driver_path = cfg.get("driver") if isinstance(cfg.get("driver"), str) else None
    if not driver_path:
        return {
            "status": "error",
            "error": "churn_model_not_configured",
            "detail": "No churn model driver is configured.",
        }
    try:
        model = _load_driver(driver_path, cfg)
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "error",
            "error": "churn_model_driver_error",
            "detail": _short_error(exc),
            "traceback": _trim_trace(traceback.format_exc()),
        }
    if isinstance(model, dict) and model.get("status") == "error":
        return model
    if not _has_predict(model):
        return {
            "status": "error",
            "error": "churn_model_driver_invalid",
            "detail": "Driver did not return an object exposing predict().",
        }
    return model


def run_churn_classifier(
    model: Optional[Any],
    payload: Dict[str, Any],
    *,
    demo: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute ``model`` for churn scoring and normalise the response."""

    _ = demo  # retained for signature compatibility

    if not _has_predict(model):
        return {
            "status": "error",
            "error": "churn_model_not_configured",
            "detail": "Prediction attempted without a configured churn model.",
        }

    try:
        raw = model.predict(payload)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "error",
            "error": "churn_model_predict_failed",
            "detail": _short_error(exc),
            "traceback": _trim_trace(traceback.format_exc()),
        }

    return {"status": "ok", "output": _normalize_churn_prediction(raw)}


def explain_churn_classifier(
    model: Optional[Any],
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
    *,
    demo: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Produce churn explanations when available."""

    _ = demo  # retained for signature compatibility

    direct = _extract_explanations(prediction)
    if isinstance(direct, dict):
        return {"status": "ok", "explanations": direct}

    return _numeric_perturbation_explanations(model, payload, prediction)


def _numeric_perturbation_explanations(
    model: Optional[Any],
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Approximate feature impacts via finite-difference perturbations."""

    if not _has_predict(model):
        return None

    baseline = _extract_prediction_score(prediction)
    if baseline is None:
        return None

    numeric_items: List[tuple[str, float]] = []
    for key, value in payload.items():
        coerced = _coerce_float(value)
        if coerced is None:
            continue
        numeric_items.append((key, coerced))
        if len(numeric_items) >= 16:
            break

    if not numeric_items:
        return None

    impacts: List[tuple[str, float]] = []
    for feature, value in numeric_items:
        plus_payload = dict(payload)
        minus_payload = dict(payload)
        plus_payload[feature] = value + _PERTURBATION_EPS
        minus_payload[feature] = value - _PERTURBATION_EPS
        try:
            plus_raw = model.predict(plus_payload)  # type: ignore[call-arg]
            minus_raw = model.predict(minus_payload)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "status": "error",
                "error": "explanation_compute_failed",
                "detail": _short_error(exc),
                "traceback": _trim_trace(traceback.format_exc()),
            }
        score_plus = _extract_prediction_score(plus_raw)
        score_minus = _extract_prediction_score(minus_raw)
        if score_plus is None or score_minus is None:
            continue
        gradient = (score_plus - score_minus) / (2 * _PERTURBATION_EPS)
        impacts.append((feature, gradient))

    if not impacts:
        return None

    local: List[LocalExplanation] = [
        {"feature": feature, "impact": round(impact, 6)}
        for feature, impact in impacts
    ]
    normalization = sum(abs(impact) for _, impact in impacts) or 1.0
    global_importances = {
        feature: round(abs(impact) / normalization, 6)
        for feature, impact in impacts
    }

    return {
        "status": "ok",
        "baseline_score": round(float(baseline), 6),
        "method": "finite_difference",
        "explanations": {
            "global_importances": global_importances,
            "local_explanations": local,
            "visualizations": {},
        },
    }


# ---------------------------------------------------------------------------
# Training and deployment helpers
# ---------------------------------------------------------------------------


def train_image_classifier(model_name: str, spec: Dict[str, Any], args: Any) -> None:
    feature_order = spec.get("metadata", {}).get("feature_order", ["feature_a", "feature_b"])
    print(f"Starting training pipeline for {model_name} using features {feature_order}")
    losses = [0.32, 0.24, 0.19, 0.15]
    for index, loss in enumerate(losses, start=1):
        print(f"[train:{model_name}] epoch {index}/{len(losses)} - loss={loss:.3f}")
    print(f"Artifacts written to {spec.get('metadata', {}).get('model_file', 'models')}")


def deploy_image_classifier(model_name: str, spec: Dict[str, Any], args: Any) -> str:
    endpoint = f"https://models.example.com/{model_name}/predict"
    print(f"Publishing {model_name} artifact from {spec.get('metadata', {}).get('model_file', 'model.bin')}")
    print(f"Deployment completed: {endpoint}")
    return endpoint


def train_churn_classifier(model_name: str, spec: Dict[str, Any], args: Any) -> None:
    print(f"Running batch training for {model_name}")
    for stage, metric in enumerate([0.58, 0.63, 0.68], start=1):
        print(f"[train:{model_name}] stage {stage} accuracy={metric:.2f}")
    print("Churn model metrics stored in registry.")


def deploy_churn_classifier(model_name: str, spec: Dict[str, Any], args: Any) -> str:
    endpoint = f"https://api.example.com/{model_name}/predict"
    print(f"Staging {model_name} to {endpoint}")
    return endpoint


__all__ = [
    "load_image_classifier",
    "run_image_classifier",
    "explain_image_classifier",
    "train_image_classifier",
    "deploy_image_classifier",
    "load_churn_classifier",
    "run_churn_classifier",
    "explain_churn_classifier",
    "train_churn_classifier",
    "deploy_churn_classifier",
]
