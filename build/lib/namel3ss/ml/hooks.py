"""Built-in training/deployment/inference hooks for demo models."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


def _ordered_features(payload: Dict[str, Any], order: Iterable[str]) -> List[float]:
    values: List[float] = []
    for key in order:
        values.append(float(payload.get(key, 0.0)))
    return values


# ---------------------------------------------------------------------------
# Image classifier hooks (deep learning placeholder)
# ---------------------------------------------------------------------------


def load_image_classifier(model_name: str, model_spec: Dict[str, Any]) -> Dict[str, Any]:
    metadata = model_spec.get("metadata", {})
    weights = metadata.get("weights", [0.6, 0.4])
    bias = float(metadata.get("bias", 0.05))
    feature_order = metadata.get("feature_order") or ["feature_a", "feature_b"]
    return {
        "model": model_name,
        "weights": [float(w) for w in weights],
        "bias": bias,
        "feature_order": list(feature_order),
    }


def run_image_classifier(
    model_name: str,
    model_instance: Dict[str, Any],
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = model_spec.get("metadata", {})
    threshold = float(metadata.get("threshold", 0.5))
    instance = model_instance or load_image_classifier(model_name, model_spec)
    order = instance.get("feature_order") or metadata.get("feature_order") or sorted(payload.keys())
    features = _ordered_features(payload, order)
    weights = instance.get("weights") or [0.5 for _ in features]
    bias = float(instance.get("bias", 0.0))
    score = bias + sum(value * weight for value, weight in zip(features, weights))
    label = "Positive" if score >= threshold else "Negative"
    return {
        "score": round(score, 4),
        "label": label,
        "features": {key: value for key, value in zip(order, features)},
    }


def explain_image_classifier(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    feature_values = prediction.get("features") or payload
    score = float(prediction.get("output", {}).get("score", prediction.get("score", 0.0)))
    contributions = {}
    total = sum(abs(value) for value in feature_values.values()) or 1.0
    for key, value in feature_values.items():
        contributions[key] = round(abs(value) / total, 4)
    return {
        "global_importances": contributions,
        "local_explanations": [
            {"feature": key, "impact": round(value, 4)}
            for key, value in feature_values.items()
        ],
        "visualizations": {
            "saliency_map": "base64://image_classifier_saliency",
            "grad_cam": "base64://image_classifier_grad_cam",
        },
        "confidence": round(score, 4),
    }


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


# ---------------------------------------------------------------------------
# Churn classifier hooks (traditional ML placeholder)
# ---------------------------------------------------------------------------


def load_churn_classifier(model_name: str, model_spec: Dict[str, Any]) -> Dict[str, Any]:
    metadata = model_spec.get("metadata", {})
    coefficients = metadata.get("coefficients", {"tenure": -0.15, "spend": -0.05, "support_calls": 0.25})
    intercept = float(metadata.get("intercept", -0.2))
    return {
        "model": model_name,
        "coefficients": {key: float(value) for key, value in coefficients.items()},
        "intercept": intercept,
    }


def run_churn_classifier(
    model_name: str,
    model_instance: Dict[str, Any],
    payload: Dict[str, Any],
    model_spec: Dict[str, Any],
) -> Dict[str, Any]:
    instance = model_instance or load_churn_classifier(model_name, model_spec)
    coefficients = instance.get("coefficients", {})
    intercept = float(instance.get("intercept", 0.0))
    score = intercept
    for feature, weight in coefficients.items():
        score += float(payload.get(feature, 0.0)) * weight
    probability = 1.0 / (1.0 + math.exp(-score))
    label = "Will Churn" if probability >= 0.5 else "Retained"
    return {
        "score": round(probability, 4),
        "label": label,
        "coefficients": coefficients,
    }


def explain_churn_classifier(
    model_name: str,
    payload: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    contributions = {}
    coefficients = prediction.get("coefficients", {})
    total = sum(abs(coefficients.get(key, 0.0)) for key in coefficients) or 1.0
    for feature, weight in coefficients.items():
        contributions[feature] = round(abs(weight) / total, 4)
    return {
        "global_importances": contributions,
        "local_explanations": [
            {
                "feature": feature,
                "impact": round(float(payload.get(feature, 0.0)) * float(coefficients.get(feature, 0.0)), 4),
            }
            for feature in coefficients
        ],
        "visualizations": {
            "feature_weights": coefficients,
        },
    }


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
