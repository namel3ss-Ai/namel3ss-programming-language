"""Default model registry definitions and helpers."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)

MODEL_REGISTRY_ENV = "NAMEL3SS_MODEL_REGISTRY"
MODEL_ROOT_ENV = "NAMEL3SS_MODEL_ROOT"

DEFAULT_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "churn_classifier": {
        "type": "sklearn",
        "framework": "scikit-learn",
        "version": "v1",
        "metrics": {"accuracy": 0.91},
        "metadata": {
            "owner": "team_data",
            "description": "Customer churn predictor",
            "coefficients": {"tenure": -0.15, "spend": -0.05, "support_calls": 0.25},
            "intercept": -0.2,
            "loader": "namel3ss.ml.hooks:load_churn_classifier",
            "runner": "namel3ss.ml.hooks:run_churn_classifier",
            "explainer": "namel3ss.ml.hooks:explain_churn_classifier",
            "trainer": "namel3ss.ml.hooks:train_churn_classifier",
            "deployer": "namel3ss.ml.hooks:deploy_churn_classifier",
        },
    },
    "image_classifier": {
        "type": "deep_learning",
        "framework": "pytorch",
        "version": "v1",
        "metrics": {"accuracy": 0.94, "loss": 0.12},
        "metadata": {
            "input_shape": [224, 224, 3],
            "model_file": "models/image_classifier.pt",
            "feature_order": ["feature_a", "feature_b"],
            "weights": [0.7, 0.3],
            "bias": 0.05,
            "threshold": 0.5,
            "loader": "namel3ss.ml.hooks:load_image_classifier",
            "runner": "namel3ss.ml.hooks:run_image_classifier",
            "explainer": "namel3ss.ml.hooks:explain_image_classifier",
            "trainer": "namel3ss.ml.hooks:train_image_classifier",
            "deployer": "namel3ss.ml.hooks:deploy_image_classifier",
        },
    },
}


def get_default_model_registry() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the default model registry."""

    return copy.deepcopy(DEFAULT_MODEL_REGISTRY)


def _coerce_mapping(payload: Any) -> Dict[str, Any]:
    """Return a shallow dict copy when ``payload`` is a mapping."""
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _merge_registry_entry(base: MutableMapping[str, Any], update_entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``base`` merged with ``update_entry`` while preserving nested metadata/metrics."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in update_entry.items():
        if key == "metadata" and isinstance(value, Mapping):
            metadata = _coerce_mapping(merged.get("metadata"))
            metadata.update(value)
            merged["metadata"] = metadata
        elif key == "metrics" and isinstance(value, Mapping):
            metrics = _coerce_mapping(merged.get("metrics"))
            metrics.update(value)
            merged["metrics"] = metrics
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_registries(
    base: Mapping[str, Dict[str, Any]],
    overrides: Optional[Mapping[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Merge two registry mappings, taking nested metadata/metrics into account."""
    if not overrides:
        return {key: copy.deepcopy(value) for key, value in base.items()}

    merged: Dict[str, Dict[str, Any]] = {
        key: copy.deepcopy(value) for key, value in base.items()
    }
    for key, payload in overrides.items():
        if not isinstance(payload, Mapping):
            continue
        current = merged.get(key)
        if current is None:
            merged[key] = copy.deepcopy(dict(payload))
        else:
            merged[key] = _merge_registry_entry(current, payload)
    return merged


def _resolve_registry_path(source: str) -> Optional[Path]:
    """Resolve ``source`` to an existing path, consulting ``NAMEL3SS_MODEL_ROOT`` when needed."""
    path = Path(source)
    if path.exists():
        return path
    if not path.is_absolute():
        root = os.getenv(MODEL_ROOT_ENV)
        if root:
            candidate = Path(root) / source
            if candidate.exists():
                return candidate
    return None


def _parse_registry_payload(raw: str) -> Dict[str, Dict[str, Any]]:
    """Parse JSON/YAML registry overrides, returning an empty mapping on failure."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        if yaml is None:
            logger.debug("Registry payload is not JSON and PyYAML is unavailable")
            return {}
        try:
            parsed = yaml.safe_load(raw)
        except Exception:  # pragma: no cover - YAML parsing failure
            logger.exception("Failed to parse model registry payload")
            return {}
    if not isinstance(parsed, Mapping):
        logger.warning("Model registry payload is not a mapping; ignoring override")
        return {}
    return {key: _coerce_mapping(value) for key, value in parsed.items()}


def _load_registry_override(source: str) -> Dict[str, Dict[str, Any]]:
    """Load overrides from a file path or inline JSON/YAML string."""
    path = _resolve_registry_path(source)
    if path is not None:
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover - IO failure
            logger.exception("Failed to read model registry override from %s", path)
            return {}
        return _parse_registry_payload(raw)
    return _parse_registry_payload(source)


def load_model_registry(
    overrides: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the active model registry, applying environment overrides.

    The registry is constructed by taking :data:`DEFAULT_MODEL_REGISTRY`,
    applying any override declared via the ``NAMEL3SS_MODEL_REGISTRY``
    environment variable (JSON/YAML payload or file path), and finally
    merging any explicit ``overrides`` provided by the caller.
    """

    registry = get_default_model_registry()

    env_override = os.getenv(MODEL_REGISTRY_ENV)
    if env_override:
        merged = _load_registry_override(env_override)
        if merged:
            registry = _merge_registries(registry, merged)

    if overrides:
        registry = _merge_registries(registry, overrides)

    return registry

