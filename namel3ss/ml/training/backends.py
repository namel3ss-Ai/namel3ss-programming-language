"""Training backend registry plus default local/ray implementations."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, TypedDict

BackendFactory = Callable[[], "TrainingBackend"]


class TrainingRunResult(TypedDict, total=False):
    """Structured payload returned by training backends."""

    status: str
    job: str
    backend: str
    model: str
    dataset: str
    objective: str
    metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    resources: Dict[str, Any]
    artifacts: Dict[str, Any]
    metadata: Dict[str, Any]
    detail: str
    error: str


class TrainingPlan(TypedDict, total=False):
    """Normalized training inputs derived from a TrainingJob spec."""

    job: str
    backend: str
    model: str
    dataset: str
    objective: str
    description: Optional[str]
    hyperparameters: Dict[str, Any]
    resources: Dict[str, Any]
    metadata: Dict[str, Any]
    metrics: List[str]
    payload: Dict[str, Any]
    output_registry: Optional[str]
    queue: Optional[str]
    overrides: Dict[str, Any]


class TrainingBackend(Protocol):
    """Runtime interface implemented by all training backends."""

    name: str

    def run(self, plan: TrainingPlan, context: Optional[Dict[str, Any]] = None) -> TrainingRunResult:
        """Execute a training job described by ``plan``."""


_BACKEND_FACTORIES: Dict[str, BackendFactory] = {}


def register_training_backend(name: str, factory: BackendFactory) -> None:
    """Register ``factory`` under ``name`` for later resolution."""

    key = (name or "").strip().lower()
    if not key:
        raise ValueError("Training backend name cannot be empty")
    if not callable(factory):
        raise TypeError("Training backend factory must be callable")
    _BACKEND_FACTORIES[key] = factory


def registered_training_backends() -> List[str]:
    """Return the list of registered backend identifiers."""

    return sorted(_BACKEND_FACTORIES.keys())


def resolve_training_backend(name: Optional[str]) -> TrainingBackend:
    """Return the backend instance for ``name`` falling back to ``local``."""

    candidate = (name or "local").strip().lower() or "local"
    factory = _BACKEND_FACTORIES.get(candidate) or _BACKEND_FACTORIES.get("local")
    if factory is None:
        raise RuntimeError("No training backends are registered")
    return factory()


def resolve_training_plan(
    job_spec: Any,
    payload: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> TrainingPlan:
    """Return a normalised training plan combining spec + runtime overrides."""

    env_map = dict(env or os.environ)
    payload_map = dict(payload or {})
    override_map = dict(overrides or {})

    compute_spec = _object_to_dict(_get_field(job_spec, "compute", {}))
    backend_name = (
        override_map.get("backend")
        or payload_map.get("backend")
        or compute_spec.get("backend")
        or "local"
    )
    hyperparameters = _resolve_env_placeholders(
        _object_to_dict(_get_field(job_spec, "hyperparameters", {})), env_map
    )
    hyperparameters = _merge_dicts(
        hyperparameters,
        _resolve_env_placeholders(_object_to_dict(payload_map.get("hyperparameters", {})), env_map),
    )
    hyperparameters = _merge_dicts(
        hyperparameters,
        _resolve_env_placeholders(_object_to_dict(override_map.get("hyperparameters", {})), env_map),
    )

    resources = _resolve_env_placeholders(_object_to_dict(compute_spec.get("resources", {})), env_map)
    resources = _merge_dicts(
        resources,
        _resolve_env_placeholders(_object_to_dict(payload_map.get("resources", {})), env_map),
    )
    resources = _merge_dicts(
        resources,
        _resolve_env_placeholders(_object_to_dict(override_map.get("resources", {})), env_map),
    )

    metadata = _resolve_env_placeholders(_object_to_dict(_get_field(job_spec, "metadata", {})), env_map)
    metadata = _merge_dicts(
        metadata,
        _resolve_env_placeholders(_object_to_dict(payload_map.get("metadata", {})), env_map),
    )
    metadata = _merge_dicts(
        metadata,
        _resolve_env_placeholders(_object_to_dict(override_map.get("metadata", {})), env_map),
    )

    plan: TrainingPlan = {
        "job": str(_get_field(job_spec, "name", "training-job")),
        "backend": str(backend_name or "local"),
        "model": str(_get_field(job_spec, "model", "")),
        "dataset": str(_get_field(job_spec, "dataset", "")),
        "objective": str(_get_field(job_spec, "objective", "")),
        "description": _get_field(job_spec, "description"),
        "hyperparameters": hyperparameters,
        "resources": resources,
        "metadata": metadata,
        "metrics": _as_string_list(_get_field(job_spec, "metrics", [])),
        "payload": dict(payload_map),
        "output_registry": _get_field(job_spec, "output_registry"),
        "queue": compute_spec.get("queue"),
        "overrides": dict(override_map),
    }
    return plan


class LocalTrainingBackend:
    """Deterministic local backend used for development/testing flows."""

    name = "local"

    def run(self, plan: TrainingPlan, context: Optional[Dict[str, Any]] = None) -> TrainingRunResult:
        context_payload = dict(context or {})
        metrics = self._build_metrics(plan)
        artifacts = self._build_artifacts(plan, context_payload)
        metadata = dict(plan.get("metadata", {}))
        metadata.setdefault("executed_at", time.time())
        metadata.setdefault("backend", self.name)
        result: TrainingRunResult = {
            "status": "ok",
            "job": plan.get("job", "training-job"),
            "backend": self.name,
            "model": plan.get("model", ""),
            "dataset": plan.get("dataset", ""),
            "objective": plan.get("objective", ""),
            "hyperparameters": plan.get("hyperparameters", {}),
            "resources": plan.get("resources", {}),
            "metrics": metrics,
            "artifacts": artifacts,
            "metadata": metadata,
        }
        description = plan.get("description")
        if description:
            result.setdefault("detail", description)
        return result

    def _build_metrics(self, plan: TrainingPlan) -> Dict[str, float]:
        names = plan.get("metrics") or ["loss"]
        job = plan.get("job", "training-job")
        hyperparameters = plan.get("hyperparameters", {})
        return {
            name: _deterministic_metric(job, name, hyperparameters, idx)
            for idx, name in enumerate(names, start=1)
        }

    def _build_artifacts(self, plan: TrainingPlan, context: Dict[str, Any]) -> Dict[str, Any]:
        dataset_rows = context.get("dataset_rows")
        sample_count = 0
        if isinstance(dataset_rows, list):
            sample_count = len(dataset_rows)
        elif isinstance(dataset_rows, int):
            sample_count = max(dataset_rows, 0)
        elif isinstance(context.get("dataset_size"), int):
            sample_count = max(context.get("dataset_size", 0), 0)
        checkpoint = f"{plan.get('job', 'training-job')}-{int(time.time())}.ckpt"
        return {
            "checkpoint": checkpoint,
            "samples": sample_count,
            "output_registry": plan.get("output_registry"),
        }


class RayTrainingBackend(LocalTrainingBackend):
    """Ray-based backend that validates the dependency before execution."""

    name = "ray"

    _AUTO = object()

    def __init__(self, ray_module: Any = _AUTO) -> None:
        if ray_module is self._AUTO:
            try:  # pragma: no cover - optional dependency
                import ray  # type: ignore

                ray_module = ray
            except Exception:  # pragma: no cover - handled via error payload
                ray_module = None
        self._ray = ray_module

    def run(self, plan: TrainingPlan, context: Optional[Dict[str, Any]] = None) -> TrainingRunResult:
        if self._ray is None:
            return {
                "status": "error",
                "job": plan.get("job", "training-job"),
                "backend": self.name,
                "error": "ray_backend_unavailable",
                "detail": "Ray backend requires the 'ray' package to be installed.",
            }
        result = super().run(plan, context)
        metadata = dict(result.get("metadata", {}))
        metadata["executor"] = "ray"
        metadata.setdefault("ray_version", getattr(self._ray, "__version__", "unknown"))
        result["metadata"] = metadata
        return result


def _object_to_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if dataclasses.is_dataclass(payload):
        return dataclasses.asdict(payload)
    if hasattr(payload, "__dict__"):
        return {key: getattr(payload, key) for key in vars(payload)}
    return {}


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _merge_dicts(*mappings: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for mapping in mappings:
        for key, value in (mapping or {}).items():
            merged[key] = value
    return merged


def _resolve_env_placeholders(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return env.get(env_name, value)
    if isinstance(value, list):
        return [_resolve_env_placeholders(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(val, env) for key, val in value.items()}
    return value


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _stable_hyperparameter_payload(hyperparameters: Mapping[str, Any]) -> str:
    if not hyperparameters:
        return "{}"
    try:
        return json.dumps(hyperparameters, sort_keys=True, default=str)
    except TypeError:
        converted = {key: str(value) for key, value in hyperparameters.items()}
        return json.dumps(converted, sort_keys=True)


def _deterministic_metric(job: str, metric: str, hyperparameters: Mapping[str, Any], salt: int) -> float:
    payload = f"{job}:{metric}:{salt}:{_stable_hyperparameter_payload(hyperparameters)}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    return round(value, 4)


register_training_backend("local", lambda: LocalTrainingBackend())
register_training_backend("ray", lambda: RayTrainingBackend())

__all__ = [
    "TrainingBackend",
    "TrainingPlan",
    "TrainingRunResult",
    "LocalTrainingBackend",
    "RayTrainingBackend",
    "register_training_backend",
    "registered_training_backends",
    "resolve_training_backend",
    "resolve_training_plan",
]
