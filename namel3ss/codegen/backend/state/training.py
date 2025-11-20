"""Training and tuning job encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, TYPE_CHECKING

from .expressions import _encode_value

if TYPE_CHECKING:
    from ....ast import EarlyStoppingSpec, HyperparamSpec, TrainingComputeSpec, TrainingJob, TuningJob


def _encode_metadata_dict(value: Optional[Dict[str, Any]], env_keys: Set[str]) -> Dict[str, Any]:
    """Encode metadata ensuring it's a dictionary."""
    encoded = _encode_value(value or {}, env_keys)
    if isinstance(encoded, dict):
        return dict(encoded)
    if encoded is None:
        return {}
    return {"value": encoded}


def _encode_training_job(job: "TrainingJob", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a training job definition."""
    from ....ast import TrainingComputeSpec
    
    compute_spec = job.compute or TrainingComputeSpec()
    return {
        "name": job.name,
        "model": job.model,
        "dataset": job.dataset,
        "objective": job.objective,
        "target": job.target,
        "features": list(job.features or []),
        "framework": job.framework,
        "hyperparameters": {
            key: _encode_value(value, env_keys) for key, value in (job.hyperparameters or {}).items()
        },
        "compute": _encode_training_compute_spec(compute_spec, env_keys),
        "split": dict(job.split or {}),
        "validation_split": job.validation_split,
        "early_stopping": _encode_early_stopping_spec(job.early_stopping, env_keys),
        "output_registry": job.output_registry,
        "metrics": list(job.metrics or []),
        "description": job.description,
        "metadata": _encode_metadata_dict(job.metadata, env_keys),
    }


def _encode_training_compute_spec(compute: "TrainingComputeSpec", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode training compute specification."""
    resources = {
        key: _encode_value(value, env_keys) for key, value in (compute.resources or {}).items()
    }
    return {
        "backend": compute.backend or "local",
        "resources": resources,
        "queue": compute.queue,
        "metadata": _encode_metadata_dict(compute.metadata, env_keys),
    }


def _encode_tuning_job(job: "TuningJob", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode a hyperparameter tuning job definition."""
    search_space = {
        name: _encode_hyperparam_spec(spec, env_keys)
        for name, spec in (job.search_space or {}).items()
    }
    return {
        "name": job.name,
        "training_job": job.training_job,
        "strategy": job.strategy,
        "max_trials": job.max_trials,
        "parallel_trials": job.parallel_trials,
        "objective_metric": job.objective_metric,
        "search_space": search_space,
        "early_stopping": _encode_early_stopping_spec(job.early_stopping, env_keys),
        "metadata": _encode_metadata_dict(job.metadata, env_keys),
    }


def _encode_hyperparam_spec(spec: "HyperparamSpec", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode hyperparameter specification."""
    values = [_encode_value(value, env_keys) for value in (spec.values or [])]
    return {
        "type": spec.type,
        "min": spec.min,
        "max": spec.max,
        "values": values,
        "log": bool(spec.log),
        "step": spec.step,
        "metadata": _encode_metadata_dict(spec.metadata, env_keys),
    }


def _encode_early_stopping_spec(spec: Optional["EarlyStoppingSpec"], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
    """Encode early stopping specification."""
    if spec is None:
        return None
    return {
        "metric": spec.metric,
        "patience": spec.patience,
        "min_delta": spec.min_delta,
        "mode": spec.mode,
        "metadata": _encode_metadata_dict(spec.metadata, env_keys),
    }
