"""Experiment encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, Set, TYPE_CHECKING

from .expressions import _encode_value, _expression_to_runtime, _expression_to_source

if TYPE_CHECKING:
    from ....ast import Experiment, ExperimentDataConfig


def _encode_experiment(experiment: "Experiment", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an experiment definition for backend state."""
    variants = {
        name: {
            "expression": _expression_to_source(value),
            "expression_expr": _expression_to_runtime(value),
        }
        for name, value in (experiment.variants or {}).items()
    }
    metrics_list = list(experiment.metrics or [])
    data_config_encoded = None
    if experiment.data_config:
        data_config_encoded = _encode_experiment_data_config(experiment.data_config, env_keys)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "variants": variants,
        "metrics": metrics_list,
        "data_config": data_config_encoded,
        "target": experiment.target,
        "metadata": _encode_value(experiment.metadata, env_keys),
    }


def _encode_experiment_data_config(config: "ExperimentDataConfig", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode experiment data configuration."""
    return {
        "source": config.source,
        "split": dict(config.split or {}),
        "filters": _encode_value(config.filters, env_keys),
        "stratify": config.stratify,
        "metadata": _encode_value(config.metadata, env_keys),
    }
