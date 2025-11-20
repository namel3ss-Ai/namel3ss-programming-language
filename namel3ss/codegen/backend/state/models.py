"""ML model encoding for backend state translation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .datasets import _encode_dataset_transform
from .expressions import _encode_value, _expression_to_source

if TYPE_CHECKING:
    from ....ast import (
        Model,
        ModelDatasetReference,
        ModelDeploymentTarget,
        ModelEvaluationMetric,
        ModelFeatureSpec,
        ModelHyperParameter,
        ModelMonitoringSpec,
        ModelRegistryInfo,
        ModelServingSpec,
        ModelTrainingSpec,
    )


def _encode_model(model: "Model", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode an ML model definition for backend state."""
    return {
        "name": model.name,
        "type": model.model_type,
        "engine": model.engine,
        "task": model.task,
        "description": model.description,
        "options": _encode_value(model.options, env_keys),
        "training": _encode_model_training(model.training, env_keys),
        "features_spec": [_encode_model_feature_spec(feature, env_keys) for feature in model.features_spec],
        "registry": _encode_model_registry(model.registry, env_keys),
        "monitoring": _encode_model_monitoring(model.monitoring, env_keys),
        "serving": _encode_model_serving_spec(model.serving, env_keys),
        "deployments": [_encode_model_deployment(deployment, env_keys) for deployment in model.deployments],
        "tags": list(model.tags or []),
        "metadata": _encode_value(model.metadata, env_keys),
    }


def _encode_model_training(training: "ModelTrainingSpec", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode model training specification."""
    return {
        "source_type": training.source_type,
        "source": training.source,
        "target": training.target,
        "features": list(training.features or []),
        "split": _encode_value(training.split, env_keys),
        "schedule": training.schedule,
        "framework": training.framework,
        "objective": training.objective,
        "loss": training.loss,
        "optimizer": training.optimizer,
        "batch_size": training.batch_size,
        "epochs": training.epochs,
        "learning_rate": training.learning_rate,
        "hyperparameters": [_encode_model_hyperparameter(param, env_keys) for param in training.hyperparameters],
        "datasets": [_encode_model_dataset_reference(reference, env_keys) for reference in training.datasets],
        "transforms": [_encode_dataset_transform(step, env_keys) for step in training.transforms],
        "callbacks": list(training.callbacks or []),
        "metadata": _encode_value(training.metadata, env_keys),
    }


def _encode_model_hyperparameter(parameter: "ModelHyperParameter", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode model hyperparameter."""
    return {
        "name": parameter.name,
        "value": _encode_value(parameter.value, env_keys),
        "tunable": parameter.tunable,
        "search_space": _encode_value(parameter.search_space, env_keys),
    }


def _encode_model_dataset_reference(reference: "ModelDatasetReference", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode model dataset reference."""
    return {
        "role": reference.role,
        "name": reference.name,
        "filters": [_expression_to_source(expr) for expr in reference.filters],
        "options": _encode_value(reference.options, env_keys),
    }


def _encode_model_feature_spec(feature: "ModelFeatureSpec", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode model feature specification."""
    return {
        "name": feature.name,
        "dtype": feature.dtype,
        "role": feature.role,
        "source": feature.source,
        "expression": _expression_to_source(feature.expression),
        "required": feature.required,
        "description": feature.description,
        "stats": _encode_value(feature.stats, env_keys),
        "options": _encode_value(feature.options, env_keys),
    }


def _encode_model_registry(registry: "ModelRegistryInfo", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode model registry information."""
    return {
        "version": registry.version,
        "accuracy": registry.accuracy,
        "metrics": _encode_value(registry.metrics, env_keys),
        "metadata": _encode_value(registry.metadata, env_keys),
        "registry_name": registry.registry_name,
        "owner": registry.owner,
        "stage": registry.stage,
        "last_updated": registry.last_updated,
        "tags": list(registry.tags or []),
        "checks": _encode_value(registry.checks, env_keys),
    }


def _encode_model_monitoring(monitoring: Optional["ModelMonitoringSpec"], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
    """Encode model monitoring specification."""
    if monitoring is None:
        return None
    return {
        "schedule": monitoring.schedule,
        "metrics": [_encode_model_monitoring_metric(metric, env_keys) for metric in monitoring.metrics],
        "alerts": _encode_value(monitoring.alerts, env_keys),
        "drift_thresholds": _encode_value(monitoring.drift_thresholds, env_keys),
    }


def _encode_model_monitoring_metric(metric: "ModelEvaluationMetric", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode model evaluation metric."""
    return {
        "name": metric.name,
        "value": _encode_value(metric.value, env_keys),
        "threshold": _encode_value(metric.threshold, env_keys),
        "goal": metric.goal,
        "higher_is_better": metric.higher_is_better,
        "dataset": metric.dataset,
        "tags": list(metric.tags or []),
        "extras": _encode_value(metric.extras, env_keys),
    }


def _encode_model_serving_spec(serving: Optional["ModelServingSpec"], env_keys: Set[str]) -> Optional[Dict[str, Any]]:
    """Encode model serving specification."""
    if serving is None:
        return None
    return {
        "endpoints": list(serving.endpoints or []),
        "realtime": _encode_value(serving.realtime, env_keys),
        "batch": _encode_value(serving.batch, env_keys),
        "resources": _encode_value(serving.resources, env_keys),
        "options": _encode_value(serving.options, env_keys),
    }


def _encode_model_deployment(deployment: "ModelDeploymentTarget", env_keys: Set[str]) -> Dict[str, Any]:
    """Encode model deployment target."""
    return {
        "name": deployment.name,
        "environment": deployment.environment,
        "strategy": deployment.strategy,
        "options": _encode_value(deployment.options, env_keys),
    }
