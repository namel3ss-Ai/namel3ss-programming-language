"""Shared machine learning integrations for Namel3ss."""

from __future__ import annotations

from .registry import DEFAULT_MODEL_REGISTRY, get_default_model_registry, load_model_registry
from .hooks import (
    deploy_churn_classifier,
    deploy_image_classifier,
    explain_churn_classifier,
    explain_image_classifier,
    load_churn_classifier,
    load_image_classifier,
    run_churn_classifier,
    run_image_classifier,
    train_churn_classifier,
    train_image_classifier,
)

__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "get_default_model_registry",
    "load_model_registry",
    "deploy_churn_classifier",
    "deploy_image_classifier",
    "explain_churn_classifier",
    "explain_image_classifier",
    "load_churn_classifier",
    "load_image_classifier",
    "run_churn_classifier",
    "run_image_classifier",
    "train_churn_classifier",
    "train_image_classifier",
]
