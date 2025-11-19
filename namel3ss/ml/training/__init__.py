"""Training backend abstractions for Namel3ss."""

from __future__ import annotations

from .backends import (
    LocalTrainingBackend,
    RayTrainingBackend,
    TrainingBackend,
    TrainingPlan,
    TrainingRunResult,
    register_training_backend,
    registered_training_backends,
    resolve_training_backend,
    resolve_training_plan,
)

# Import production backends to register them
from . import production_backends  # noqa: F401

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
