"""Helpers for stitching together runtime section snippets."""

from __future__ import annotations

from typing import List

from ..runtime_sections import (
    ACTIONS_SECTION,
    CONFIG_SECTION,
    CONNECTORS_SECTION,
    CONTEXT_SECTION,
    CRUD_SECTION,
    DATASET_SECTION,
    ERRORS_SECTION,
    FRAMES_SECTION,
    INSIGHTS_SECTION,
    LLM_SECTION,
    MODELS_SECTION,
    OBSERVABILITY_SECTION,
    PREDICTION_SECTION,
    PUBSUB_SECTION,
    REGISTRY_SECTION,
    RENDERING_SECTION,
    SECURITY_SECTION,
    STREAMS_SECTION,
    TRAINING_SECTION,
)


def collect_runtime_sections() -> List[str]:
    """Return the ordered runtime section snippets."""
    return [
        CONFIG_SECTION,
        SECURITY_SECTION,
        ERRORS_SECTION,
        OBSERVABILITY_SECTION,
        PUBSUB_SECTION,
        STREAMS_SECTION,
        CONTEXT_SECTION,
        DATASET_SECTION,
        FRAMES_SECTION,
        CRUD_SECTION,
        ACTIONS_SECTION,
        RENDERING_SECTION,
        REGISTRY_SECTION,
        LLM_SECTION,
        MODELS_SECTION,
        TRAINING_SECTION,
        CONNECTORS_SECTION,
        PREDICTION_SECTION,
        INSIGHTS_SECTION,
    ]


__all__ = ["collect_runtime_sections"]
