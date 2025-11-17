"""Helpers for stitching together runtime section snippets."""

from __future__ import annotations

from typing import List

from ..runtime_sections import (
    ACTIONS_SECTION,
    CONFIG_SECTION,
    CONNECTORS_SECTION,
    CONTEXT_SECTION,
    DATASET_SECTION,
    CRUD_SECTION,
    INSIGHTS_SECTION,
    LLM_SECTION,
    MODELS_SECTION,
    PUBSUB_SECTION,
    PREDICTION_SECTION,
    REGISTRY_SECTION,
    RENDERING_SECTION,
    STREAMS_SECTION,
)


def collect_runtime_sections() -> List[str]:
    """Return the ordered runtime section snippets."""
    return [
        CONFIG_SECTION,
        PUBSUB_SECTION,
        STREAMS_SECTION,
        CONTEXT_SECTION,
        DATASET_SECTION,
    CRUD_SECTION,
        ACTIONS_SECTION,
        RENDERING_SECTION,
        REGISTRY_SECTION,
        LLM_SECTION,
        MODELS_SECTION,
        CONNECTORS_SECTION,
        PREDICTION_SECTION,
        INSIGHTS_SECTION,
    ]


__all__ = ["collect_runtime_sections"]
