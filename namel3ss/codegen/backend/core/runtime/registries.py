"""Registry initialisation helpers for the runtime module."""

from __future__ import annotations

from typing import Any, Dict, List

from namel3ss.codegen.backend.state import BackendState
from ..utils import _assign_literal
from .pages import _page_to_dict


def render_registries_block(
    state: BackendState,
    configured_model_registry: Dict[str, Any],
    *,
    embed_insights: bool,
    enable_realtime: bool,
) -> str:
    """Generate code that initialises maps consumed by the runtime."""
    registries: List[str] = [
        _assign_literal("APP", "Dict[str, Any]", state.app),
        _assign_literal("DATASETS", "Dict[str, Dict[str, Any]]", state.datasets),
        _assign_literal("FRAMES", "Dict[str, Dict[str, Any]]", state.frames),
        _assign_literal("CONNECTORS", "Dict[str, Dict[str, Any]]", state.connectors),
        _assign_literal("AI_CONNECTORS", "Dict[str, Dict[str, Any]]", state.ai_connectors),
        _assign_literal("AI_MODELS", "Dict[str, Dict[str, Any]]", state.ai_models),
        _assign_literal("LLM_REGISTRY", "Dict[str, Dict[str, Any]]", state.llms),
        _assign_literal("TOOL_REGISTRY", "Dict[str, Dict[str, Any]]", state.tools),
        _assign_literal("AI_MEMORIES", "Dict[str, Dict[str, Any]]", state.memories),
        _assign_literal("AI_PROMPTS", "Dict[str, Dict[str, Any]]", state.prompts),
        _assign_literal("AI_TEMPLATES", "Dict[str, Dict[str, Any]]", state.templates),
        _assign_literal("AI_CHAINS", "Dict[str, Dict[str, Any]]", state.chains),
        _assign_literal("PLANNERS", "Dict[str, Dict[str, Any]]", state.planners),
        _assign_literal("PLANNING_WORKFLOWS", "Dict[str, Dict[str, Any]]", state.planning_workflows),
        _assign_literal("AGENT_DEFS", "Dict[str, Dict[str, Any]]", state.agents),
        _assign_literal("AGENT_GRAPHS", "Dict[str, Dict[str, Any]]", state.graphs),
        _assign_literal("AI_EXPERIMENTS", "Dict[str, Dict[str, Any]]", state.experiments),
        _assign_literal("TRAINING_JOBS", "Dict[str, Dict[str, Any]]", state.training_jobs),
        _assign_literal("TUNING_JOBS", "Dict[str, Dict[str, Any]]", state.tuning_jobs),
        _assign_literal("INSIGHTS", "Dict[str, Dict[str, Any]]", state.insights),
        _assign_literal("CRUD_RESOURCES", "Dict[str, Dict[str, Any]]", state.crud_resources),
        _assign_literal("EVALUATORS", "Dict[str, Dict[str, Any]]", state.evaluators),
        _assign_literal("METRICS", "Dict[str, Dict[str, Any]]", state.metrics),
        _assign_literal("GUARDRAILS", "Dict[str, Dict[str, Any]]", state.guardrails),
        _assign_literal("APP_QUERIES", "Dict[str, Dict[str, Any]]", state.queries),
        _assign_literal("KNOWLEDGE_MODULES", "Dict[str, Dict[str, Any]]", state.knowledge_modules),
        _assign_literal(
            "MODEL_REGISTRY",
            "Dict[str, Dict[str, Any]]",
            configured_model_registry,
        ),
        "MODEL_CACHE: Dict[str, Any] = {}",
        "MODEL_LOADERS: Dict[str, Callable[[str, Dict[str, Any]], Any]] = {}",
        "MODEL_RUNNERS: Dict[str, Callable[[str, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}",
        "MODEL_EXPLAINERS: Dict[str, Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}",
        _assign_literal("MODELS", "Dict[str, Dict[str, Any]]", state.models),
        _assign_literal(
            "PAGES",
            "List[Dict[str, Any]]",
            [_page_to_dict(page) for page in state.pages],
        ),
        "PAGE_SPEC_BY_SLUG: Dict[str, Dict[str, Any]] = {page['slug']: page for page in PAGES}",
        _assign_literal("ENV_KEYS", "List[str]", state.env_keys),
        f"EMBED_INSIGHTS: bool = {'True' if embed_insights else 'False'}",
        f"REALTIME_ENABLED: bool = {'True' if enable_realtime else 'False'}",
        "CONNECTOR_DRIVERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[List[Dict[str, Any]]]]] = {}",
        "DATASET_TRANSFORMS: Dict[str, Callable[[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]], List[Dict[str, Any]]]] = {}",
        "TRAINING_JOB_HISTORY: Dict[str, List[Dict[str, Any]]] = defaultdict(list)",
        "TUNING_JOB_HISTORY: Dict[str, List[Dict[str, Any]]] = defaultdict(list)",
    ]
    return "\n".join(registries).rstrip()


__all__ = ["render_registries_block"]
