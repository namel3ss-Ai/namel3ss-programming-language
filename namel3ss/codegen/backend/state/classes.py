"""Dataclasses for backend state encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PageComponent:
    """Serializable representation of a page component."""

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None


@dataclass
class PageSpec:
    """Encoded data for a page required by the backend generator."""

    name: str
    route: str
    slug: str
    index: int
    api_path: str
    reactive: bool = False
    refresh_policy: Optional[Dict[str, Any]] = None
    components: List[PageComponent] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendState:
    """Container for backend-facing data derived from the AST."""

    app: Dict[str, Any]
    datasets: Dict[str, Dict[str, Any]]
    frames: Dict[str, Dict[str, Any]]
    connectors: Dict[str, Dict[str, Any]]
    ai_connectors: Dict[str, Dict[str, Any]]
    ai_models: Dict[str, Dict[str, Any]]
    llms: Dict[str, Dict[str, Any]]  # First-class LLM definitions
    tools: Dict[str, Dict[str, Any]]  # First-class Tool definitions
    indices: Dict[str, Dict[str, Any]]  # RAG index definitions
    rag_pipelines: Dict[str, Dict[str, Any]]  # RAG pipeline definitions
    memories: Dict[str, Dict[str, Any]]
    prompts: Dict[str, Dict[str, Any]]
    insights: Dict[str, Dict[str, Any]]
    models: Dict[str, Dict[str, Any]]
    templates: Dict[str, Dict[str, Any]]
    chains: Dict[str, Dict[str, Any]]
    agents: Dict[str, Dict[str, Any]]  # Agent definitions
    graphs: Dict[str, Dict[str, Any]]  # Multi-agent graph definitions
    experiments: Dict[str, Dict[str, Any]]
    training_jobs: Dict[str, Dict[str, Any]]
    tuning_jobs: Dict[str, Dict[str, Any]]
    crud_resources: Dict[str, Dict[str, Any]]
    evaluators: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Dict[str, Any]]
    guardrails: Dict[str, Dict[str, Any]]
    eval_suites: Dict[str, Dict[str, Any]]
    queries: Dict[str, Dict[str, Any]]  # Logic queries
    knowledge_modules: Dict[str, Dict[str, Any]]  # Knowledge bases
    pages: List[PageSpec]
    env_keys: List[str]
