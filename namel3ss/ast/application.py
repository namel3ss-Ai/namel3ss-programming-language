"""Application-level AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Theme
from .ai import (
    AIModel,
    Chain,
    Connector,
    Memory,
    Prompt,
    Template,
    TrainingJob,
    TuningJob,
    LLMDefinition,
    ToolDefinition,
    RLHFJob,
)
from .rag import IndexDefinition, RagPipelineDefinition
from .agents import AgentDefinition, GraphDefinition
from .experiments import Experiment
from .datasets import Dataset, RefreshPolicy
from .frames import Frame
from .insights import Insight
from .models import Model
from .pages import PageStatement, VariableAssignment
from .crud import CrudResource
from .eval import Evaluator, Metric, Guardrail, EvalSuiteDefinition
from .policy import PolicyDefinition
from .expressions import FunctionDef, RuleDef
from .logic import KnowledgeModule, LogicQuery


@dataclass
class Page:
    name: str
    route: str
    statements: List[PageStatement] = field(default_factory=list)
    reactive: bool = False
    refresh_policy: Optional[RefreshPolicy] = None
    layout: Dict[str, Any] = field(default_factory=dict)


@dataclass
class App:
    name: str
    database: Optional[str] = None
    theme: Theme = field(default_factory=Theme)
    variables: List[VariableAssignment] = field(default_factory=list)
    datasets: List[Dataset] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    pages: List[Page] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    models: List[Model] = field(default_factory=list)
    connectors: List[Connector] = field(default_factory=list)
    ai_models: List[AIModel] = field(default_factory=list)
    llms: List[LLMDefinition] = field(default_factory=list)  # First-class LLM definitions
    tools: List[ToolDefinition] = field(default_factory=list)  # First-class tool definitions
    prompts: List[Prompt] = field(default_factory=list)
    memories: List[Memory] = field(default_factory=list)
    templates: List[Template] = field(default_factory=list)
    chains: List[Chain] = field(default_factory=list)
    experiments: List[Experiment] = field(default_factory=list)
    crud_resources: List[CrudResource] = field(default_factory=list)
    evaluators: List[Evaluator] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    guardrails: List[Guardrail] = field(default_factory=list)
    eval_suites: List[EvalSuiteDefinition] = field(default_factory=list)
    training_jobs: List[TrainingJob] = field(default_factory=list)
    tuning_jobs: List[TuningJob] = field(default_factory=list)
    rlhf_jobs: List[RLHFJob] = field(default_factory=list)  # RLHF training jobs
    indices: List[IndexDefinition] = field(default_factory=list)  # RAG retrieval indices
    rag_pipelines: List[RagPipelineDefinition] = field(default_factory=list)  # RAG pipeline definitions
    agents: List[AgentDefinition] = field(default_factory=list)  # Agent definitions
    graphs: List[GraphDefinition] = field(default_factory=list)  # Multi-agent graph definitions
    policies: List[PolicyDefinition] = field(default_factory=list)  # Safety policies
    functions: List[FunctionDef] = field(default_factory=list)  # User-defined functions
    rules: List[RuleDef] = field(default_factory=list)  # Logic programming rules
    knowledge_modules: List[KnowledgeModule] = field(default_factory=list)  # Knowledge bases (facts + rules)
    queries: List[LogicQuery] = field(default_factory=list)  # Logic queries


__all__ = [
    "Page",
    "App",
]
