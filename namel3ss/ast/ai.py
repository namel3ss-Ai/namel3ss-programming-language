"""AI-centric AST nodes for connectors, templates, prompts, and chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .base import Expression


@dataclass
class Connector:
    name: str
    connector_type: str
    provider: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    @property
    def category(self) -> str:
        return self.connector_type


@dataclass
class AIModel:
    """Declarative handle for a provider-backed AI model."""

    name: str
    provider: str
    model_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Template:
    name: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    """Declarative memory store definition."""

    name: str
    scope: str = "session"
    kind: str = "list"
    max_items: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptField:
    """Structured schema information for prompt inputs/outputs."""

    name: str
    field_type: str = "text"
    required: bool = True
    description: Optional[str] = None
    default: Optional[Any] = None
    enum: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prompt:
    """A named, reusable prompt with typed inputs and outputs."""

    name: str
    model: str
    template: str
    input_fields: List[PromptField] = field(default_factory=list)
    output_fields: List[PromptField] = field(default_factory=list)
    args: List['PromptArgument'] = field(default_factory=list)  # New: typed arguments for parameterized prompts
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    effects: Set[str] = field(default_factory=set)


@dataclass
class StepEvaluationConfig:
    evaluators: List[str] = field(default_factory=list)
    guardrail: Optional[str] = None


@dataclass
class ChainStep:
    kind: str
    target: str
    options: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    stop_on_error: bool = True
    evaluation: Optional[StepEvaluationConfig] = None


@dataclass
class WorkflowIfBlock:
    condition: Expression
    then_steps: List["WorkflowNode"] = field(default_factory=list)
    elif_steps: List[Tuple[Expression, List["WorkflowNode"]]] = field(default_factory=list)
    else_steps: List["WorkflowNode"] = field(default_factory=list)


@dataclass
class WorkflowForBlock:
    loop_var: str
    source_kind: str = "expression"
    source_name: Optional[str] = None
    source_expression: Optional[Expression] = None
    body: List["WorkflowNode"] = field(default_factory=list)
    max_iterations: Optional[int] = None


@dataclass
class WorkflowWhileBlock:
    condition: Expression
    body: List["WorkflowNode"] = field(default_factory=list)
    max_iterations: Optional[int] = None


@dataclass
class Chain:
    name: str
    input_key: str = "input"
    steps: List["WorkflowNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    declared_effect: Optional[str] = None
    effects: Set[str] = field(default_factory=set)


WorkflowNode = Union[ChainStep, WorkflowIfBlock, WorkflowForBlock, WorkflowWhileBlock]


# ---------------------------------------------------------------------------
# Training and tuning primitives
# ---------------------------------------------------------------------------


HyperparameterValue = Union[Expression, Any]


@dataclass
class TrainingComputeSpec:
    backend: str = "local"
    resources: Dict[str, Any] = field(default_factory=dict)
    queue: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    name: str
    model: str
    dataset: str
    objective: str
    hyperparameters: Dict[str, HyperparameterValue] = field(default_factory=dict)
    compute: TrainingComputeSpec = field(default_factory=TrainingComputeSpec)
    output_registry: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperparamSpec:
    type: str
    min: Optional[float] = None
    max: Optional[float] = None
    values: Optional[List[Any]] = None
    log: bool = False
    step: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EarlyStoppingSpec:
    metric: str
    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningJob:
    name: str
    training_job: str
    search_space: Dict[str, HyperparamSpec] = field(default_factory=dict)
    strategy: str = "grid"
    max_trials: int = 1
    parallel_trials: int = 1
    early_stopping: Optional[EarlyStoppingSpec] = None
    objective_metric: str = "loss"
    metadata: Dict[str, Any] = field(default_factory=dict)




@dataclass
class LLMDefinition:
    """
    First-class LLM block defining a language model with provider and configuration.
    
    Example DSL syntax:
        llm chat_gpt_4o {
            provider: openai
            model: gpt-4o
            temperature: 0.7
            max_tokens: 2048
        }
    """
    name: str
    provider: str  # openai, anthropic, vertex, azure_openai, local
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass
class ToolDefinition:
    """
    First-class tool block defining an external tool with typed input/output schemas.
    
    Example DSL syntax:
        tool get_weather {
            type: http
            endpoint: https://api.weather.com/v1/current
            method: GET
            input_schema: {
                location: string,
                units: string
            }
            output_schema: {
                temperature: number,
                conditions: string
            }
            timeout: 10.0
        }
    """
    name: str
    type: str = "http"  # http, python, database, vector_search (extensible)
    endpoint: Optional[str] = None
    method: str = "POST"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[SourceLocation] = None


@dataclass
class PromptArgument:
    """
    Typed argument for parameterized prompt templates.
    
    Example DSL syntax:
        prompt summarize {
            args: {
                text: string,
                max_length: number = 100
            }
            template: "Summarize the following text in {max_length} words: {text}"
        }
    """
    name: str
    arg_type: str  # string, number, boolean, object, array
    required: bool = True
    default: Any = None
    description: Optional[str] = None
    location: Optional[SourceLocation] = None


__all__ = [
    "Connector",
    "AIModel",
    "Template",
    "Memory",
    "PromptField",
    "Prompt",
    "PromptArgument",
    "ChainStep",
    "StepEvaluationConfig",
    "WorkflowIfBlock",
    "WorkflowForBlock",
    "WorkflowWhileBlock",
    "Chain",
    "LLMDefinition",
    "ToolDefinition",
    "TrainingComputeSpec",
    "TrainingJob",
    "HyperparamSpec",
    "EarlyStoppingSpec",
    "TuningJob",
]
