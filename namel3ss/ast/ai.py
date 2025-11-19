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
    args: List['PromptArgument'] = field(default_factory=list)  # Typed arguments for parameterized prompts
    output_schema: Optional['OutputSchema'] = None  # Structured output schema
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
    policy_name: Optional[str] = None  # Reference to safety policy


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
    target: Optional[str] = None
    features: List[str] = field(default_factory=list)
    framework: Optional[str] = None
    hyperparameters: Dict[str, HyperparameterValue] = field(default_factory=dict)
    compute: TrainingComputeSpec = field(default_factory=TrainingComputeSpec)
    split: Dict[str, float] = field(default_factory=dict)
    validation_split: Optional[float] = None
    early_stopping: Optional[EarlyStoppingSpec] = None
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
                max_length: int = 100
            }
            template: "Summarize the following text in {max_length} words: {text}"
        }
    """
    name: str
    arg_type: str  # string, int, float, bool, list, object
    required: bool = True
    default: Any = None
    description: Optional[str] = None
    location: Optional[SourceLocation] = None


@dataclass
class EnumType:
    """
    Enum type constraint for output schema fields.
    
    Example:
        enum["billing", "technical", "account", "other"]
    """
    values: List[str]
    
    def __str__(self) -> str:
        vals = ', '.join(f'"{v}"' for v in self.values)
        return f"enum[{vals}]"


@dataclass
class OutputFieldType:
    """
    Type specification for an output schema field.
    
    Supports:
    - Primitives: string, int, float, bool
    - Collections: list[T]
    - Nested objects: {field: type, ...}
    - Enums: enum["val1", "val2"]
    """
    base_type: str  # string, int, float, bool, list, object, enum
    element_type: Optional['OutputFieldType'] = None  # For list[T]
    enum_values: Optional[List[str]] = None  # For enum types
    nested_fields: Optional[List['OutputField']] = None  # For object types
    nullable: bool = False
    
    def __str__(self) -> str:
        if self.base_type == "enum" and self.enum_values:
            vals = ', '.join(f'"{v}"' for v in self.enum_values)
            return f"enum[{vals}]"
        elif self.base_type == "list" and self.element_type:
            return f"list[{self.element_type}]"
        elif self.base_type == "object" and self.nested_fields:
            return "object{...}"
        suffix = "?" if self.nullable else ""
        return f"{self.base_type}{suffix}"


@dataclass
class OutputField:
    """
    A single field in an output schema.
    
    Example:
        category: enum["billing", "technical"]
        confidence: float
        tags: list[string]
    """
    name: str
    field_type: OutputFieldType
    required: bool = True
    description: Optional[str] = None
    location: Optional[SourceLocation] = None


@dataclass
class OutputSchema:
    """
    Structured output schema for a prompt.
    
    Example DSL:
        output_schema: {
            category: enum["billing", "technical", "account"],
            urgency: enum["low", "medium", "high"],
            needs_handoff: bool,
            confidence: float
        }
    """
    fields: List[OutputField]
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for LLM providers."""
        properties = {}
        required = []
        
        for field in self.fields:
            properties[field.name] = self._field_type_to_json_schema(field.field_type)
            if field.description:
                properties[field.name]["description"] = field.description
            if field.required:
                required.append(field.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
    
    def _field_type_to_json_schema(self, ft: OutputFieldType) -> Dict[str, Any]:
        """Convert OutputFieldType to JSON Schema type definition."""
        if ft.base_type == "enum" and ft.enum_values:
            return {"type": "string", "enum": ft.enum_values}
        elif ft.base_type == "list" and ft.element_type:
            return {
                "type": "array",
                "items": self._field_type_to_json_schema(ft.element_type)
            }
        elif ft.base_type == "object" and ft.nested_fields:
            nested_props = {}
            nested_required = []
            for nf in ft.nested_fields:
                nested_props[nf.name] = self._field_type_to_json_schema(nf.field_type)
                if nf.required:
                    nested_required.append(nf.name)
            return {
                "type": "object",
                "properties": nested_props,
                "required": nested_required
            }
        else:
            # Map N3 types to JSON Schema types
            type_map = {
                "string": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
            }
            return {"type": type_map.get(ft.base_type, "string")}


__all__ = [
    "Connector",
    "AIModel",
    "Template",
    "Memory",
    "PromptField",
    "Prompt",
    "PromptArgument",
    "EnumType",
    "OutputFieldType",
    "OutputField",
    "OutputSchema",
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
