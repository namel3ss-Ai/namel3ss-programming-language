"""
Centralized validation for AI subsystem constructs.

This module provides production-grade validation functions for all AI-related
AST nodes. All validation logic is centralized here to ensure consistency,
maintainability, and comprehensive error reporting.

Design principles:
- Accept fully typed objects
- Perform deep invariant checks
- Raise domain-specific AIValidationError on failure
- Provide actionable error messages with hints
- No I/O or side effects - pure validation logic
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .errors import AIValidationError

if TYPE_CHECKING:
    from .connectors import Connector
    from .memory import Memory
    from .models import AIModel
    from .prompts import Prompt, PromptArgument, PromptField, OutputSchema, OutputField
    from .templates import Template
    from .tools import LLMDefinition, ToolDefinition
    from .training import TrainingJob, TuningJob, HyperparamSpec
    from .workflows import Chain, ChainStep


# ============================================================================
# CONNECTOR VALIDATION
# ============================================================================

def validate_connector(connector: 'Connector') -> None:
    """
    Validate a Connector configuration.
    
    Ensures:
    - Name is non-empty
    - Connector type is specified
    - Config is a valid dictionary
    
    Args:
        connector: Connector instance to validate
        
    Raises:
        AIValidationError: If validation fails
        
    Example:
        connector = Connector(name="db", connector_type="postgres")
        validate_connector(connector)  # OK
        
        invalid = Connector(name="", connector_type="postgres")
        validate_connector(invalid)  # Raises AIValidationError
    """
    if not connector.name or not connector.name.strip():
        raise AIValidationError(
            "Connector name cannot be empty",
            construct_type="Connector",
            field="name",
            code="AI001",
            hint="Provide a meaningful connector name"
        )
        
    if not connector.connector_type or not connector.connector_type.strip():
        raise AIValidationError(
            "Connector type must be specified",
            construct_type="Connector",
            construct_name=connector.name,
            field="connector_type",
            code="AI002",
            hint="Specify a type like 'postgres', 'mongodb', 'redis', etc."
        )
        
    if not isinstance(connector.config, dict):
        raise AIValidationError(
            "Connector config must be a dictionary",
            construct_type="Connector",
            construct_name=connector.name,
            field="config",
            value=type(connector.config).__name__,
            code="AI003"
        )


# ============================================================================
# MEMORY VALIDATION
# ============================================================================

def validate_memory(memory: 'Memory') -> None:
    """
    Validate a Memory store configuration.
    
    Ensures:
    - Name is non-empty
    - Scope is valid (session, user, global)
    - Kind is valid (list, dict, vector)
    - max_items is positive if specified
    
    Args:
        memory: Memory instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not memory.name or not memory.name.strip():
        raise AIValidationError(
            "Memory name cannot be empty",
            construct_type="Memory",
            field="name",
            code="AI010"
        )
        
    valid_scopes = {"session", "user", "global"}
    if memory.scope not in valid_scopes:
        raise AIValidationError(
            f"Invalid memory scope: '{memory.scope}'",
            construct_type="Memory",
            construct_name=memory.name,
            field="scope",
            value=memory.scope,
            code="AI011",
            hint=f"Scope must be one of: {', '.join(sorted(valid_scopes))}"
        )
        
    valid_kinds = {"list", "dict", "vector"}
    if memory.kind not in valid_kinds:
        raise AIValidationError(
            f"Invalid memory kind: '{memory.kind}'",
            construct_type="Memory",
            construct_name=memory.name,
            field="kind",
            value=memory.kind,
            code="AI012",
            hint=f"Kind must be one of: {', '.join(sorted(valid_kinds))}"
        )
        
    if memory.max_items is not None and memory.max_items <= 0:
        raise AIValidationError(
            "max_items must be positive",
            construct_type="Memory",
            construct_name=memory.name,
            field="max_items",
            value=memory.max_items,
            code="AI013",
            hint="Specify a positive integer or omit for unlimited"
        )


# ============================================================================
# AI MODEL VALIDATION
# ============================================================================

def validate_ai_model(model: 'AIModel') -> None:
    """
    Validate an AIModel configuration.
    
    Ensures:
    - Name is non-empty
    - Provider is specified
    - Model name is specified
    - Config is a valid dictionary
    
    Args:
        model: AIModel instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not model.name or not model.name.strip():
        raise AIValidationError(
            "Model name cannot be empty",
            construct_type="AIModel",
            field="name",
            code="AI020"
        )
        
    if not model.provider or not model.provider.strip():
        raise AIValidationError(
            "Model provider must be specified",
            construct_type="AIModel",
            construct_name=model.name,
            field="provider",
            code="AI021",
            hint="Specify a provider like 'openai', 'anthropic', 'google', etc."
        )
        
    if not model.model_name or not model.model_name.strip():
        raise AIValidationError(
            "Model name must be specified",
            construct_type="AIModel",
            construct_name=model.name,
            field="model_name",
            code="AI022",
            hint="Specify a model identifier like 'gpt-4', 'claude-3-opus', etc."
        )
        
    if not isinstance(model.config, dict):
        raise AIValidationError(
            "Model config must be a dictionary",
            construct_type="AIModel",
            construct_name=model.name,
            field="config",
            value=type(model.config).__name__,
            code="AI023"
        )


# ============================================================================
# TEMPLATE VALIDATION
# ============================================================================

def validate_template(template: 'Template') -> None:
    """
    Validate a Template configuration.
    
    Ensures:
    - Name is non-empty
    - Prompt text is non-empty
    
    Args:
        template: Template instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not template.name or not template.name.strip():
        raise AIValidationError(
            "Template name cannot be empty",
            construct_type="Template",
            field="name",
            code="AI030"
        )
        
    if not template.prompt or not template.prompt.strip():
        raise AIValidationError(
            "Template prompt cannot be empty",
            construct_type="Template",
            construct_name=template.name,
            field="prompt",
            code="AI031",
            hint="Provide a non-empty prompt string"
        )


# ============================================================================
# PROMPT VALIDATION
# ============================================================================

def validate_prompt_field(field: 'PromptField', context: str = "PromptField") -> None:
    """
    Validate a PromptField schema definition.
    
    Args:
        field: PromptField instance to validate
        context: Context string for error messages
        
    Raises:
        AIValidationError: If validation fails
    """
    if not field.name or not field.name.strip():
        raise AIValidationError(
            "Field name cannot be empty",
            construct_type=context,
            field="name",
            code="AI040"
        )
        
    if not field.field_type or not field.field_type.strip():
        raise AIValidationError(
            f"Field type cannot be empty for field '{field.name}'",
            construct_type=context,
            field="field_type",
            code="AI041"
        )


def validate_prompt_argument(arg: 'PromptArgument', prompt_name: str) -> None:
    """
    Validate a PromptArgument definition.
    
    Args:
        arg: PromptArgument instance to validate
        prompt_name: Name of the prompt this argument belongs to
        
    Raises:
        AIValidationError: If validation fails
    """
    if not arg.name or not arg.name.strip():
        raise AIValidationError(
            "Argument name cannot be empty",
            construct_type="PromptArgument",
            construct_name=prompt_name,
            field="name",
            code="AI042"
        )
        
    valid_types = {"string", "int", "float", "bool", "list", "object"}
    if arg.arg_type not in valid_types:
        raise AIValidationError(
            f"Invalid argument type: '{arg.arg_type}'",
            construct_type="PromptArgument",
            construct_name=f"{prompt_name}.{arg.name}",
            field="arg_type",
            value=arg.arg_type,
            code="AI043",
            hint=f"Type must be one of: {', '.join(sorted(valid_types))}"
        )


def validate_output_field(field: 'OutputField', prompt_name: str) -> None:
    """
    Validate an OutputField in an output schema.
    
    Args:
        field: OutputField instance to validate
        prompt_name: Name of the prompt this field belongs to
        
    Raises:
        AIValidationError: If validation fails
    """
    if not field.name or not field.name.strip():
        raise AIValidationError(
            "Output field name cannot be empty",
            construct_type="OutputField",
            construct_name=prompt_name,
            field="name",
            code="AI044"
        )


def validate_output_schema(schema: 'OutputSchema', prompt_name: str) -> None:
    """
    Validate an OutputSchema definition.
    
    Args:
        schema: OutputSchema instance to validate
        prompt_name: Name of the prompt this schema belongs to
        
    Raises:
        AIValidationError: If validation fails
    """
    if not schema.fields:
        raise AIValidationError(
            "Output schema must have at least one field",
            construct_type="OutputSchema",
            construct_name=prompt_name,
            field="fields",
            code="AI045",
            hint="Define at least one output field"
        )
        
    for field in schema.fields:
        validate_output_field(field, prompt_name)


def validate_prompt(prompt: 'Prompt') -> None:
    """
    Validate a Prompt configuration.
    
    Ensures:
    - Name is non-empty
    - Template is specified
    - Model is specified
    - Arguments are valid if present
    - Output schema is valid if present
    
    Args:
        prompt: Prompt instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not prompt.name or not prompt.name.strip():
        raise AIValidationError(
            "Prompt name cannot be empty",
            construct_type="Prompt",
            field="name",
            code="AI046"
        )
        
    if not prompt.template or not prompt.template.strip():
        raise AIValidationError(
            "Prompt template cannot be empty",
            construct_type="Prompt",
            construct_name=prompt.name,
            field="template",
            code="AI047",
            hint="Provide a template string or reference"
        )
        
    if not prompt.model or not prompt.model.strip():
        raise AIValidationError(
            "Prompt model cannot be empty",
            construct_type="Prompt",
            construct_name=prompt.name,
            field="model",
            code="AI048",
            hint="Specify an AI model reference"
        )
        
    # Validate arguments if present
    for arg in prompt.args:
        validate_prompt_argument(arg, prompt.name)
        
    # Validate output schema if present
    if prompt.output_schema is not None:
        validate_output_schema(prompt.output_schema, prompt.name)


# ============================================================================
# LLM & TOOL VALIDATION
# ============================================================================

def validate_llm_definition(llm: 'LLMDefinition') -> None:
    """
    Validate an LLMDefinition configuration.
    
    Ensures:
    - Name is non-empty
    - Numeric parameters are in valid ranges if specified
    
    Args:
        llm: LLMDefinition instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not llm.name or not llm.name.strip():
        raise AIValidationError(
            "LLM name cannot be empty",
            construct_type="LLMDefinition",
            field="name",
            code="AI050"
        )
        
    if llm.temperature is not None and not (0.0 <= llm.temperature <= 2.0):
        raise AIValidationError(
            f"Temperature must be between 0.0 and 2.0, got {llm.temperature}",
            construct_type="LLMDefinition",
            construct_name=llm.name,
            field="temperature",
            value=llm.temperature,
            code="AI051"
        )
        
    if llm.max_tokens is not None and llm.max_tokens <= 0:
        raise AIValidationError(
            "max_tokens must be positive",
            construct_type="LLMDefinition",
            construct_name=llm.name,
            field="max_tokens",
            value=llm.max_tokens,
            code="AI052"
        )
        
    if llm.top_p is not None and not (0.0 <= llm.top_p <= 1.0):
        raise AIValidationError(
            f"top_p must be between 0.0 and 1.0, got {llm.top_p}",
            construct_type="LLMDefinition",
            construct_name=llm.name,
            field="top_p",
            value=llm.top_p,
            code="AI053"
        )


def validate_tool_definition(tool: 'ToolDefinition') -> None:
    """
    Validate a ToolDefinition configuration.
    
    Ensures:
    - Name is non-empty
    - Description is non-empty
    - Parameters schema is a valid dict
    
    Args:
        tool: ToolDefinition instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not tool.name or not tool.name.strip():
        raise AIValidationError(
            "Tool name cannot be empty",
            construct_type="ToolDefinition",
            field="name",
            code="AI054"
        )
        
    if not tool.description or not tool.description.strip():
        raise AIValidationError(
            "Tool description cannot be empty",
            construct_type="ToolDefinition",
            construct_name=tool.name,
            field="description",
            code="AI055",
            hint="Provide a clear description for the LLM"
        )
        
    if not isinstance(tool.parameters, dict):
        raise AIValidationError(
            "Tool parameters must be a dictionary",
            construct_type="ToolDefinition",
            construct_name=tool.name,
            field="parameters",
            value=type(tool.parameters).__name__,
            code="AI056"
        )


# ============================================================================
# WORKFLOW & CHAIN VALIDATION
# ============================================================================

def validate_chain_step(step: 'ChainStep', chain_name: str) -> None:
    """
    Validate a ChainStep configuration.
    
    Ensures:
    - Kind is non-empty
    - Target is non-empty
    - Options is a valid dict
    
    Args:
        step: ChainStep instance to validate
        chain_name: Name of the chain this step belongs to
        
    Raises:
        AIValidationError: If validation fails
    """
    if not step.kind or not step.kind.strip():
        raise AIValidationError(
            "Step kind cannot be empty",
            construct_type="ChainStep",
            construct_name=chain_name,
            field="kind",
            code="AI060",
            hint="Specify a kind like 'prompt', 'tool', 'chain', etc."
        )
        
    if not step.target or not step.target.strip():
        raise AIValidationError(
            "Step target cannot be empty",
            construct_type="ChainStep",
            construct_name=chain_name,
            field="target",
            code="AI061",
            hint="Specify the name of the prompt/tool/chain to execute"
        )
        
    if not isinstance(step.options, dict):
        raise AIValidationError(
            "Step options must be a dictionary",
            construct_type="ChainStep",
            construct_name=f"{chain_name}.{step.name or step.target}",
            field="options",
            value=type(step.options).__name__,
            code="AI062"
        )


def validate_chain(chain: 'Chain') -> None:
    """
    Validate a Chain workflow configuration.
    
    Ensures:
    - Name is non-empty
    - Has at least one step
    - All steps are valid
    
    Args:
        chain: Chain instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not chain.name or not chain.name.strip():
        raise AIValidationError(
            "Chain name cannot be empty",
            construct_type="Chain",
            field="name",
            code="AI063"
        )
        
    if not chain.steps:
        raise AIValidationError(
            "Chain must have at least one step",
            construct_type="Chain",
            construct_name=chain.name,
            field="steps",
            code="AI064",
            hint="Define at least one workflow step"
        )
        
    # Validate each step
    for i, step in enumerate(chain.steps):
        try:
            if hasattr(step, 'kind'):  # It's a ChainStep
                validate_chain_step(step, chain.name)
        except AIValidationError as e:
            # Add step index context
            e.hint = f"Error in step {i}: {e.hint}" if e.hint else f"Error in step {i}"
            raise


# ============================================================================
# TRAINING & TUNING VALIDATION
# ============================================================================

def validate_training_job(job: 'TrainingJob') -> None:
    """
    Validate a TrainingJob configuration.
    
    Ensures:
    - Name is non-empty
    - Model and dataset are specified
    - Objective is valid
    - Numeric parameters are in valid ranges
    
    Args:
        job: TrainingJob instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not job.name or not job.name.strip():
        raise AIValidationError(
            "Training job name cannot be empty",
            construct_type="TrainingJob",
            field="name",
            code="AI070"
        )
        
    if not job.model or not job.model.strip():
        raise AIValidationError(
            "Training job model cannot be empty",
            construct_type="TrainingJob",
            construct_name=job.name,
            field="model",
            code="AI071"
        )
        
    if not job.dataset or not job.dataset.strip():
        raise AIValidationError(
            "Training job dataset cannot be empty",
            construct_type="TrainingJob",
            construct_name=job.name,
            field="dataset",
            code="AI072"
        )
        
    if not job.objective or not job.objective.strip():
        raise AIValidationError(
            "Training job objective cannot be empty",
            construct_type="TrainingJob",
            construct_name=job.name,
            field="objective",
            code="AI073",
            hint="Specify objective like 'classification', 'regression', etc."
        )
        
    # Validate split ratios if specified
    if job.split:
        total = sum(job.split.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise AIValidationError(
                f"Split ratios must sum to 1.0, got {total}",
                construct_type="TrainingJob",
                construct_name=job.name,
                field="split",
                value=total,
                code="AI074",
                hint="Ensure train + validation + test = 1.0"
            )


def validate_tuning_job(job: 'TuningJob') -> None:
    """
    Validate a TuningJob configuration.
    
    Ensures:
    - Name is non-empty
    - Base training job is specified
    - Has at least one hyperparameter to tune
    
    Args:
        job: TuningJob instance to validate
        
    Raises:
        AIValidationError: If validation fails
    """
    if not job.name or not job.name.strip():
        raise AIValidationError(
            "Tuning job name cannot be empty",
            construct_type="TuningJob",
            field="name",
            code="AI075"
        )
        
    if not job.base_training_job or not job.base_training_job.strip():
        raise AIValidationError(
            "Tuning job must reference a base training job",
            construct_type="TuningJob",
            construct_name=job.name,
            field="base_training_job",
            code="AI076"
        )
        
    if not job.hyperparameters:
        raise AIValidationError(
            "Tuning job must specify at least one hyperparameter",
            construct_type="TuningJob",
            construct_name=job.name,
            field="hyperparameters",
            code="AI077",
            hint="Define search spaces for hyperparameters to optimize"
        )


__all__ = [
    "validate_connector",
    "validate_memory",
    "validate_ai_model",
    "validate_template",
    "validate_prompt",
    "validate_prompt_field",
    "validate_prompt_argument",
    "validate_output_field",
    "validate_output_schema",
    "validate_llm_definition",
    "validate_tool_definition",
    "validate_chain",
    "validate_chain_step",
    "validate_training_job",
    "validate_tuning_job",
]
