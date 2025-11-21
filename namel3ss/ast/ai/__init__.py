"""
AI subsystem for namel3ss programming language.

This subpackage contains all AI-related AST nodes, organized into coherent modules:

Core Components:
- connectors: External service integration configurations
- memory: Conversational state and context management
- models: AI model references and configurations  
- templates: Reusable prompt templates
- prompts: Typed prompt definitions with structured I/O
- tools: LLM definitions and tool/function specifications
- workflows: Multi-step chains and workflow orchestration
- training: ML training and hyperparameter tuning jobs

Supporting Infrastructure:
- errors: Domain-specific exception types
- validation: Centralized validation functions

Usage:
    from namel3ss.ast.ai import Prompt, Chain, AIModel, Connector
    
    # Or import directly from modules
    from namel3ss.ast.ai.prompts import Prompt, OutputSchema
    from namel3ss.ast.ai.workflows import Chain, ChainStep
    from namel3ss.ast.ai.validation import validate_prompt, validate_chain

Design Principles:
- Production-grade: No toy examples, demo data, or shortcuts
- Strongly typed: Comprehensive type hints throughout
- Well-documented: Detailed docstrings for all public APIs
- Validated: Centralized validation with clear error messages
- Maintainable: Single responsibility per module
- Backwards compatible: Existing imports continue to work

For validation, always use the centralized validators from .validation:
    from namel3ss.ast.ai.validation import (
        validate_prompt,
        validate_chain,
        validate_ai_model,
        validate_connector,
        ...
    )

For error handling, use domain-specific exceptions from .errors:
    from namel3ss.ast.ai.errors import (
        AIValidationError,
        AIConfigurationError,
        AIExecutionError,
    )
"""

from __future__ import annotations

# Core AI constructs
from .connectors import Connector
from .memory import Memory
from .models import AIModel
from .templates import Template

# Prompt system
from .prompts import (
    EnumType,
    OutputField,
    OutputFieldType,
    OutputSchema,
    Prompt,
    PromptArgument,
    PromptField,
)

# Tools and LLMs
from .tools import (
    LLMDefinition,
    ToolDefinition,
)

# Workflows and chains
from .workflows import (
    Chain,
    ChainStep,
    StepEvaluationConfig,
    WorkflowForBlock,
    WorkflowIfBlock,
    WorkflowNode,
    WorkflowWhileBlock,
)

# Training and tuning
from .training import (
    EarlyStoppingSpec,
    HyperparamSpec,
    HyperparameterValue,
    TrainingComputeSpec,
    TrainingJob,
    TuningJob,
)

# RLHF training
from .rlhf import (
    RLHFJob,
    RLHFPEFTConfig,
    RLHFAlgorithmConfig,
    RLHFComputeSpec,
    RLHFLoggingConfig,
    RLHFSafetyConfig,
    ConfigValue,
)

# Error types
from .errors import (
    AIConfigurationError,
    AIExecutionError,
    AIValidationError,
)

# Validation functions
from .validation import (
    validate_ai_model,
    validate_chain,
    validate_chain_step,
    validate_connector,
    validate_llm_definition,
    validate_memory,
    validate_output_field,
    validate_output_schema,
    validate_prompt,
    validate_prompt_argument,
    validate_prompt_field,
    validate_template,
    validate_tool_definition,
    validate_training_job,
    validate_tuning_job,
)

__all__ = [
    # Core constructs
    "Connector",
    "AIModel",
    "Template",
    "Memory",
    # Prompts and schemas
    "PromptField",
    "PromptArgument",
    "EnumType",
    "OutputFieldType",
    "OutputField",
    "OutputSchema",
    "Prompt",
    # Workflows and chains
    "StepEvaluationConfig",
    "ChainStep",
    "WorkflowIfBlock",
    "WorkflowForBlock",
    "WorkflowWhileBlock",
    "Chain",
    "WorkflowNode",
    # Training and tuning
    "TrainingComputeSpec",
    "TrainingJob",
    "HyperparamSpec",
    "EarlyStoppingSpec",
    "TuningJob",
    "HyperparameterValue",
    # RLHF training
    "RLHFJob",
    "RLHFPEFTConfig",
    "RLHFAlgorithmConfig",
    "RLHFComputeSpec",
    "RLHFLoggingConfig",
    "RLHFSafetyConfig",
    "ConfigValue",
    # Tools and LLMs
    "LLMDefinition",
    "ToolDefinition",
    # Errors
    "AIValidationError",
    "AIConfigurationError",
    "AIExecutionError",
    # Validation
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
