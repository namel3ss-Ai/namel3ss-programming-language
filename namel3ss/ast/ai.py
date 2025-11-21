"""
AI-centric AST nodes for connectors, templates, prompts, and chains.

REFACTORED: This module has been reorganized into a subpackage at namel3ss.ast.ai/
This file remains as a thin compatibility shim for any code that imports from namel3ss.ast.ai

Preferred imports (new):
    from namel3ss.ast.ai import Connector, AIModel, Prompt, Chain
    from namel3ss.ast import Connector, AIModel, Prompt, Chain  # Also works

Legacy imports (still supported):
    from namel3ss.ast.ai import Connector, AIModel, Prompt, Chain

The new subpackage structure at namel3ss.ast.ai/ provides:
- Better organization (one responsibility per module)
- Centralized validation (ai.validation module)
- Production-grade error handling (ai.errors module)
- Comprehensive documentation
- Easier maintenance and testing

For validation, use:
    from namel3ss.ast.ai.validation import validate_prompt, validate_chain
    
For error handling, use:
    from namel3ss.ast.ai.errors import AIValidationError, AIConfigurationError
"""

# Re-export all symbols from the new subpackage structure
from .ai import (
    # Core constructs
    Connector,
    AIModel,
    Template,
    Memory,
    # Prompts and schemas  
    PromptField,
    PromptArgument,
    EnumType,
    OutputFieldType,
    OutputField,
    OutputSchema,
    Prompt,
    # Workflows and chains
    StepEvaluationConfig,
    ChainStep,
    WorkflowIfBlock,
    WorkflowForBlock,
    WorkflowWhileBlock,
    Chain,
    WorkflowNode,
    # Training and tuning
    TrainingComputeSpec,
    TrainingJob,
    HyperparamSpec,
    EarlyStoppingSpec,
    TuningJob,
    HyperparameterValue,
    # Tools and LLMs
    LLMDefinition,
    ToolDefinition,
    # Errors and validation
    AIValidationError,
    AIConfigurationError,
    AIExecutionError,
)

__all__ = [
    # Connectors and models
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
    # Tools and LLMs
    "LLMDefinition",
    "ToolDefinition",
]
