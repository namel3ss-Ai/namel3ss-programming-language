"""
AI-centric AST nodes for connectors, templates, prompts, and chains.

DEPRECATED: This module has been refactored into modular files.
Please import from namel3ss.ast instead:
    from namel3ss.ast import Connector, AIModel, Memory, Template, Prompt, Chain, etc.

This file remains as a compatibility shim for any internal code that
imports directly from namel3ss.ast.ai.
"""

# Re-export all symbols from the new modular structure
from .ai_connectors import Connector
from .ai_models import AIModel
from .ai_templates import Template
from .ai_memory import Memory
from .ai_prompts import (
    EnumType,
    OutputField,
    OutputFieldType,
    OutputSchema,
    Prompt,
    PromptArgument,
    PromptField,
)
from .ai_workflows import (
    Chain,
    ChainStep,
    StepEvaluationConfig,
    WorkflowForBlock,
    WorkflowIfBlock,
    WorkflowNode,
    WorkflowWhileBlock,
)
from .ai_training import (
    EarlyStoppingSpec,
    HyperparamSpec,
    HyperparameterValue,
    TrainingComputeSpec,
    TrainingJob,
    TuningJob,
)
from .ai_tools import (
    LLMDefinition,
    ToolDefinition,
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
