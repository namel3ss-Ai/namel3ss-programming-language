"""Centralized configuration validation for AST declarations.

This module provides production-grade validation logic for configuration
dictionaries used in AST construction. It handles:

- Cross-field validation (mutually exclusive fields)
- Type-specific field restrictions
- Clear, actionable error messages
- Consistent error handling across all declaration types

Design Philosophy:
- Single source of truth for validation rules
- Fail fast with clear messages
- Deterministic error behavior
- No silent data loss or unexpected coercion
"""

from typing import Any, Dict, Set, Optional, Type
from dataclasses import is_dataclass
from .errors import create_syntax_error


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


# Field restrictions per declaration type
# Maps AST class name -> set of fields that are NOT allowed
DISALLOWED_FIELDS: Dict[str, Set[str]] = {
    "RagPipelineDefinition": {
        "temperature", "top_p", "top_k", "max_tokens",
        "frequency_penalty", "presence_penalty"
    },
    "IndexDefinition": {
        "temperature", "top_p", "model", "llm"
    },
    "Dataset": {
        "temperature", "model", "llm"
    },
}

# Mutually exclusive field groups
# Each tuple represents fields that cannot coexist
MUTUALLY_EXCLUSIVE_GROUPS = [
    # Legacy vs modern prompt schemas
    ("input_fields", "args"),
    ("output_fields", "output_schema"),
    # Model specification conflicts
    ("model", "llm"),  # Handled specially in Prompt - both allowed with precedence
]

# Prompt-specific validation rules
PROMPT_LEGACY_MODERN_CONFLICTS = [
    ("input", "args", "Use 'args' for modern typed arguments, not legacy 'input'"),
    ("output", "output_schema", "Use 'output_schema' for structured outputs, not legacy 'output'"),
]


def validate_field_restrictions(
    dataclass_type: Type,
    config: Dict[str, Any],
    path: str = "",
    line: int = 0,
    column: int = 0
) -> None:
    """
    Validate that no disallowed fields are present for this declaration type.
    
    Args:
        dataclass_type: The target AST dataclass
        config: Configuration dictionary to validate
        path: Source file path (for error reporting)
        line: Line number (for error reporting)
        column: Column number (for error reporting)
        
    Raises:
        N3SyntaxError: If disallowed fields are found
    """
    class_name = dataclass_type.__name__
    disallowed = DISALLOWED_FIELDS.get(class_name, set())
    
    if not disallowed:
        return
    
    # Check for disallowed fields
    found_disallowed = set(config.keys()) & disallowed
    
    if found_disallowed:
        fields_str = ", ".join(f"'{f}'" for f in sorted(found_disallowed))
        raise create_syntax_error(
            f"Fields {fields_str} are not valid for {class_name}. "
            f"These are model-level parameters and should not be used here. "
            f"Consider moving them to an LLM definition or removing them.",
            path=path,
            line=line,
            column=column,
        )


def validate_prompt_legacy_vs_modern(
    config: Dict[str, Any],
    path: str = "",
    line: int = 0,
    column: int = 0
) -> None:
    """
    Validate prompt-specific legacy vs modern field conflicts.
    
    Enforces that prompts use either legacy (input/output) or modern
    (args/output_schema) fields, not both. This prevents ambiguous
    configurations.
    
    Args:
        config: Configuration dictionary for a prompt
        path: Source file path
        line: Line number
        column: Column number
        
    Raises:
        N3SyntaxError: If conflicting fields are found
    """
    for legacy_key, modern_key, message in PROMPT_LEGACY_MODERN_CONFLICTS:
        if legacy_key in config and modern_key in config:
            raise create_syntax_error(
                f"Cannot use both '{legacy_key}' and '{modern_key}' in the same prompt. "
                f"{message}",
                path=path,
                line=line,
                column=column,
            )


def validate_prompt_model_aliases(
    config: Dict[str, Any],
    path: str = "",
    line: int = 0,
    column: int = 0
) -> Dict[str, Any]:
    """
    Handle model/llm aliasing for prompts with clear precedence.
    
    Strategy:
    - If only 'llm' is present (not 'model'), treat it as an alias for 'model'
    - If both present, keep both as separate fields (llm goes to parameters)
    - If only 'model' is present, use it as-is
    - No validation errors - accept all configurations
    
    This provides backwards compatibility while allowing "llm" as a parameter
    field name when needed.
    
    Args:
        config: Configuration dictionary
        path: Source file path
        line: Line number
        column: Column number
        
    Returns:
        Config (unchanged - aliasing happens in filter_config_for_dataclass)
    """
    # No modifications here - let the filtering layer handle aliasing
    # This validation function is a no-op but kept for future enhancements
    return config


def validate_prompt_name_override(
    config: Dict[str, Any],
    declared_name: str,
    path: str = "",
    line: int = 0,
    column: int = 0
) -> Dict[str, Any]:
    """
    Prevent 'name' field inside prompt block from overriding declared name.
    
    The prompt's canonical name comes from: prompt "name" { ... }
    Any 'name' inside the block is treated as metadata, not the prompt name.
    
    Args:
        config: Configuration dictionary
        declared_name: Name from prompt "name" declaration
        path: Source file path
        line: Line number
        column: Column number
        
    Returns:
        Config with 'name' moved to metadata if present
    """
    if "name" in config:
        # Move to metadata instead of silently ignoring
        if "metadata" not in config:
            config["metadata"] = {}
        
        if not isinstance(config["metadata"], dict):
            config["metadata"] = {}
        
        config["metadata"]["internal_name"] = config.pop("name")
    
    return config


def validate_config_for_declaration(
    dataclass_type: Type,
    config: Dict[str, Any],
    declared_name: Optional[str] = None,
    path: str = "",
    line: int = 0,
    column: int = 0
) -> Dict[str, Any]:
    """
    Main entry point for config validation before AST construction.
    
    This function orchestrates all validation checks in a single pass:
    1. Field restrictions (e.g. no temperature in RAG)
    2. Type-specific rules (e.g. prompt legacy vs modern)
    3. Alias resolution (e.g. model/llm in prompts)
    4. Name override prevention
    
    Args:
        dataclass_type: Target AST dataclass type
        config: Raw configuration dictionary
        declared_name: Name from declaration (for prompts, agents, etc.)
        path: Source file path
        line: Line number
        column: Column number
        
    Returns:
        Validated and potentially modified config dictionary
        
    Raises:
        N3SyntaxError: If validation fails
    """
    # Make a copy to avoid mutating input
    config = dict(config)
    
    # Check field restrictions
    validate_field_restrictions(dataclass_type, config, path, line, column)
    
    # Prompt-specific validation
    class_name = dataclass_type.__name__
    if class_name == "Prompt":
        # Legacy vs modern conflict detection
        validate_prompt_legacy_vs_modern(config, path, line, column)
        
        # Model/llm alias resolution
        config = validate_prompt_model_aliases(config, path, line, column)
        
        # Name override prevention
        if declared_name:
            config = validate_prompt_name_override(
                config, declared_name, path, line, column
            )
    
    return config


__all__ = [
    "ConfigValidationError",
    "validate_field_restrictions",
    "validate_prompt_legacy_vs_modern",
    "validate_prompt_model_aliases",
    "validate_prompt_name_override",
    "validate_config_for_declaration",
]
