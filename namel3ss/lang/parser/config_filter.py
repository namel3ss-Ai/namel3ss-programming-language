"""Configuration filtering and aliasing for AST constructor arguments.

This module provides production-grade utilities for safely transforming user
configuration dictionaries from DSL blocks into valid AST dataclass constructor
arguments, with support for:

- Dataclass introspection (no hard-coded field lists)
- DSL → AST field aliasing (e.g., "llm" -> "llm_name")
- Unknown field routing to config/metadata sinks
- Preservation of dataclass defaults
- Forward compatibility with future fields

Design Principles:
1. Never directly unpack user config dicts into constructors
2. Use dataclass introspection for valid fields
3. Support semantic aliasing between DSL and AST names
4. Route unknown fields safely (never lose user data)
5. Respect dataclass defaults (don't override unnecessarily)
"""

from dataclasses import dataclass, fields as dataclass_fields, MISSING
from typing import Any, Dict, Set, Tuple, Type, Optional
import dataclasses

from .config_validator import validate_config_for_declaration


# Centralized alias mappings for all AST types
# Maps DSL field name -> AST constructor parameter name
AGENT_ALIASES = {
    "llm": "llm_name",
    "tools": "tool_names",
    "memory": "memory_config",
    "system": "system_prompt",
    "policy": "policy_name",
    "input": "input_key",
}

LLM_ALIASES = {
    "system": "system_prompt",
    "max_length": "max_tokens",
    "top_k_sampling": "top_k",
}

PROMPT_ALIASES = {
    "llm": "model",  # Allow 'llm' as alias for 'model' in prompts
    "system": "system_prompt",
    "max_length": "max_tokens",
}

CHAIN_ALIASES = {
    "prompts": "prompt_names",
    "llm": "llm_name",
}

RAG_ALIASES = {
}

DATASET_ALIASES = {
    "path": "source_path",
    "format": "file_format",
}

GRAPH_ALIASES = {
    "start": "start_agent",
    "termination": "termination_agents",
    "max_iterations": "max_hops",
}

TOOL_ALIASES = {
    "parameters": "params",
    "params": "parameters",  # Bidirectional for flexibility
}

# Master registry - add more as declarations are implemented
ALIAS_REGISTRY: Dict[str, Dict[str, str]] = {
    "AgentDefinition": AGENT_ALIASES,
    "LLMDefinition": LLM_ALIASES,
    "Prompt": PROMPT_ALIASES,
    "Chain": CHAIN_ALIASES,
    "RAGPipeline": RAG_ALIASES,
    "Dataset": DATASET_ALIASES,
    "GraphDefinition": GRAPH_ALIASES,
    "ToolDefinition": TOOL_ALIASES,
}


def _get_dataclass_fields(dataclass_type: Type) -> Set[str]:
    """
    Extract all field names from a dataclass using introspection.
    
    Args:
        dataclass_type: The dataclass type to introspect
        
    Returns:
        Set of field names that the dataclass constructor accepts
        
    Raises:
        TypeError: If dataclass_type is not a dataclass
    """
    if not dataclasses.is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type.__name__} is not a dataclass")
    
    return {f.name for f in dataclass_fields(dataclass_type)}


def _has_config_sink(dataclass_type: Type) -> Tuple[bool, Optional[str]]:
    """
    Check if dataclass has a field for storing unknown configuration.
    
    Args:
        dataclass_type: The dataclass type to check
        
    Returns:
        Tuple of (has_sink, sink_field_name)
        - (True, "parameters") if class is Prompt (special case)
        - (True, "config") if class has a 'config' field
        - (True, "metadata") if class has a 'metadata' field (and no config)
        - (False, None) if no sink field exists
    
    Note:
        Prompt is a special case because it has both 'parameters' (for model
        settings) and 'metadata' (for versioning/tags). Unknown fields should
        go to 'parameters' by default for backwards compatibility.
    """
    field_names = _get_dataclass_fields(dataclass_type)
    class_name = dataclass_type.__name__
    
    # Special case: Prompt uses 'parameters' as the primary sink
    if class_name == "Prompt" and "parameters" in field_names:
        return True, "parameters"
    
    if "config" in field_names:
        return True, "config"
    elif "metadata" in field_names:
        return True, "metadata"
    else:
        return False, None


def filter_config_for_dataclass(
    config: Dict[str, Any],
    dataclass_type: Type,
    aliases: Optional[Dict[str, str]] = None,
    required_fields: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Filter and transform user config dict for safe AST dataclass construction.
    
    This function implements the core filtering logic:
    1. Introspects the dataclass to get valid constructor fields
    2. Applies DSL → AST field aliasing
    3. Separates known fields from unknown fields
    4. Returns filtered kwargs and leftover config
    
    Args:
        config: Raw configuration dictionary from DSL block parsing
        dataclass_type: Target AST dataclass type (e.g., AgentDefinition)
        aliases: Optional DSL→AST field name mapping (default: use ALIAS_REGISTRY)
        required_fields: Optional set of fields that must be present in config
                        (for validation purposes)
    
    Returns:
        Tuple of (constructor_kwargs, leftover_config):
        - constructor_kwargs: Dict of validated args ready for dataclass(**kwargs)
        - leftover_config: Dict of unknown fields (to be stored in config/metadata)
    
    Example:
        >>> config = {
        ...     "llm": "claude",
        ...     "tools": ["search"],
        ...     "temperature": 0.7,
        ...     "custom_field": 42
        ... }
        >>> kwargs, leftover = filter_config_for_dataclass(
        ...     config, AgentDefinition, AGENT_ALIASES
        ... )
        >>> kwargs
        {"llm_name": "claude", "tool_names": ["search"], "temperature": 0.7}
        >>> leftover
        {"custom_field": 42}
    
    Design Notes:
        - Never modifies the input config dict
        - Respects dataclass defaults (only passes explicitly provided values)
        - Unknown fields are preserved, not silently dropped
        - Thread-safe (no shared mutable state)
    """
    # Get valid fields for this dataclass
    valid_fields = _get_dataclass_fields(dataclass_type)
    
    # Get aliases for this dataclass type
    if aliases is None:
        class_name = dataclass_type.__name__
        aliases = ALIAS_REGISTRY.get(class_name, {})
    
    # Special handling for Prompt: disable llm->model aliasing if model exists
    class_name = dataclass_type.__name__
    if class_name == "Prompt" and "model" in config and "llm" in aliases:
        # Create a copy of aliases without the llm->model mapping
        aliases = {k: v for k, v in aliases.items() if k != "llm"}
    
    # Prepare output dictionaries
    constructor_kwargs = {}
    leftover_config = {}
    
    # Process each config entry
    for key, value in config.items():
        # Check if this key has an alias
        target_field = aliases.get(key, key)
        
        # Determine if this field is valid for the constructor
        if target_field in valid_fields:
            constructor_kwargs[target_field] = value
        else:
            # Unknown field - preserve in leftover
            leftover_config[key] = value
    
    return constructor_kwargs, leftover_config


def build_dataclass_with_config(
    dataclass_type: Type,
    config: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
    required_fields: Optional[Set[str]] = None,
    declared_name: Optional[str] = None,
    path: str = "",
    line: int = 0,
    column: int = 0,
    **extra_kwargs
) -> Any:
    """
    Construct a dataclass instance with safe config filtering and leftover handling.
    
    This is the main entry point for parser declaration methods. It handles:
    1. Centralized validation via validate_config_for_declaration()
    2. Config filtering and aliasing via filter_config_for_dataclass()
    3. Leftover config routing to config/metadata fields
    4. Merging with any explicitly provided kwargs
    5. Construction with proper error handling
    
    Args:
        dataclass_type: The AST dataclass to instantiate
        config: Raw config dict from DSL block
        aliases: Optional field aliases (default: use ALIAS_REGISTRY)
        required_fields: Optional set of required field names
        declared_name: Name from declaration (for prompts, agents, etc.)
        path: Source file path (for error reporting)
        line: Line number (for error reporting)
        column: Column number (for error reporting)
        **extra_kwargs: Additional kwargs to merge (e.g., name, location)
    
    Returns:
        Constructed dataclass instance
        
    Raises:
        TypeError: If unknown fields exist and no config/metadata sink available
        ValueError: If required fields are missing
        N3SyntaxError: If validation fails
    
    Example:
        >>> agent = build_dataclass_with_config(
        ...     AgentDefinition,
        ...     config={"llm": "claude", "tools": ["search"], "custom": 42},
        ...     name="my_agent"
        ... )
        >>> agent.llm_name
        "claude"
        >>> agent.config
        {"custom": 42}
    """
    # Step 1: Run centralized validation
    validated_config = validate_config_for_declaration(
        dataclass_type,
        config,
        declared_name=declared_name,
        path=path,
        line=line,
        column=column
    )
    
    # Step 2: Filter config into constructor kwargs and leftovers
    constructor_kwargs, leftover = filter_config_for_dataclass(
        validated_config, dataclass_type, aliases, required_fields
    )
    
    # Check if there's a sink for leftover config
    has_sink, sink_field = _has_config_sink(dataclass_type)
    
    # Handle leftover configuration
    if leftover:
        if has_sink:
            # Merge leftover into existing config/metadata if present
            existing_sink = constructor_kwargs.get(sink_field, {})
            if not isinstance(existing_sink, dict):
                existing_sink = {}
            existing_sink.update(leftover)
            constructor_kwargs[sink_field] = existing_sink
        else:
            # No sink - raise descriptive error
            class_name = dataclass_type.__name__
            unknown_keys = ", ".join(leftover.keys())
            raise TypeError(
                f"{class_name} does not support unknown fields: {unknown_keys}. "
                f"Valid fields are: {', '.join(sorted(_get_dataclass_fields(dataclass_type)))}"
            )
    
    # Merge with extra kwargs (explicit parameters override config)
    final_kwargs = {**constructor_kwargs, **extra_kwargs}
    
    # Validate required fields if specified
    if required_fields:
        missing = required_fields - set(final_kwargs.keys())
        if missing:
            raise ValueError(
                f"Missing required fields for {dataclass_type.__name__}: "
                f"{', '.join(sorted(missing))}"
            )
    
    # Construct the dataclass instance
    try:
        return dataclass_type(**final_kwargs)
    except TypeError as e:
        # Provide better error message
        raise TypeError(
            f"Error constructing {dataclass_type.__name__}: {e}"
        ) from e


__all__ = [
    "filter_config_for_dataclass",
    "build_dataclass_with_config",
    "AGENT_ALIASES",
    "LLM_ALIASES",
    "PROMPT_ALIASES",
    "CHAIN_ALIASES",
    "RAG_ALIASES",
    "DATASET_ALIASES",
    "GRAPH_ALIASES",
    "TOOL_ALIASES",
    "ALIAS_REGISTRY",
]
