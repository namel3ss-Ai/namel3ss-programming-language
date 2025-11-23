"""Memory configuration validation for the Namel3ss standard library."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .policies import MemoryPolicy, get_memory_policy_spec


class MemoryValidationError(Exception):
    """Raised when memory configuration validation fails."""
    pass


def validate_memory_config(
    policy: Union[str, MemoryPolicy],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Validate a memory configuration against standard library specifications.
    
    Args:
        policy: Memory policy to validate
        config: Configuration parameters to validate
        
    Returns:
        Dictionary of validation errors (empty if valid)
        
    Raises:
        MemoryValidationError: If policy is not recognized
    """
    errors: Dict[str, str] = {}
    config = config or {}
    
    try:
        spec = get_memory_policy_spec(policy)
    except ValueError as e:
        raise MemoryValidationError(str(e)) from e
    
    # Validate max_items parameter
    if "max_items" in config:
        if not spec.supports_max_items:
            errors["max_items"] = f"Policy '{spec.policy.value}' does not support max_items parameter"
        else:
            max_items = config["max_items"]
            if not isinstance(max_items, int) or max_items <= 0:
                errors["max_items"] = "max_items must be a positive integer"
    
    # Validate window_size parameter  
    if "window_size" in config:
        if not spec.supports_window_size:
            errors["window_size"] = f"Policy '{spec.policy.value}' does not support window_size parameter"
        else:
            window_size = config["window_size"]
            if not isinstance(window_size, int) or window_size <= 0:
                errors["window_size"] = "window_size must be a positive integer"
    
    # Validate summarization parameters
    summarization_fields = [
        "summarizer", "max_summary_tokens", "summary_trigger_messages",
        "summary_trigger_tokens", "summary_recent_window"
    ]
    
    for field in summarization_fields:
        if field in config and not spec.supports_summarization:
            errors[field] = f"Policy '{spec.policy.value}' does not support summarization parameter '{field}'"
    
    # Policy-specific validation
    if spec.policy == MemoryPolicy.SUMMARY:
        _validate_summary_config(config, errors)
    elif spec.policy == MemoryPolicy.CONVERSATION_WINDOW:
        _validate_window_config(config, errors)
    
    return errors


def _validate_summary_config(config: Dict[str, Any], errors: Dict[str, str]) -> None:
    """Validate summarization-specific configuration."""
    
    if "max_summary_tokens" in config:
        max_tokens = config["max_summary_tokens"]
        if not isinstance(max_tokens, int) or max_tokens <= 0 or max_tokens > 8192:
            errors["max_summary_tokens"] = "max_summary_tokens must be between 1 and 8192"
    
    if "summary_trigger_messages" in config:
        trigger_msgs = config["summary_trigger_messages"]
        if not isinstance(trigger_msgs, int) or trigger_msgs <= 0:
            errors["summary_trigger_messages"] = "summary_trigger_messages must be a positive integer"
    
    if "summary_trigger_tokens" in config:
        trigger_tokens = config["summary_trigger_tokens"]
        if not isinstance(trigger_tokens, int) or trigger_tokens <= 0:
            errors["summary_trigger_tokens"] = "summary_trigger_tokens must be a positive integer"
    
    if "summary_recent_window" in config:
        recent_window = config["summary_recent_window"]
        if not isinstance(recent_window, int) or recent_window < 0:
            errors["summary_recent_window"] = "summary_recent_window must be a non-negative integer"
    
    if "summarizer" in config:
        summarizer = config["summarizer"]
        if not isinstance(summarizer, str) or not summarizer.strip():
            errors["summarizer"] = "summarizer must be a non-empty string"
        elif "/" not in summarizer:
            errors["summarizer"] = "summarizer must be in format 'provider/model' (e.g., 'openai/gpt-4o-mini')"


def _validate_window_config(config: Dict[str, Any], errors: Dict[str, str]) -> None:
    """Validate conversation window-specific configuration."""
    
    # Ensure window_size is reasonable if max_items is also set
    if "window_size" in config and "max_items" in config:
        window_size = config.get("window_size", 0)
        max_items = config.get("max_items", 0)
        
        if isinstance(window_size, int) and isinstance(max_items, int):
            if window_size > max_items:
                errors["window_size"] = "window_size cannot be larger than max_items"


def validate_memory_config_strict(
    policy: Union[str, MemoryPolicy],
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Validate memory configuration and raise exception if invalid.
    
    Args:
        policy: Memory policy to validate
        config: Configuration parameters to validate
        
    Raises:
        MemoryValidationError: If configuration is invalid
    """
    errors = validate_memory_config(policy, config)
    if errors:
        error_msgs = [f"{field}: {msg}" for field, msg in errors.items()]
        raise MemoryValidationError(f"Memory configuration validation failed: {'; '.join(error_msgs)}")


def suggest_memory_config(
    policy: Union[str, MemoryPolicy],
    **overrides: Any
) -> Dict[str, Any]:
    """
    Generate a suggested configuration for a memory policy.
    
    Args:
        policy: Memory policy
        **overrides: Configuration overrides
        
    Returns:
        Suggested configuration dictionary
    """
    spec = get_memory_policy_spec(policy)
    config = spec.default_config.copy()
    config.update(overrides)
    return config