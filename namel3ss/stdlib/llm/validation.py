"""LLM configuration validation for the Namel3ss standard library."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .config import LLMConfigField, get_llm_config_spec


class LLMValidationError(Exception):
    """Raised when LLM configuration validation fails."""
    pass


def validate_llm_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate an LLM configuration against standard library specifications.
    
    Args:
        config: LLM configuration parameters to validate
        
    Returns:
        Dictionary of validation errors (empty if valid)
    """
    errors: Dict[str, str] = {}
    
    # Check required fields
    required_fields = []
    for field, spec in get_standard_llm_fields().items():
        if spec.required:
            required_fields.append(field.value)
    
    for field_name in required_fields:
        if field_name not in config or config[field_name] is None:
            errors[field_name] = f"Required field '{field_name}' is missing"
    
    # Validate each provided field
    for field_name, value in config.items():
        if field_name.startswith('_'):  # Skip private fields
            continue
            
        try:
            field_enum = LLMConfigField(field_name)
            spec = get_llm_config_spec(field_enum)
        except ValueError:
            # Unknown field - warn but don't error (allow provider-specific fields)
            continue
        
        if not spec.validate_value(value):
            errors[field_name] = _get_validation_error_message(spec, value)
    
    # Cross-field validation
    _validate_field_combinations(config, errors)
    
    return errors


def _get_validation_error_message(spec, value: Any) -> str:
    """Generate a descriptive validation error message."""
    field_name = spec.field.value
    
    if value is None and spec.required:
        return f"'{field_name}' is required"
    
    if spec.field_type == 'float':
        if not isinstance(value, (int, float)):
            return f"'{field_name}' must be a number"
        if spec.min_value is not None and value < spec.min_value:
            return f"'{field_name}' must be >= {spec.min_value}"
        if spec.max_value is not None and value > spec.max_value:
            return f"'{field_name}' must be <= {spec.max_value}"
    
    elif spec.field_type == 'int':
        if not isinstance(value, int):
            return f"'{field_name}' must be an integer"
        if spec.min_value is not None and value < spec.min_value:
            return f"'{field_name}' must be >= {spec.min_value}"
        if spec.max_value is not None and value > spec.max_value:
            return f"'{field_name}' must be <= {spec.max_value}"
    
    elif spec.field_type == 'str':
        if not isinstance(value, str):
            return f"'{field_name}' must be a string"
        if spec.valid_values and value not in spec.valid_values:
            return f"'{field_name}' must be one of: {', '.join(spec.valid_values)}"
    
    elif spec.field_type == 'bool':
        if not isinstance(value, bool):
            return f"'{field_name}' must be a boolean (true/false)"
    
    elif spec.field_type == 'list':
        if not isinstance(value, list):
            return f"'{field_name}' must be a list"
    
    return f"'{field_name}' has invalid value: {value}"


def _validate_field_combinations(config: Dict[str, Any], errors: Dict[str, str]) -> None:
    """Validate cross-field constraints and combinations."""
    
    # Ensure temperature and top_p aren't both very restrictive
    temp = config.get('temperature')
    top_p = config.get('top_p') 
    
    if (isinstance(temp, (int, float)) and temp < 0.1 and 
        isinstance(top_p, (int, float)) and top_p < 0.1):
        errors['temperature'] = "Very low temperature and top_p may cause repetitive outputs"
    
    # Validate stop sequences format
    stop_sequences = config.get('stop_sequences')
    if isinstance(stop_sequences, list):
        for i, seq in enumerate(stop_sequences):
            if not isinstance(seq, str):
                errors['stop_sequences'] = f"Stop sequence at index {i} must be a string"
                break
            if len(seq) == 0:
                errors['stop_sequences'] = f"Empty stop sequence at index {i}"
                break


def validate_llm_config_strict(config: Dict[str, Any]) -> None:
    """
    Validate LLM configuration and raise exception if invalid.
    
    Args:
        config: LLM configuration parameters to validate
        
    Raises:
        LLMValidationError: If configuration is invalid
    """
    errors = validate_llm_config(config)
    if errors:
        error_msgs = [f"{field}: {msg}" for field, msg in errors.items()]
        raise LLMValidationError(f"LLM configuration validation failed: {'; '.join(error_msgs)}")


def suggest_llm_config(
    provider: str,
    model: str,
    use_case: str = "general",
    **overrides: Any
) -> Dict[str, Any]:
    """
    Generate a suggested LLM configuration based on use case.
    
    Args:
        provider: LLM provider name
        model: Model identifier
        use_case: Use case hint ('general', 'creative', 'precise', 'fast')
        **overrides: Configuration overrides
        
    Returns:
        Suggested configuration dictionary
    """
    from .config import get_standard_llm_config
    
    config = get_standard_llm_config()
    config.update({
        'provider': provider,
        'model': model
    })
    
    # Use case specific adjustments
    if use_case == 'creative':
        config.update({
            'temperature': 0.8,
            'top_p': 0.95,
            'frequency_penalty': 0.1
        })
    elif use_case == 'precise':
        config.update({
            'temperature': 0.2,
            'top_p': 0.8,
            'frequency_penalty': 0.0
        })
    elif use_case == 'fast':
        config.update({
            'temperature': 0.7,
            'max_tokens': 512,
            'stream': True
        })
    
    config.update(overrides)
    return config


def get_standard_llm_fields():
    """Get standard LLM fields for validation."""
    from .config import STANDARD_LLM_FIELDS
    return STANDARD_LLM_FIELDS