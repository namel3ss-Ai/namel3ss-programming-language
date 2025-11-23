"""Tool configuration validation for the Namel3ss standard library."""

from __future__ import annotations

from typing import Any, Dict, Union

from .base import ToolCategory, get_tool_spec


class ToolValidationError(Exception):
    """Raised when tool configuration validation fails."""
    pass


def validate_tool_config(
    category: Union[str, ToolCategory],
    config: Dict[str, Any]
) -> Dict[str, str]:
    """
    Validate a tool configuration against standard library specifications.
    
    Args:
        category: Tool category to validate against
        config: Tool configuration parameters to validate
        
    Returns:
        Dictionary of validation errors (empty if valid)
        
    Raises:
        ToolValidationError: If category is not recognized
    """
    try:
        spec = get_tool_spec(category)
    except ValueError as e:
        raise ToolValidationError(str(e)) from e
    
    return spec.validate_config(config)


def validate_tool_config_strict(
    category: Union[str, ToolCategory],
    config: Dict[str, Any]
) -> None:
    """
    Validate tool configuration and raise exception if invalid.
    
    Args:
        category: Tool category to validate against
        config: Tool configuration parameters to validate
        
    Raises:
        ToolValidationError: If configuration is invalid
    """
    errors = validate_tool_config(category, config)
    if errors:
        error_msgs = [f"{field}: {msg}" for field, msg in errors.items()]
        category_name = category.value if hasattr(category, 'value') else str(category)
        raise ToolValidationError(
            f"Tool configuration validation failed for category '{category_name}': "
            f"{'; '.join(error_msgs)}"
        )


def suggest_tool_config(
    category: Union[str, ToolCategory],
    **overrides: Any
) -> Dict[str, Any]:
    """
    Generate a suggested configuration for a tool category.
    
    Args:
        category: Tool category
        **overrides: Configuration overrides
        
    Returns:
        Suggested configuration dictionary
    """
    spec = get_tool_spec(category)
    
    # Start with minimal required config
    config = {}
    
    # Add category-specific defaults
    if spec.category == ToolCategory.HTTP:
        config.update({
            'method': 'POST',
            'timeout': 30,
            'headers': {'Content-Type': 'application/json'}
        })
    elif spec.category == ToolCategory.DATABASE:
        config.update({
            'query_type': 'select',
            'timeout': 30,
            'result_limit': 100
        })
    elif spec.category == ToolCategory.VECTOR_SEARCH:
        config.update({
            'top_k': 10,
            'similarity_threshold': 0.5
        })
    
    config.update(overrides)
    return config


def get_tool_template(category: Union[str, ToolCategory]) -> str:
    """
    Generate a template configuration string for a tool category.
    
    Args:
        category: Tool category
        
    Returns:
        Template configuration as formatted string
    """
    spec = get_tool_spec(category)
    
    lines = [f'tool "my_{spec.category.value}_tool" {{']
    lines.append(f'    description: "Description for {spec.category.value} tool"')
    
    # Add required fields as comments
    if spec.required_fields:
        lines.append('')
        lines.append('    # Required fields:')
        for field in spec.required_fields:
            if field == 'method' and spec.category == ToolCategory.HTTP:
                lines.append(f'    {field}: "POST"')
            elif field == 'url' and spec.category == ToolCategory.HTTP:
                lines.append(f'    {field}: "https://api.example.com/endpoint"')
            elif field == 'connection' and spec.category == ToolCategory.DATABASE:
                lines.append(f'    {field}: "my_database"')
            elif field == 'query_type' and spec.category == ToolCategory.DATABASE:
                lines.append(f'    {field}: "select"')
            elif field == 'index_name' and spec.category == ToolCategory.VECTOR_SEARCH:
                lines.append(f'    {field}: "my_vector_index"')
            else:
                lines.append(f'    {field}: "..."')
    
    # Add common optional fields as comments
    if spec.optional_fields:
        lines.append('')
        lines.append('    # Optional fields:')
        for field in spec.optional_fields[:3]:  # Show first 3 optional fields
            lines.append(f'    # {field}: ...')
        if len(spec.optional_fields) > 3:
            lines.append(f'    # ... and {len(spec.optional_fields) - 3} more')
    
    lines.append('}')
    return '\\n'.join(lines)