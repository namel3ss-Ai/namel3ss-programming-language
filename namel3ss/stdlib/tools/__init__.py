"""Tools module of the Namel3ss standard library."""

from .base import (
    ToolCategory,
    ToolInterface,
    HTTPToolSpec,
    DatabaseToolSpec,
    VectorSearchToolSpec,
    STANDARD_TOOL_SPECS,
    get_tool_spec,
    list_tool_categories,
    get_category_description,
)

from .validation import (
    ToolValidationError,
    validate_tool_config,
    validate_tool_config_strict,
    suggest_tool_config,
    get_tool_template,
)

__all__ = [
    # Interface definitions
    "ToolCategory",
    "ToolInterface", 
    "HTTPToolSpec",
    "DatabaseToolSpec",
    "VectorSearchToolSpec", 
    "STANDARD_TOOL_SPECS",
    "get_tool_spec",
    "list_tool_categories",
    "get_category_description",
    
    # Validation
    "ToolValidationError",
    "validate_tool_config",
    "validate_tool_config_strict",
    "suggest_tool_config", 
    "get_tool_template",
]