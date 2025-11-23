"""LLM module of the Namel3ss standard library."""

from .config import (
    LLMConfigField,
    LLMConfigSpec,
    STANDARD_LLM_FIELDS,
    get_llm_config_spec,
    list_llm_config_fields,
    get_field_description,
    get_default_value,
    get_standard_llm_config,
)

from .validation import (
    LLMValidationError,
    validate_llm_config,
    validate_llm_config_strict,
    suggest_llm_config,
)

__all__ = [
    # Configuration definitions
    "LLMConfigField",
    "LLMConfigSpec", 
    "STANDARD_LLM_FIELDS",
    "get_llm_config_spec",
    "list_llm_config_fields",
    "get_field_description",
    "get_default_value",
    "get_standard_llm_config",
    
    # Validation
    "LLMValidationError",
    "validate_llm_config",
    "validate_llm_config_strict", 
    "suggest_llm_config",
]