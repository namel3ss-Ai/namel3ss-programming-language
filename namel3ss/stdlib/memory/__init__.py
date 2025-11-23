"""Memory module of the Namel3ss standard library."""

from .policies import (
    MemoryPolicy,
    MemoryPolicySpec,
    STANDARD_MEMORY_POLICIES,
    get_memory_policy_spec,
    list_memory_policies,
    get_policy_description,
    get_default_config,
)

from .validation import (
    MemoryValidationError,
    validate_memory_config,
    validate_memory_config_strict,
    suggest_memory_config,
)

__all__ = [
    # Policy definitions
    "MemoryPolicy",
    "MemoryPolicySpec",
    "STANDARD_MEMORY_POLICIES",
    "get_memory_policy_spec",
    "list_memory_policies", 
    "get_policy_description",
    "get_default_config",
    
    # Validation
    "MemoryValidationError",
    "validate_memory_config",
    "validate_memory_config_strict",
    "suggest_memory_config",
]