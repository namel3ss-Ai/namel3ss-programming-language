"""
Namel3ss Standard Library.

Provides canonical, production-ready AI primitives built on top of the core language:
- Memory policies for conversation management
- LLM configuration standards and validation
- Tool interface definitions for HTTP, DB, and vector operations
- Provider-neutral abstractions and best practices

The standard library is designed to be:
- Provider neutral (targets interfaces, not specific vendors)
- Type safe (enforced by the language's type system)
- Production ready (robust validation and error handling)
- Discoverable (well-documented and CLI-accessible)
"""

from .memory import (
    MemoryPolicy,
    MemoryPolicySpec,
    STANDARD_MEMORY_POLICIES,
    validate_memory_config,
    get_memory_policy_spec,
)

from .llm import (
    LLMConfigField,
    LLMConfigSpec,
    STANDARD_LLM_FIELDS,
    validate_llm_config,
    get_llm_config_spec,
)

from .tools import (
    ToolCategory,
    ToolInterface,
    HTTPToolSpec,
    DatabaseToolSpec,
    VectorSearchToolSpec,
    validate_tool_config,
    get_tool_spec,
)

from .registry import StandardLibraryRegistry, get_stdlib_registry

__version__ = "1.0.0"

__all__ = [
    # Memory policies
    "MemoryPolicy",
    "MemoryPolicySpec", 
    "STANDARD_MEMORY_POLICIES",
    "validate_memory_config",
    "get_memory_policy_spec",
    
    # LLM configuration
    "LLMConfigField",
    "LLMConfigSpec",
    "STANDARD_LLM_FIELDS",
    "validate_llm_config", 
    "get_llm_config_spec",
    
    # Tool interfaces
    "ToolCategory",
    "ToolInterface",
    "HTTPToolSpec",
    "DatabaseToolSpec", 
    "VectorSearchToolSpec",
    "validate_tool_config",
    "get_tool_spec",
    
    # Registry
    "StandardLibraryRegistry",
    "get_stdlib_registry",
]