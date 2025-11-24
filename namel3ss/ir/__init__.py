"""
Intermediate Representation (IR) for Namel3ss.

This package provides a runtime-agnostic intermediate representation of compiled
Namel3ss programs. The IR separates language semantics from runtime implementation,
enabling multiple backend targets (HTTP/REST, serverless, gRPC, etc.).

Core Concepts:
--------------
- **BackendIR**: Complete backend specification with endpoints, agents, prompts
- **FrontendIR**: Frontend specification with pages, components, routing
- **EndpointIR**: Runtime-agnostic API endpoint specification
- **AgentSpec**: Multi-agent orchestration specification
- **PromptSpec**: Structured prompt specification with schemas

Usage:
------
    from namel3ss import Parser, compile_to_ir
    from namel3ss.ir import BackendIR, FrontendIR
    
    # Parse and compile to IR
    parser = Parser(source_code)
    module = parser.parse()
    app = module.body[0]
    
    backend_ir = compile_to_ir(app, target='backend')
    frontend_ir = compile_to_ir(app, target='frontend')
    
    # Runtime adapters consume IR
    # (from runtime-specific packages)
"""

from .spec import (
    # Core IR types
    BackendIR,
    FrontendIR,
    
    # Backend specifications
    EndpointIR,
    AgentSpec,
    PromptSpec,
    ToolSpec,
    DatasetSpec,
    FrameSpec,
    MemorySpec,
    ChainSpec,
    InsightSpec,
    LocalModelSpec,
    
    # Frontend specifications
    PageSpec,
    ComponentSpec,
    RouteSpec,
    
    # Type specifications
    TypeSpec,
    SchemaField,
    
    # Enumerations
    HTTPMethod,
    MemoryScope,
    CacheStrategy,
)

from .builder import (
    build_backend_ir,
    build_frontend_ir,
)

from .serialization import (
    serialize_backend_ir,
    deserialize_backend_ir,
    serialize_frontend_ir,
    deserialize_frontend_ir,
)

# Plugin IR extensions
from .plugins import (
    PluginType,
    PluginSpec,
    PluginProvidedSpec,
    PluginProvidedToolSpec,
    PluginProvidedConnectorSpec,
    PluginProvidedDatasetSpec,
    PluginProvidedTemplateSpec,
    EnhancedToolSpec,
    EnhancedConnectorSpec,
    PluginRequirementSpec,
    BackendIRWithPlugins,
    validate_plugin_ir,
)

__all__ = [
    # Core IR
    "BackendIR",
    "FrontendIR",
    
    # Backend specs
    "EndpointIR",
    "AgentSpec",
    "PromptSpec",
    "ToolSpec",
    "DatasetSpec",
    "FrameSpec",
    "MemorySpec",
    "ChainSpec",
    "InsightSpec",
    
    # Frontend specs
    "PageSpec",
    "ComponentSpec",
    "RouteSpec",
    
    # Types
    "TypeSpec",
    "SchemaField",
    
    # Enums
    "HTTPMethod",
    "MemoryScope",
    "CacheStrategy",
    
    # Builder functions
    "build_backend_ir",
    "build_frontend_ir",
    
    # Serialization
    "serialize_backend_ir",
    "deserialize_backend_ir",
    "serialize_frontend_ir",
    "deserialize_frontend_ir",
    
    # Plugin IR types
    "PluginType",
    "PluginSpec",
    "PluginProvidedSpec",
    "PluginProvidedToolSpec",
    "PluginProvidedConnectorSpec",
    "PluginProvidedDatasetSpec",
    "PluginProvidedTemplateSpec",
    "EnhancedToolSpec",
    "EnhancedConnectorSpec",
    "PluginRequirementSpec",
    "BackendIRWithPlugins",
    "validate_plugin_ir",
]
