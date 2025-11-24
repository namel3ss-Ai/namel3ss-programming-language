"""
IR extensions for plugin support in Namel3ss.

Extends the Intermediate Representation to include plugin references,
enabling runtime-agnostic representation of plugin-based functionality.

Key Components:
    - PluginSpec: Plugin metadata and configuration in IR
    - PluginProvidedSpec: Base for plugin-provided functionality
    - Enhanced IR specs with plugin support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from .spec import TypeSpec


class PluginType(str, Enum):
    """Plugin types in IR representation."""
    CONNECTOR = "connector"
    TOOL = "tool"
    DATASET = "dataset"
    TEMPLATE = "template"
    PROVIDER = "provider"
    EVALUATOR = "evaluator"
    TRANSFORMER = "transformer"
    MIXED = "mixed"


@dataclass
class PluginSpec:
    """
    Plugin specification in IR.
    
    Runtime-agnostic representation of a plugin reference with
    all metadata needed for runtime resolution and instantiation.
    """
    
    name: str
    """Plugin name (e.g., 'acme.postgres_connector')"""
    
    version_constraint: Optional[str] = None
    """Version constraint (e.g., '^1.2.0')"""
    
    entry_point: Optional[str] = None
    """Specific entry point within plugin"""
    
    plugin_types: Set[PluginType] = field(default_factory=set)
    """Types of functionality this plugin provides"""
    
    required_capabilities: Set[str] = field(default_factory=set)
    """Capabilities required by this plugin"""
    
    config_schema: Optional[TypeSpec] = None
    """Configuration schema for this plugin"""
    
    security_metadata: Dict[str, Any] = field(default_factory=dict)
    """Security-related metadata"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional plugin metadata"""


@dataclass
class PluginProvidedSpec:
    """
    Base specification for plugin-provided functionality.
    
    Contains common fields for any functionality that's provided
    by a plugin rather than built-in implementations.
    """
    
    plugin_spec: PluginSpec
    """Plugin providing this functionality"""
    
    required: bool = True
    """Whether plugin is required (fail if unavailable)"""
    
    config: Dict[str, Any] = field(default_factory=dict)
    """Configuration passed to plugin"""
    
    fallback_spec: Optional[Any] = None
    """Optional fallback if plugin unavailable"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""


@dataclass
class PluginProvidedToolSpec(PluginProvidedSpec):
    """
    Tool specification provided by a plugin.
    
    Extends ToolSpec to include plugin provenance and configuration.
    """
    
    name: str = ""
    """Tool name"""
    
    input_schema: Optional[TypeSpec] = None
    """Tool input schema"""
    
    output_schema: Optional[TypeSpec] = None
    """Tool output schema"""
    
    timeout: float = 30.0
    """Tool execution timeout"""
    
    retry_config: Dict[str, Any] = field(default_factory=dict)
    """Retry configuration"""


@dataclass
class PluginProvidedConnectorSpec(PluginProvidedSpec):
    """
    Connector specification provided by a plugin.
    
    Extends ConnectorSpec to include plugin provenance.
    """
    
    name: str = ""
    """Connector name"""
    
    connector_type: str = "generic"
    """Type of connector (postgres, redis, etc.)"""
    
    connection_schema: Optional[TypeSpec] = None
    """Schema for connection configuration"""


@dataclass
class PluginProvidedDatasetSpec(PluginProvidedSpec):
    """
    Dataset specification provided by a plugin.
    
    Represents datasets loaded through plugin adapters.
    """
    
    name: str = ""
    """Dataset name"""
    
    schema: Optional[TypeSpec] = None
    """Dataset schema"""
    
    refresh_config: Dict[str, Any] = field(default_factory=dict)
    """Dataset refresh configuration"""


@dataclass
class PluginProvidedTemplateSpec(PluginProvidedSpec):
    """
    Template specification provided by a plugin.
    
    Represents prompts, chains, or agents provided by plugins.
    """
    
    name: str = ""
    """Template name"""
    
    template_type: str = "generic"
    """Type of template (prompt, chain, agent)"""
    
    input_schema: Optional[TypeSpec] = None
    """Template input schema"""
    
    output_schema: Optional[TypeSpec] = None
    """Template output schema"""
    
    template_config: Dict[str, Any] = field(default_factory=dict)
    """Template-specific configuration"""


# Enhanced existing IR specs with plugin support

@dataclass
class EnhancedToolSpec:
    """
    Enhanced tool specification with optional plugin support.
    
    Backwards-compatible extension of ToolSpec that can represent
    both built-in tools and plugin-provided tools.
    """
    
    name: str
    input_schema: TypeSpec
    output_schema: TypeSpec
    timeout: float = 30.0
    
    # Built-in tool fields
    tool_type: Optional[str] = None  # "http", "python", "database"
    endpoint: Optional[str] = None
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Plugin support
    plugin_spec: Optional[PluginSpec] = None
    plugin_config: Dict[str, Any] = field(default_factory=dict)
    
    # Common fields
    description: Optional[str] = None
    retry_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_plugin_provided(self) -> bool:
        """Check if tool is provided by a plugin."""
        return self.plugin_spec is not None


@dataclass
class EnhancedConnectorSpec:
    """
    Enhanced connector specification with optional plugin support.
    
    Can represent both built-in connectors and plugin-provided ones.
    """
    
    name: str
    connector_type: str
    
    # Built-in connector fields
    provider: str = ""
    connection_string: Optional[str] = None
    
    # Plugin support
    plugin_spec: Optional[PluginSpec] = None
    plugin_config: Dict[str, Any] = field(default_factory=dict)
    
    # Common fields
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_plugin_provided(self) -> bool:
        """Check if connector is provided by a plugin."""
        return self.plugin_spec is not None
    
    @property
    def effective_provider(self) -> str:
        """Get effective provider (plugin or built-in)."""
        if self.plugin_spec:
            return f"plugin:{self.plugin_spec.name}"
        return self.provider


@dataclass
class PluginRequirementSpec:
    """
    Specification for plugin requirements in compiled IR.
    
    Represents the plugins needed by a compiled Namel3ss program,
    including version constraints and capability requirements.
    """
    
    required_plugins: List[PluginSpec] = field(default_factory=list)
    """Plugins required by this program"""
    
    optional_plugins: List[PluginSpec] = field(default_factory=list)
    """Optional plugins that enhance functionality"""
    
    capability_requirements: Dict[str, List[str]] = field(default_factory=dict)
    """Capabilities required by each plugin"""
    
    fallback_configurations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Fallback configurations when plugins unavailable"""
    
    compatibility_metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata for compatibility checking"""
    
    def get_all_required_capabilities(self) -> Set[str]:
        """Get all capabilities required by all plugins."""
        all_capabilities = set()
        for plugin_spec in self.required_plugins:
            all_capabilities.update(plugin_spec.required_capabilities)
        return all_capabilities
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginSpec]:
        """Get plugins providing specific type of functionality."""
        matching_plugins = []
        for plugin_spec in self.required_plugins + self.optional_plugins:
            if plugin_type in plugin_spec.plugin_types:
                matching_plugins.append(plugin_spec)
        return matching_plugins


# Extensions to existing IR types

@dataclass
class BackendIRWithPlugins:
    """
    Extended BackendIR with plugin support.
    
    Includes plugin requirements and enhanced specifications
    for tools and connectors that may be plugin-provided.
    """
    
    # Standard BackendIR fields would be here
    endpoints: List[Any] = field(default_factory=list)
    prompts: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, Any] = field(default_factory=dict)
    
    # Plugin extensions
    plugin_requirements: PluginRequirementSpec = field(default_factory=PluginRequirementSpec)
    """Plugin requirements for this backend"""
    
    enhanced_tools: Dict[str, EnhancedToolSpec] = field(default_factory=dict)
    """Tools with potential plugin support"""
    
    enhanced_connectors: Dict[str, EnhancedConnectorSpec] = field(default_factory=dict)
    """Connectors with potential plugin support"""
    
    plugin_provided_datasets: Dict[str, PluginProvidedDatasetSpec] = field(default_factory=dict)
    """Datasets provided by plugins"""
    
    plugin_provided_templates: Dict[str, PluginProvidedTemplateSpec] = field(default_factory=dict)
    """Templates provided by plugins"""
    
    def get_required_plugins(self) -> List[str]:
        """Get names of all required plugins."""
        return [spec.name for spec in self.plugin_requirements.required_plugins]
    
    def has_plugin_dependencies(self) -> bool:
        """Check if this backend depends on any plugins."""
        return (
            len(self.plugin_requirements.required_plugins) > 0 or
            any(tool.is_plugin_provided for tool in self.enhanced_tools.values()) or
            any(conn.is_plugin_provided for conn in self.enhanced_connectors.values()) or
            len(self.plugin_provided_datasets) > 0 or
            len(self.plugin_provided_templates) > 0
        )


def validate_plugin_ir(ir: BackendIRWithPlugins) -> List[str]:
    """
    Validate plugin-related IR specifications.
    
    Returns list of validation errors, empty if valid.
    """
    errors = []
    
    # Check that all plugin references are declared
    declared_plugins = {spec.name for spec in ir.plugin_requirements.required_plugins}
    declared_plugins.update(spec.name for spec in ir.plugin_requirements.optional_plugins)
    
    # Check tools
    for tool_name, tool_spec in ir.enhanced_tools.items():
        if tool_spec.plugin_spec and tool_spec.plugin_spec.name not in declared_plugins:
            errors.append(
                f"Tool {tool_name} references undeclared plugin {tool_spec.plugin_spec.name}"
            )
    
    # Check connectors
    for conn_name, conn_spec in ir.enhanced_connectors.items():
        if conn_spec.plugin_spec and conn_spec.plugin_spec.name not in declared_plugins:
            errors.append(
                f"Connector {conn_name} references undeclared plugin {conn_spec.plugin_spec.name}"
            )
    
    # Check datasets
    for dataset_name, dataset_spec in ir.plugin_provided_datasets.items():
        if dataset_spec.plugin_spec.name not in declared_plugins:
            errors.append(
                f"Dataset {dataset_name} references undeclared plugin {dataset_spec.plugin_spec.name}"
            )
    
    # Check templates
    for template_name, template_spec in ir.plugin_provided_templates.items():
        if template_spec.plugin_spec.name not in declared_plugins:
            errors.append(
                f"Template {template_name} references undeclared plugin {template_spec.plugin_spec.name}"
            )
    
    return errors


__all__ = [
    # Plugin IR types
    "PluginType",
    "PluginSpec",
    "PluginProvidedSpec",
    "PluginProvidedToolSpec",
    "PluginProvidedConnectorSpec", 
    "PluginProvidedDatasetSpec",
    "PluginProvidedTemplateSpec",
    
    # Enhanced IR types
    "EnhancedToolSpec",
    "EnhancedConnectorSpec",
    "PluginRequirementSpec",
    "BackendIRWithPlugins",
    
    # Validation
    "validate_plugin_ir",
]