"""
AST extensions for plugin references in Namel3ss language.

Adds first-class plugin support to the AST, allowing plugins to be
referenced directly in N3 code for tools, connectors, datasets, and templates.

New AST nodes:
- PluginReference: Reference to an external plugin
- PluginUsage: Usage declaration for importing plugins
- Enhanced tool/connector nodes with plugin support

Syntax examples:
    use plugin "acme.postgres_connector"
    
    tool "translate" provided_by plugin "acme.translation_tools"
    
    connector "db" {
        type: postgres
        provider: plugin "acme.postgres_connector"
        config: { ... }
    }
    
    dataset "analytics" from plugin "acme.analytics_connector" {
        config: { ... }
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .source_location import SourceLocation


@dataclass
class PluginReference:
    """
    Reference to an external plugin in N3 code.
    
    Represents a reference to a plugin that provides tools, connectors,
    datasets, or other functionality. The plugin must be installed and
    compatible with the current Namel3ss environment.
    
    Example DSL:
        plugin "acme.postgres_connector" version "^1.2.0"
    """
    
    name: str
    """Plugin name (e.g., 'acme.postgres_connector')"""
    
    version_constraint: Optional[str] = None
    """Optional version constraint (e.g., '^1.2.0', '>=1.0.0')"""
    
    alias: Optional[str] = None
    """Optional alias for plugin reference"""
    
    entry_point: Optional[str] = None
    """Specific entry point within plugin"""
    
    source_location: Optional[SourceLocation] = None
    """Source location in N3 code"""
    
    def __str__(self) -> str:
        """String representation of plugin reference."""
        result = f'plugin "{self.name}"'
        
        if self.version_constraint:
            result += f' version "{self.version_constraint}"'
        
        if self.alias:
            result += f' as {self.alias}'
        
        if self.entry_point:
            result += f' entry "{self.entry_point}"'
        
        return result


@dataclass 
class PluginUsage:
    """
    Plugin usage declaration (use plugin statement).
    
    Declares that a plugin should be made available in the current scope.
    Similar to import statements in other languages.
    
    Example DSL:
        use plugin "acme.postgres_connector" version "^1.2.0" as postgres
        use plugin "acme.translation_tools" entry "translate_text"
    """
    
    plugin_ref: PluginReference
    """Reference to the plugin being used"""
    
    scope: str = "global"
    """Scope where plugin is available ('global', 'local')"""
    
    required: bool = True
    """Whether plugin is required (compilation fails if missing)"""
    
    source_location: Optional[SourceLocation] = None
    """Source location in N3 code"""


@dataclass
class PluginProvidedTool:
    """
    Tool definition that's provided by a plugin.
    
    Extends the standard tool definition to include plugin provenance
    and configuration delegation to the plugin.
    
    Example DSL:
        tool "translate" provided_by plugin "acme.translation_tools" {
            entry: "translate_text"
            config: {
                default_language: "en"
            }
        }
    """
    
    name: str
    """Tool name"""
    
    plugin_ref: PluginReference
    """Plugin providing this tool"""
    
    config: Dict[str, Any] = field(default_factory=dict)
    """Tool-specific configuration passed to plugin"""
    
    description: Optional[str] = None
    """Tool description"""
    
    source_location: Optional[SourceLocation] = None
    """Source location in N3 code"""


@dataclass
class PluginProvidedConnector:
    """
    Connector definition that's provided by a plugin.
    
    Extends the standard connector definition to include plugin provenance
    and configuration delegation to the plugin.
    
    Example DSL:
        connector "main_db" {
            type: postgres
            provider: plugin "acme.postgres_connector"
            config: {
                host: "{{env.DB_HOST}}"
                database: "production"
            }
        }
    """
    
    name: str
    """Connector name"""
    
    connector_type: str
    """Type of connector (postgres, redis, s3, etc.)"""
    
    plugin_ref: PluginReference
    """Plugin providing this connector"""
    
    config: Dict[str, Any] = field(default_factory=dict)
    """Connector configuration"""
    
    description: Optional[str] = None
    """Connector description"""
    
    source_location: Optional[SourceLocation] = None
    """Source location in N3 code"""


@dataclass
class PluginProvidedDataset:
    """
    Dataset definition that's provided by a plugin.
    
    Enables datasets to be loaded from plugin-provided adapters
    for various data sources and formats.
    
    Example DSL:
        dataset "analytics" from plugin "acme.analytics_connector" {
            query: "SELECT * FROM events WHERE created_at > '2024-01-01'"
            cache_policy: ttl
            ttl_seconds: 3600
        }
    """
    
    name: str
    """Dataset name"""
    
    plugin_ref: PluginReference
    """Plugin providing this dataset"""
    
    config: Dict[str, Any] = field(default_factory=dict)
    """Dataset configuration"""
    
    description: Optional[str] = None
    """Dataset description"""
    
    source_location: Optional[SourceLocation] = None
    """Source location in N3 code"""


@dataclass
class PluginProvidedTemplate:
    """
    Template (prompt, chain, agent) provided by a plugin.
    
    Enables sharing of reusable templates through the plugin ecosystem.
    
    Example DSL:
        template "support_agent" from plugin "acme.support_templates" {
            variant: "v2"
            config: {
                company_name: "ACME Corp"
                support_level: "premium"
            }
        }
    """
    
    name: str
    """Template name"""
    
    template_type: str
    """Type of template (prompt, chain, agent)"""
    
    plugin_ref: PluginReference
    """Plugin providing this template"""
    
    config: Dict[str, Any] = field(default_factory=dict)
    """Template configuration"""
    
    description: Optional[str] = None
    """Template description"""
    
    source_location: Optional[SourceLocation] = None
    """Source location in N3 code"""


# Extensions to existing AST nodes to support plugins

@dataclass
class EnhancedTool:
    """
    Enhanced tool definition with optional plugin support.
    
    Backwards-compatible extension of existing tool definitions
    that optionally supports plugin-provided implementations.
    """
    
    name: str
    """Tool name"""
    
    tool_type: str
    """Tool type (http, python, database, etc.)"""
    
    # Standard tool fields
    endpoint: Optional[str] = None
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    
    # Plugin extension
    plugin_ref: Optional[PluginReference] = None
    """Optional plugin providing tool implementation"""
    
    config: Dict[str, Any] = field(default_factory=dict)
    """Tool configuration"""
    
    description: Optional[str] = None
    """Tool description"""
    
    source_location: Optional[SourceLocation] = None
    """Source location in N3 code"""
    
    @property
    def is_plugin_provided(self) -> bool:
        """Check if tool is provided by a plugin."""
        return self.plugin_ref is not None


@dataclass
class EnhancedConnector:
    """
    Enhanced connector definition with optional plugin support.
    
    Backwards-compatible extension of existing connector definitions
    that optionally supports plugin-provided implementations.
    """
    
    name: str
    """Connector name"""
    
    connector_type: str
    """Connector type (postgres, redis, s3, etc.)"""
    
    # Standard connector fields
    provider: str = ""
    
    # Plugin extension
    plugin_ref: Optional[PluginReference] = None
    """Optional plugin providing connector implementation"""
    
    config: Dict[str, Any] = field(default_factory=dict)
    """Connector configuration"""
    
    description: Optional[str] = None
    """Connector description"""
    
    source_location: Optional[SourceLocation] = None
    """Source location in N3 code"""
    
    @property
    def is_plugin_provided(self) -> bool:
        """Check if connector is provided by a plugin."""
        return self.plugin_ref is not None
    
    @property
    def effective_provider(self) -> str:
        """Get effective provider (plugin or standard provider)."""
        if self.plugin_ref:
            return f"plugin:{self.plugin_ref.name}"
        return self.provider


@dataclass
class PluginCapabilityRequirement:
    """
    Represents a capability requirement imposed by plugin usage.
    
    Used during compilation to track what capabilities are needed
    for the security system to validate at runtime.
    """
    
    plugin_name: str
    """Name of plugin requiring capability"""
    
    capability: str
    """Capability name required"""
    
    reason: str
    """Human-readable reason for requirement"""
    
    optional: bool = False
    """Whether capability is optional"""


@dataclass
class PluginMetadata:
    """
    Compilation-time metadata about plugins used in a program.
    
    Collected during AST processing to enable validation,
    capability checking, and runtime plugin resolution.
    """
    
    used_plugins: Set[str] = field(default_factory=set)
    """Set of plugin names referenced in code"""
    
    plugin_references: List[PluginReference] = field(default_factory=list)
    """All plugin references found in AST"""
    
    capability_requirements: List[PluginCapabilityRequirement] = field(default_factory=list)
    """Capability requirements from plugins"""
    
    plugin_provided_tools: List[PluginProvidedTool] = field(default_factory=list)
    """Tools provided by plugins"""
    
    plugin_provided_connectors: List[PluginProvidedConnector] = field(default_factory=list)
    """Connectors provided by plugins"""
    
    plugin_provided_datasets: List[PluginProvidedDataset] = field(default_factory=list)
    """Datasets provided by plugins"""
    
    plugin_provided_templates: List[PluginProvidedTemplate] = field(default_factory=list)
    """Templates provided by plugins"""
    
    def add_plugin_usage(self, plugin_ref: PluginReference) -> None:
        """Add a plugin reference to metadata."""
        self.used_plugins.add(plugin_ref.name)
        self.plugin_references.append(plugin_ref)
    
    def add_capability_requirement(
        self, 
        plugin_name: str, 
        capability: str, 
        reason: str,
        optional: bool = False
    ) -> None:
        """Add a capability requirement."""
        req = PluginCapabilityRequirement(
            plugin_name=plugin_name,
            capability=capability,
            reason=reason,
            optional=optional
        )
        self.capability_requirements.append(req)
    
    def get_required_capabilities(self) -> Set[str]:
        """Get all required capabilities from plugins."""
        return {
            req.capability 
            for req in self.capability_requirements 
            if not req.optional
        }
    
    def get_plugins_by_type(self, plugin_type: str) -> List[str]:
        """Get plugin names providing specific type of functionality."""
        plugins = set()
        
        if plugin_type == "tool":
            plugins.update(tool.plugin_ref.name for tool in self.plugin_provided_tools)
        elif plugin_type == "connector":
            plugins.update(conn.plugin_ref.name for conn in self.plugin_provided_connectors)
        elif plugin_type == "dataset":
            plugins.update(ds.plugin_ref.name for ds in self.plugin_provided_datasets)
        elif plugin_type == "template":
            plugins.update(tmpl.plugin_ref.name for tmpl in self.plugin_provided_templates)
        
        return list(plugins)


# Utility functions for AST processing

def extract_plugin_metadata(ast_node: Any) -> PluginMetadata:
    """
    Extract plugin metadata from an AST node tree.
    
    Recursively traverses AST to find all plugin references
    and build comprehensive metadata for compilation.
    """
    metadata = PluginMetadata()
    
    # This would be implemented to traverse the AST tree
    # and extract all plugin-related nodes
    
    return metadata


def validate_plugin_references(metadata: PluginMetadata) -> List[str]:
    """
    Validate plugin references in metadata.
    
    Returns list of validation errors, empty list if valid.
    """
    errors = []
    
    # Check for duplicate plugin names with different versions
    plugin_versions = {}
    for ref in metadata.plugin_references:
        if ref.name in plugin_versions:
            if ref.version_constraint != plugin_versions[ref.name]:
                errors.append(
                    f"Conflicting version constraints for plugin {ref.name}: "
                    f"{plugin_versions[ref.name]} vs {ref.version_constraint}"
                )
        else:
            plugin_versions[ref.name] = ref.version_constraint
    
    # Additional validations would go here
    
    return errors


__all__ = [
    # Core plugin AST nodes
    "PluginReference",
    "PluginUsage", 
    "PluginProvidedTool",
    "PluginProvidedConnector",
    "PluginProvidedDataset",
    "PluginProvidedTemplate",
    
    # Enhanced existing nodes
    "EnhancedTool",
    "EnhancedConnector",
    
    # Metadata and validation
    "PluginCapabilityRequirement",
    "PluginMetadata",
    "extract_plugin_metadata",
    "validate_plugin_references",
]