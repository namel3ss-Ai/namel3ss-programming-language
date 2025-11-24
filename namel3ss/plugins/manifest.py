"""
Production-grade plugin manifest and metadata handling for Namel3ss.

This module defines the plugin manifest format (n3-plugin.toml) and provides
robust validation, loading, and metadata management for the plugin ecosystem.

Key Components:
    - PluginManifest: Core plugin metadata and configuration
    - PluginType: Enumeration of supported plugin types
    - PluginCompatibility: Version compatibility specification
    - PluginSecurity: Security and capability declarations
    - PluginEntryPoint: Entry point definitions for plugin factories
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import tomllib
from pydantic import BaseModel, Field, validator

from namel3ss.security.capabilities import CapabilityType
from namel3ss.security.base import PermissionLevel


class PluginType(str, Enum):
    """Supported plugin types for the Namel3ss ecosystem."""
    
    CONNECTOR = "connector"          # Database, API, service connectors
    TOOL = "tool"                   # Callable functions, integrations
    DATASET = "dataset"             # Dataset adapters and processors
    TEMPLATE = "template"           # Prompt, chain, agent templates
    PROVIDER = "provider"           # LLM, embedding, vector store providers
    EVALUATOR = "evaluator"         # Model evaluation and testing tools
    TRANSFORMER = "transformer"    # Data transformation utilities
    MIXED = "mixed"                # Multi-type plugins


class PluginEntryPointType(str, Enum):
    """Types of plugin entry points."""
    
    FACTORY = "factory"             # Factory function returning instances
    CLASS = "class"                 # Class constructor
    MODULE = "module"               # Module with register_plugin function
    CALLABLE = "callable"           # Direct callable implementation


@dataclass(frozen=True)
class SemVer:
    """Semantic version with comparison support."""
    
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build: Optional[str] = None
    
    @classmethod
    def parse(cls, version: str) -> SemVer:
        """Parse semantic version string."""
        # Basic semver regex
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        match = re.match(pattern, version)
        
        if not match:
            raise ValueError(f"Invalid semantic version: {version}")
        
        major, minor, patch, pre_release, build = match.groups()
        
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            pre_release=pre_release,
            build=build
        )
    
    def __str__(self) -> str:
        """Format as semantic version string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.pre_release:
            version += f"-{self.pre_release}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __lt__(self, other: SemVer) -> bool:
        """Compare versions for ordering."""
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # Handle pre-release versions
        if self.pre_release is None and other.pre_release is not None:
            return False  # Release > pre-release
        elif self.pre_release is not None and other.pre_release is None:
            return True   # Pre-release < release
        elif self.pre_release is not None and other.pre_release is not None:
            return self.pre_release < other.pre_release
        
        return False  # Equal
    
    def __le__(self, other: SemVer) -> bool:
        """Less than or equal comparison."""
        return self < other or self == other
    
    def __gt__(self, other: SemVer) -> bool:
        """Greater than comparison."""
        return not (self <= other)
    
    def __ge__(self, other: SemVer) -> bool:
        """Greater than or equal comparison."""
        return not (self < other)
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, SemVer):
            return False
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch and
                self.pre_release == other.pre_release and
                self.build == other.build)
    
    def satisfies(self, constraint: str) -> bool:
        """Check if version satisfies constraint (^1.2.3, ~1.2.3, >=1.0.0, etc.)."""
        # Simplified constraint matching - can be expanded
        if constraint.startswith("^"):
            # Compatible release (caret range)
            target = SemVer.parse(constraint[1:])
            return (self.major == target.major and 
                   (self.minor > target.minor or 
                    (self.minor == target.minor and self.patch >= target.patch)))
        elif constraint.startswith("~"):
            # Approximate equivalent (tilde range)
            target = SemVer.parse(constraint[1:])
            return (self.major == target.major and 
                   self.minor == target.minor and 
                   self.patch >= target.patch)
        elif constraint.startswith(">="):
            target = SemVer.parse(constraint[2:])
            return self >= target
        elif constraint.startswith("<="):
            target = SemVer.parse(constraint[2:])
            return self <= target
        elif constraint.startswith("=="):
            target = SemVer.parse(constraint[2:])
            return self == target
        else:
            # Exact match
            target = SemVer.parse(constraint)
            return self == target


class PluginCompatibility(BaseModel):
    """Plugin compatibility specification."""
    
    min_namel3ss_version: Optional[str] = Field(
        None, 
        description="Minimum Namel3ss language version required"
    )
    max_namel3ss_version: Optional[str] = Field(
        None,
        description="Maximum Namel3ss language version supported"
    )
    required_extras: Set[str] = Field(
        default_factory=set,
        description="Required Namel3ss extras (e.g., 'ai', 'sql', 'websockets')"
    )
    python_version: Optional[str] = Field(
        None,
        description="Required Python version constraint"
    )
    
    @validator('min_namel3ss_version', 'max_namel3ss_version')
    def validate_versions(cls, v):
        if v is not None:
            try:
                SemVer.parse(v)
            except ValueError as e:
                raise ValueError(f"Invalid version format: {e}")
        return v


class PluginSecurity(BaseModel):
    """Plugin security and capability declarations."""
    
    required_capabilities: Set[CapabilityType] = Field(
        default_factory=set,
        description="Capabilities required by this plugin"
    )
    default_permission_level: PermissionLevel = Field(
        default=PermissionLevel.READ_ONLY,
        description="Default permission level for plugin operations"
    )
    network_access: bool = Field(
        default=False,
        description="Whether plugin requires network access"
    )
    filesystem_access: bool = Field(
        default=False,
        description="Whether plugin requires filesystem access"
    )
    subprocess_access: bool = Field(
        default=False,
        description="Whether plugin requires subprocess execution"
    )
    sensitive_data_access: bool = Field(
        default=False,
        description="Whether plugin accesses sensitive data"
    )
    
    class Config:
        use_enum_values = True


class PluginEntryPoint(BaseModel):
    """Plugin entry point definition."""
    
    type: PluginEntryPointType = Field(
        description="Type of entry point"
    )
    module: str = Field(
        description="Python module path"
    )
    attribute: Optional[str] = Field(
        None,
        description="Attribute name within module (for factory/class types)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameters for entry point"
    )
    
    class Config:
        use_enum_values = True


class PluginAuthor(BaseModel):
    """Plugin author information."""
    
    name: str = Field(description="Author name")
    email: Optional[str] = Field(None, description="Author email")
    url: Optional[str] = Field(None, description="Author website")


class PluginRepository(BaseModel):
    """Plugin repository information."""
    
    type: str = Field(description="Repository type (git, hg, etc.)")
    url: str = Field(description="Repository URL")


class PluginManifest(BaseModel):
    """
    Complete plugin manifest specification.
    
    This represents the parsed content of n3-plugin.toml files and provides
    comprehensive metadata for plugin discovery, validation, and loading.
    """
    
    # Basic metadata
    name: str = Field(
        description="Plugin name (must be unique in registry)"
    )
    version: str = Field(
        description="Plugin version (semantic versioning)"
    )
    description: str = Field(
        description="Brief description of plugin functionality"
    )
    
    # Plugin classification
    plugin_types: Set[PluginType] = Field(
        description="Types of functionality this plugin provides"
    )
    keywords: Set[str] = Field(
        default_factory=set,
        description="Keywords for discovery and search"
    )
    
    # Authorship and licensing
    authors: List[PluginAuthor] = Field(
        description="Plugin authors"
    )
    license: str = Field(
        description="SPDX license identifier"
    )
    homepage: Optional[str] = Field(
        None,
        description="Plugin homepage URL"
    )
    repository: Optional[PluginRepository] = Field(
        None,
        description="Source code repository"
    )
    documentation: Optional[str] = Field(
        None,
        description="Documentation URL"
    )
    
    # Compatibility and dependencies
    compatibility: PluginCompatibility = Field(
        default_factory=PluginCompatibility,
        description="Version compatibility requirements"
    )
    security: PluginSecurity = Field(
        default_factory=PluginSecurity,
        description="Security and capability declarations"
    )
    
    # Entry points
    entry_points: Dict[str, PluginEntryPoint] = Field(
        description="Plugin entry points by name"
    )
    
    # Configuration schema
    config_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="JSON schema for plugin configuration"
    )
    
    # Registry metadata (set by registry, not by plugin authors)
    registry_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata set by plugin registry"
    )
    
    @validator('name')
    def validate_name(cls, v):
        """Validate plugin name format."""
        if not re.match(r'^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$', v):
            raise ValueError(
                "Plugin name must be lowercase, start and end with alphanumeric, "
                "and contain only letters, numbers, dots, hyphens, and underscores"
            )
        return v
    
    @validator('version')
    def validate_version(cls, v):
        """Validate semantic version format."""
        try:
            SemVer.parse(v)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {e}")
        return v
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Validate keywords format."""
        for keyword in v:
            if not re.match(r'^[a-z0-9]([a-z0-9-]*[a-z0-9])?$', keyword):
                raise ValueError(
                    f"Invalid keyword '{keyword}': must be lowercase alphanumeric with hyphens"
                )
        return v
    
    @validator('entry_points')
    def validate_entry_points(cls, v):
        """Validate entry points are not empty."""
        if not v:
            raise ValueError("Plugin must define at least one entry point")
        return v
    
    @property
    def semver(self) -> SemVer:
        """Get parsed semantic version."""
        return SemVer.parse(self.version)
    
    def is_compatible_with_namel3ss(self, namel3ss_version: str) -> bool:
        """Check if plugin is compatible with Namel3ss version."""
        namel3ss_semver = SemVer.parse(namel3ss_version)
        compatibility = self.compatibility
        
        if compatibility.min_namel3ss_version:
            if not namel3ss_semver >= SemVer.parse(compatibility.min_namel3ss_version):
                return False
        
        if compatibility.max_namel3ss_version:
            if not namel3ss_semver <= SemVer.parse(compatibility.max_namel3ss_version):
                return False
        
        return True
    
    def requires_capabilities(self) -> Set[CapabilityType]:
        """Get all capabilities required by this plugin."""
        return self.security.required_capabilities
    
    class Config:
        use_enum_values = True


def load_plugin_manifest(manifest_path: Union[str, Path]) -> PluginManifest:
    """
    Load and validate plugin manifest from n3-plugin.toml file.
    
    Args:
        manifest_path: Path to n3-plugin.toml file
        
    Returns:
        Validated PluginManifest instance
        
    Raises:
        FileNotFoundError: If manifest file doesn't exist
        ValueError: If manifest format is invalid
        ValidationError: If manifest validation fails
    """
    manifest_path = Path(manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Plugin manifest not found: {manifest_path}")
    
    try:
        with open(manifest_path, 'rb') as f:
            manifest_data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in manifest file: {e}")
    
    # Extract plugin section
    plugin_data = manifest_data.get('plugin')
    if not plugin_data:
        raise ValueError("Manifest must contain [plugin] section")
    
    return PluginManifest(**plugin_data)


def validate_plugin_manifest_content(content: str) -> PluginManifest:
    """
    Validate plugin manifest from TOML content string.
    
    Args:
        content: TOML content as string
        
    Returns:
        Validated PluginManifest instance
        
    Raises:
        ValueError: If manifest format is invalid
        ValidationError: If manifest validation fails
    """
    try:
        manifest_data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML content: {e}")
    
    plugin_data = manifest_data.get('plugin')
    if not plugin_data:
        raise ValueError("Manifest must contain [plugin] section")
    
    return PluginManifest(**plugin_data)


__all__ = [
    # Core types
    "PluginManifest",
    "PluginType", 
    "PluginEntryPointType",
    "PluginCompatibility",
    "PluginSecurity",
    "PluginEntryPoint",
    "PluginAuthor",
    "PluginRepository",
    "SemVer",
    
    # Loading functions
    "load_plugin_manifest",
    "validate_plugin_manifest_content",
]