"""
Package and module system for namel3ss.

This module provides the core types and utilities for the namel3ss package system,
including package manifests, dependency resolution, and module organization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from packaging import specifiers, version


# Exception classes
class PackageSystemError(Exception):
    """Base exception for package system errors."""
    pass


class PackageNotFoundError(PackageSystemError):
    """Raised when a required package cannot be found."""
    pass


class DependencyCycleError(PackageSystemError):
    """Raised when a circular dependency is detected."""
    pass


class PackageVersionConflictError(PackageSystemError):
    """Raised when package version requirements conflict."""
    pass


class InvalidPackageManifestError(PackageSystemError):
    """Raised when a package manifest is invalid or malformed."""
    pass


# Version constraint parsing
VERSION_CONSTRAINT_PATTERN = re.compile(
    r'^(?P<op>>=|<=|==|!=|>|<|~=|\^)?(?P<version>.+)$'
)


@dataclass
class PackageIdentifier:
    """Uniquely identifies a package with name and version."""
    
    name: str
    version: str
    
    def __str__(self) -> str:
        return f"{self.name}@{self.version}"
    
    @classmethod
    def parse(cls, identifier: str) -> PackageIdentifier:
        """Parse package identifier from string like 'my.pkg@1.0.0'."""
        if '@' in identifier:
            name, ver = identifier.rsplit('@', 1)
            return cls(name=name, version=ver)
        else:
            return cls(name=identifier, version="*")


@dataclass
class VersionConstraint:
    """Represents a version constraint like '>=1.0.0' or '^2.1.0'."""
    
    operator: str
    version_spec: str
    
    def __post_init__(self):
        self._specifier = specifiers.SpecifierSet(f"{self.operator}{self.version_spec}")
    
    def matches(self, version_string: str) -> bool:
        """Check if a version satisfies this constraint."""
        try:
            ver = version.Version(version_string)
            return ver in self._specifier
        except version.InvalidVersion:
            return False
    
    def __str__(self) -> str:
        return f"{self.operator}{self.version_spec}"
    
    @classmethod
    def parse(cls, constraint_str: str) -> VersionConstraint:
        """Parse version constraint from string like '>=1.0.0'."""
        match = VERSION_CONSTRAINT_PATTERN.match(constraint_str.strip())
        if not match:
            raise ValueError(f"Invalid version constraint: {constraint_str}")
        
        op = match.group('op') or '=='
        ver = match.group('version')
        return cls(operator=op, version_spec=ver)


@dataclass
class PackageDependency:
    """Represents a dependency on another package."""
    
    name: str
    constraint: VersionConstraint
    source: Optional[str] = None  # For git/local sources
    optional: bool = False
    
    def __str__(self) -> str:
        result = f"{self.name} {self.constraint}"
        if self.source:
            result += f" from {self.source}"
        if self.optional:
            result += " (optional)"
        return result
    
    @classmethod
    def parse(cls, name: str, spec: Union[str, Dict]) -> PackageDependency:
        """Parse dependency from manifest specification."""
        if isinstance(spec, str):
            return cls(name=name, constraint=VersionConstraint.parse(spec))
        elif isinstance(spec, dict):
            constraint_str = spec.get('version', '*')
            return cls(
                name=name,
                constraint=VersionConstraint.parse(constraint_str),
                source=spec.get('source'),
                optional=spec.get('optional', False)
            )
        else:
            raise ValueError(f"Invalid dependency specification for {name}: {spec}")


@dataclass
class ModuleExportConfig:
    """Configuration for module exports and visibility."""
    
    public: List[str] = field(default_factory=list)  # Explicitly public symbols
    private: List[str] = field(default_factory=list)  # Explicitly private symbols
    default_visibility: str = "public"  # "public" or "private"
    
    def is_public(self, symbol_name: str) -> bool:
        """Check if a symbol should be publicly visible."""
        if symbol_name in self.private:
            return False
        if symbol_name in self.public:
            return True
        return self.default_visibility == "public"


@dataclass
class PackageManifest:
    """Represents a namel3ss.toml package manifest."""
    
    name: str
    version: str
    description: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # Module organization
    modules: List[str] = field(default_factory=list)  # Explicit module list
    main_module: Optional[str] = None  # Entry point module
    
    # Dependencies
    dependencies: Dict[str, PackageDependency] = field(default_factory=dict)
    dev_dependencies: Dict[str, PackageDependency] = field(default_factory=dict)
    
    # Module configuration
    module_exports: Dict[str, ModuleExportConfig] = field(default_factory=dict)
    
    # Package metadata
    namel3ss_version: Optional[str] = None  # Minimum namel3ss version required
    
    @classmethod
    def from_dict(cls, data: Dict) -> PackageManifest:
        """Create PackageManifest from parsed TOML data."""
        # Required fields
        name = data['package']['name']
        version = data['package']['version']
        
        # Optional package metadata
        pkg_data = data.get('package', {})
        description = pkg_data.get('description')
        authors = pkg_data.get('authors', [])
        homepage = pkg_data.get('homepage')
        repository = pkg_data.get('repository')
        license_name = pkg_data.get('license')
        keywords = pkg_data.get('keywords', [])
        
        # Module configuration
        modules = data.get('modules', [])
        main_module = data.get('main')
        
        # Dependencies
        deps = {}
        deps_data = data.get('dependencies', {})
        if isinstance(deps_data, dict):
            for dep_name, dep_spec in deps_data.items():
                deps[dep_name] = PackageDependency.parse(dep_name, dep_spec)
        
        dev_deps = {}
        dev_deps_data = data.get('dev-dependencies', {})
        if isinstance(dev_deps_data, dict):
            for dep_name, dep_spec in dev_deps_data.items():
                dev_deps[dep_name] = PackageDependency.parse(dep_name, dep_spec)
        
        # Module exports
        module_exports = {}
        for mod_name, export_config in data.get('module-exports', {}).items():
            module_exports[mod_name] = ModuleExportConfig(
                public=export_config.get('public', []),
                private=export_config.get('private', []),
                default_visibility=export_config.get('default', 'public')
            )
        
        namel3ss_version = data.get('namel3ss', {}).get('version')
        
        return cls(
            name=name,
            version=version,
            description=description,
            authors=authors,
            homepage=homepage,
            repository=repository,
            license=license_name,
            keywords=keywords,
            modules=modules,
            main_module=main_module,
            dependencies=deps,
            dev_dependencies=dev_deps,
            module_exports=module_exports,
            namel3ss_version=namel3ss_version
        )


@dataclass
class ModuleReference:
    """Represents a reference to a module within a package or workspace."""
    
    name: str  # Hierarchical name like "analytics.llms"
    file_path: Path  # Absolute path to .ai file
    package_name: Optional[str] = None  # Package this module belongs to
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified name including package."""
        if self.package_name:
            return f"{self.package_name}::{self.name}"
        return self.name
    
    def get_parent_module(self) -> Optional[str]:
        """Get parent module name if this is a submodule."""
        parts = self.name.split('.')
        if len(parts) > 1:
            return '.'.join(parts[:-1])
        return None


@dataclass
class PackageInfo:
    """Information about a discovered package."""
    
    manifest: PackageManifest
    root_path: Path
    modules: Dict[str, ModuleReference] = field(default_factory=dict)
    
    def get_module_path(self, module_name: str) -> Optional[Path]:
        """Get file path for a module in this package."""
        module_ref = self.modules.get(module_name)
        return module_ref.file_path if module_ref else None
    
    def list_module_names(self) -> List[str]:
        """List all module names in this package."""
        return list(self.modules.keys())


@dataclass 
class WorkspaceConfig:
    """Configuration for a namel3ss workspace."""
    
    name: Optional[str] = None
    package_paths: List[str] = field(default_factory=lambda: ["packages", "libs"])
    module_paths: List[str] = field(default_factory=lambda: ["."])
    default_package_source: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> WorkspaceConfig:
        """Create WorkspaceConfig from parsed TOML data."""
        workspace_data = data.get('workspace', {})
        return cls(
            name=workspace_data.get('name'),
            package_paths=workspace_data.get('package_paths', ["packages", "libs"]),
            module_paths=workspace_data.get('module_paths', ["."]),
            default_package_source=workspace_data.get('default_package_source')
        )


# Exception types for package system
class PackageSystemError(Exception):
    """Base exception for package system errors."""
    pass


class PackageNotFoundError(PackageSystemError):
    """Raised when a required package cannot be found."""
    pass


class PackageVersionConflictError(PackageSystemError):
    """Raised when package version constraints conflict."""
    pass


class DependencyCycleError(PackageSystemError):
    """Raised when a circular dependency is detected."""
    pass


class ModuleNotFoundError(PackageSystemError):
    """Raised when a required module cannot be found."""
    pass


__all__ = [
    # Core types
    'PackageIdentifier',
    'VersionConstraint', 
    'PackageDependency',
    'ModuleExportConfig',
    'PackageManifest',
    'ModuleReference',
    'PackageInfo',
    'WorkspaceConfig',
    
    # Exceptions
    'PackageSystemError',
    'PackageNotFoundError',
    'PackageVersionConflictError', 
    'DependencyCycleError',
    'ModuleNotFoundError',
]