"""
Package discovery and resolution for namel3ss workspace.

This module handles finding packages, resolving dependencies,
and building package/module mappings for the namel3ss system.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from namel3ss.packages import (
    PackageInfo, PackageManifest, ModuleReference, WorkspaceConfig,
    PackageNotFoundError, DependencyCycleError, PackageVersionConflictError,
    VersionConstraint
)

# Valid file extensions for Namel3ss DSL files (avoid circular import)
VALID_EXTENSIONS = {".ai"}

def _derive_module_name(path: Path, root: Path) -> str:
    """Derive module name from file path relative to root."""
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        parts = [path.stem]
    else:
        parts = list(relative.parts)
        if parts:
            parts[-1] = Path(parts[-1]).stem
    name = ".".join(part for part in parts if part)
    return name or path.stem


class PackageDiscovery:
    """Discovers packages and modules in a workspace."""
    
    def __init__(self, workspace_root: Path, config: Optional[WorkspaceConfig] = None):
        self.workspace_root = workspace_root.resolve()
        self.config = config or WorkspaceConfig()
        self._package_cache: Dict[str, PackageInfo] = {}
        self._workspace_modules: Dict[str, ModuleReference] = {}
    
    def discover_workspace(self) -> Tuple[Dict[str, PackageInfo], Dict[str, ModuleReference]]:
        """
        Discover all packages and workspace modules.
        
        Returns:
            Tuple of (packages_by_name, workspace_modules_by_name)
        """
        packages = self._discover_packages()
        workspace_modules = self._discover_workspace_modules()
        
        return packages, workspace_modules
    
    def _discover_packages(self) -> Dict[str, PackageInfo]:
        """Discover all packages in package paths."""
        packages = {}
        
        for package_path_str in self.config.package_paths:
            package_root = self.workspace_root / package_path_str
            if not package_root.exists():
                continue
                
            # Look for package directories with namel3ss.toml
            for item in package_root.iterdir():
                if item.is_dir():
                    manifest_path = item / "namel3ss.toml"
                    if manifest_path.exists():
                        try:
                            package = self._load_package(item, manifest_path)
                            if package.manifest.name in packages:
                                raise PackageVersionConflictError(
                                    f"Duplicate package found: {package.manifest.name} at "
                                    f"{packages[package.manifest.name].root_path} and {package.root_path}"
                                )
                            packages[package.manifest.name] = package
                        except Exception as e:
                            # Log warning but continue discovery
                            print(f"Warning: Failed to load package at {item}: {e}")
        
        return packages
    
    def _discover_workspace_modules(self) -> Dict[str, ModuleReference]:
        """Discover modules directly in the workspace (not in packages)."""
        modules = {}
        
        for module_path_str in self.config.module_paths:
            module_root = self.workspace_root / module_path_str
            if not module_root.exists():
                continue
            
            # Find all .ai files
            for ai_file in self._find_ai_files(module_root):
                # Skip if this file is already part of a package
                if self._is_file_in_package(ai_file):
                    continue
                
                module_name = _derive_module_name(ai_file, module_root)
                if module_name in modules:
                    raise PackageVersionConflictError(
                        f"Duplicate workspace module: {module_name} at "
                        f"{modules[module_name].file_path} and {ai_file}"
                    )
                
                modules[module_name] = ModuleReference(
                    name=module_name,
                    file_path=ai_file,
                    package_name=None
                )
        
        return modules
    
    def _load_package(self, package_dir: Path, manifest_path: Path) -> PackageInfo:
        """Load a single package from its directory."""
        # Parse manifest
        with open(manifest_path, 'rb') as f:
            manifest_data = tomllib.load(f)
        
        manifest = PackageManifest.from_dict(manifest_data)
        
        # Discover modules in package
        modules = {}
        
        if manifest.modules:
            # Use explicit module list from manifest
            for module_name in manifest.modules:
                module_path = self._resolve_module_path(package_dir, module_name)
                if module_path and module_path.exists():
                    modules[module_name] = ModuleReference(
                        name=module_name,
                        file_path=module_path,
                        package_name=manifest.name
                    )
                else:
                    print(f"Warning: Module {module_name} listed in manifest but not found at {module_path}")
        else:
            # Auto-discover modules in package directory
            for ai_file in self._find_ai_files(package_dir):
                module_name = _derive_module_name(ai_file, package_dir)
                modules[module_name] = ModuleReference(
                    name=module_name,
                    file_path=ai_file,
                    package_name=manifest.name
                )
        
        return PackageInfo(
            manifest=manifest,
            root_path=package_dir,
            modules=modules
        )
    
    def _resolve_module_path(self, package_dir: Path, module_name: str) -> Path:
        """Resolve module name to file path within package."""
        # Convert hierarchical name to path: "analytics.llms" -> "analytics/llms.ai"
        parts = module_name.split('.')
        relative_path = Path(*parts[:-1]) / f"{parts[-1]}.ai"
        return package_dir / relative_path
    
    def _find_ai_files(self, root: Path) -> List[Path]:
        """Find all .ai files recursively in a directory."""
        ai_files = []
        for ext in VALID_EXTENSIONS:
            ai_files.extend(root.rglob(f"*{ext}"))
        return sorted(ai_files)
    
    def _is_file_in_package(self, file_path: Path) -> bool:
        """Check if a file is part of a package (has namel3ss.toml in parent hierarchy)."""
        current = file_path.parent
        workspace_root = self.workspace_root
        
        while current != workspace_root and current != current.parent:
            if (current / "namel3ss.toml").exists():
                return True
            current = current.parent
        
        return False


class DependencyResolver:
    """Resolves package dependencies and builds dependency graph."""
    
    def __init__(self, packages: Dict[str, PackageInfo]):
        self.packages = packages
        self._resolution_cache: Dict[str, List[str]] = {}
    
    def resolve_dependencies(self, root_packages: List[str]) -> List[PackageInfo]:
        """
        Resolve all dependencies for the given root packages.
        
        Returns packages in dependency order (dependencies before dependents).
        """
        visited = set()
        visiting = set()
        result = []
        
        for package_name in root_packages:
            self._visit_package(package_name, visited, visiting, result)
        
        # Return PackageInfo objects in dependency order
        return [self.packages[name] for name in result if name in self.packages]
    
    def _visit_package(self, package_name: str, visited: Set[str], visiting: Set[str], result: List[str]):
        """DFS visit for topological sort with cycle detection."""
        if package_name in visiting:
            raise DependencyCycleError(f"Circular dependency detected involving package: {package_name}")
        
        if package_name in visited:
            return
        
        # Check if package exists
        if package_name not in self.packages:
            raise PackageNotFoundError(f"Package not found: {package_name}")
        
        visiting.add(package_name)
        
        # Visit all dependencies
        package = self.packages[package_name]
        for dep_name, dependency in package.manifest.dependencies.items():
            # Find compatible version
            compatible_package = self._find_compatible_package(dep_name, dependency.constraint)
            if not compatible_package:
                raise PackageNotFoundError(
                    f"No compatible version found for dependency {dep_name} {dependency.constraint} "
                    f"required by package {package_name}"
                )
            
            self._visit_package(compatible_package, visited, visiting, result)
        
        visiting.remove(package_name)
        visited.add(package_name)
        result.append(package_name)
    
    def _find_compatible_package(self, package_name: str, constraint: VersionConstraint) -> Optional[str]:
        """Find a package that satisfies the version constraint."""
        # For now, we only support exact package names
        # In a full implementation, this would handle version resolution across multiple packages
        if package_name in self.packages:
            package = self.packages[package_name]
            if constraint.matches(package.manifest.version):
                return package_name
        return None


class ModuleResolver:
    """Resolves module imports across packages and workspace."""
    
    def __init__(self, packages: Dict[str, PackageInfo], workspace_modules: Dict[str, ModuleReference]):
        self.packages = packages
        self.workspace_modules = workspace_modules
        self._module_index = self._build_module_index()
    
    def _build_module_index(self) -> Dict[str, ModuleReference]:
        """Build complete index of all available modules."""
        index = {}
        
        # Add workspace modules
        for name, module_ref in self.workspace_modules.items():
            index[name] = module_ref
        
        # Add package modules  
        for package_name, package_info in self.packages.items():
            for module_name, module_ref in package_info.modules.items():
                # Modules can be referenced by name or qualified name
                index[module_name] = module_ref
                index[module_ref.qualified_name] = module_ref
        
        return index
    
    def resolve_module_import(self, import_name: str, current_module: Optional[str] = None) -> ModuleReference:
        """
        Resolve a module import statement.
        
        Args:
            import_name: The module name being imported
            current_module: The module making the import (for relative resolution)
            
        Returns:
            ModuleReference for the imported module
            
        Raises:
            ModuleNotFoundError: If the module cannot be found
        """
        # Try direct lookup first
        if import_name in self._module_index:
            return self._module_index[import_name]
        
        # Try relative import resolution if current_module is provided
        if current_module and not import_name.startswith('.'):
            # Try package-relative import
            current_ref = self._module_index.get(current_module)
            if current_ref and current_ref.package_name:
                qualified_name = f"{current_ref.package_name}::{import_name}"
                if qualified_name in self._module_index:
                    return self._module_index[qualified_name]
        
        # Handle relative imports (starting with '.')
        if import_name.startswith('.') and current_module:
            resolved_name = self._resolve_relative_import(import_name, current_module)
            if resolved_name in self._module_index:
                return self._module_index[resolved_name]
        
        raise ModuleNotFoundError(f"Module not found: {import_name}")
    
    def _resolve_relative_import(self, relative_name: str, current_module: str) -> str:
        """Resolve relative import like '.utils' or '..shared.utils'."""
        # Count leading dots
        dots = 0
        for char in relative_name:
            if char == '.':
                dots += 1
            else:
                break
        
        # Get relative part
        relative_part = relative_name[dots:]
        
        # Split current module into parts
        current_parts = current_module.split('.')
        
        # Calculate parent module
        if dots > len(current_parts):
            raise ModuleNotFoundError(f"Relative import {relative_name} goes beyond package root")
        
        parent_parts = current_parts[:-dots] if dots > 0 else current_parts
        
        # Combine with relative part
        if relative_part:
            return '.'.join(parent_parts + [relative_part])
        else:
            return '.'.join(parent_parts)
    
    def list_available_modules(self, package_name: Optional[str] = None) -> List[str]:
        """List all available modules, optionally filtered by package."""
        if package_name is None:
            return list(self._module_index.keys())
        
        modules = []
        for module_ref in self._module_index.values():
            if module_ref.package_name == package_name:
                modules.append(module_ref.name)
        
        return sorted(modules)


def load_workspace_config(workspace_root: Path) -> WorkspaceConfig:
    """Load workspace configuration from namel3ss.toml in workspace root."""
    config_path = workspace_root / "namel3ss.toml"
    
    if config_path.exists():
        try:
            with open(config_path, 'rb') as f:
                data = tomllib.load(f)
            return WorkspaceConfig.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to parse workspace config at {config_path}: {e}")
    
    return WorkspaceConfig()


__all__ = [
    'PackageDiscovery',
    'DependencyResolver', 
    'ModuleResolver',
    'load_workspace_config',
]