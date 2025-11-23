"""AST nodes for module declarations and imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ImportedName:
    """Symbol imported from another module."""

    name: str
    alias: Optional[str] = None


@dataclass
class Import:
    """Represents an import statement inside a module."""

    module: str
    names: Optional[List[ImportedName]] = None
    alias: Optional[str] = None
    is_relative: bool = False  # For relative imports like '.utils'


@dataclass
class UseStatement:
    """Represents a 'use' statement for importing modules or symbols."""
    
    module_path: str  # e.g., "analytics.llms" or "my_org.prompts::customer_support"
    imported_items: Optional[List[ImportedName]] = None  # Specific items to import
    alias: Optional[str] = None  # Alias for the entire module
    is_wildcard: bool = False  # For 'use module::*'
    
    @property
    def package_name(self) -> Optional[str]:
        """Extract package name from module path if present."""
        if '::' in self.module_path:
            return self.module_path.split('::', 1)[0]
        return None
    
    @property
    def module_name(self) -> str:
        """Extract module name from module path."""
        if '::' in self.module_path:
            return self.module_path.split('::', 1)[1]
        return self.module_path


@dataclass
class ModuleDeclaration:
    """Explicit module declaration with hierarchical name."""
    
    name: str  # Hierarchical name like "analytics.llms"
    exports: Optional[List[str]] = None  # Explicit export list
    
    @property
    def package_parts(self) -> List[str]:
        """Get module name parts for hierarchical organization."""
        return self.name.split('.')
    
    @property
    def parent_module(self) -> Optional[str]:
        """Get parent module name if this is a submodule."""
        parts = self.package_parts
        if len(parts) > 1:
            return '.'.join(parts[:-1])
        return None


@dataclass
class PackageDeclaration:
    """Declaration of package-level information in a module."""
    
    name: str
    version: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ExportStatement:
    """Explicit export statement for controlling module visibility."""
    
    symbols: List[str]  # Symbol names to export
    is_public: bool = True  # Whether to export publicly or privately


@dataclass
class ModuleSpec:
    """Parsed representation of a module file."""

    name: Optional[str]
    imports: List[Union[Import, UseStatement]] = field(default_factory=list)
    module_declaration: Optional[ModuleDeclaration] = None
    package_declaration: Optional[PackageDeclaration] = None
    exports: List[ExportStatement] = field(default_factory=list)


__all__ = [
    "ImportedName", 
    "Import", 
    "UseStatement",
    "ModuleDeclaration",
    "PackageDeclaration", 
    "ExportStatement",
    "ModuleSpec"
]
