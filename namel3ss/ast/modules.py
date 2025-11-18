"""AST nodes for module declarations and imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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


@dataclass
class ModuleSpec:
    """Parsed representation of a module file."""

    name: Optional[str]
    imports: List[Import] = field(default_factory=list)


__all__ = ["ImportedName", "Import", "ModuleSpec"]
