"""Utilities for loading .ai source trees into Program ASTs with package support."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Tuple

from namel3ss.ast import App, Module, Program
from namel3ss.parser import Parser
from namel3ss.packages.discovery import PackageDiscovery, load_workspace_config
from namel3ss.packages import PackageInfo, ModuleReference

# Valid file extensions for Namel3ss DSL files
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


def _discover_source_files(root: Path) -> List[Path]:
    """Discover source files with valid extensions (.ai)."""
    if root.is_file():
        return [root] if root.suffix.lower() in VALID_EXTENSIONS else []
    
    # Find all files with valid extensions
    paths = []
    for ext in VALID_EXTENSIONS:
        paths.extend(path for path in root.rglob(f"*{ext}") if path.is_file())
    
    return sorted(paths)


def _iter_source_files(root: Path) -> Iterator[Path]:
    for path in _discover_source_files(root):
        yield path


def _parse_module(source_path: Path) -> Module:
    text = source_path.read_text(encoding="utf-8")
    parser = Parser(text, path=str(source_path))
    module = parser.parse()
    module.path = str(source_path)
    return module


def load_program(root_path: str | PathLike[str]) -> Program:
    """
    Load a program with legacy single-directory behavior.
    
    For backward compatibility. Use load_workspace_program() for 
    full package and module system support.
    """
    root = Path(root_path).resolve()
    project_root = root if root.is_dir() else root.parent
    module_paths: List[Path] = list(_iter_source_files(root))
    if not module_paths:
        raise FileNotFoundError(f"No .ai files found at {root}")
    
    modules: List[Module] = []
    for path in module_paths:
        module = _parse_module(path)
        if not module.name:
            module.name = _derive_module_name(path, project_root)
        modules.append(module)
    return Program(modules=modules)


def load_workspace_program(
    workspace_root: str | PathLike[str], 
    *,
    entry_file: Optional[str | PathLike[str]] = None,
    include_packages: bool = True
) -> Tuple[Program, Dict[str, PackageInfo]]:
    """
    Load a program from a workspace with full package system support.
    
    Args:
        workspace_root: Root directory of the workspace
        entry_file: Optional entry point file
        include_packages: Whether to load packages from package paths
    
    Returns:
        Tuple of (Program, packages_by_name)
    """
    workspace_root = Path(workspace_root).resolve()
    
    # Load workspace configuration
    config = load_workspace_config(workspace_root)
    
    # Discover packages and modules
    discovery = PackageDiscovery(workspace_root, config)
    packages, workspace_modules = discovery.discover_workspace()
    
    # Collect all modules for the program
    all_modules: List[Module] = []
    
    # Add workspace modules
    for module_ref in workspace_modules.values():
        module = _parse_module(module_ref.file_path)
        if not module.name:
            module.name = module_ref.name
        all_modules.append(module)
    
    # Add package modules if requested
    if include_packages:
        for package_info in packages.values():
            for module_ref in package_info.modules.values():
                module = _parse_module(module_ref.file_path)
                # For package modules, always use qualified name
                module.name = module_ref.qualified_name
                all_modules.append(module)
    
    # Create program
    if not all_modules:
        raise FileNotFoundError(f"No .ai modules found in workspace {workspace_root}")
    
    program = Program(modules=all_modules)
    
    return program, packages


def get_workspace_info(workspace_root: str | PathLike[str]) -> Dict[str, object]:
    """
    Get information about a workspace without fully loading it.
    
    Returns:
        Dictionary with workspace information including packages and modules
    """
    workspace_root = Path(workspace_root).resolve()
    config = load_workspace_config(workspace_root)
    discovery = PackageDiscovery(workspace_root, config)
    packages, workspace_modules = discovery.discover_workspace()
    
    return {
        'workspace_root': str(workspace_root),
        'config': config.name if config else None,
        'packages': {name: {
            'name': pkg.manifest.name,
            'version': pkg.manifest.version,
            'description': pkg.manifest.description,
            'module_count': len(pkg.modules),
            'modules': list(pkg.modules.keys())
        } for name, pkg in packages.items()},
        'workspace_modules': {name: {
            'name': ref.name,
            'file_path': str(ref.file_path),
            'qualified_name': ref.qualified_name
        } for name, ref in workspace_modules.items()},
        'total_packages': len(packages),
        'total_modules': len(packages) * 10 + len(workspace_modules)  # Estimate
    }


def find_workspace_root(start_path: str | PathLike[str]) -> Optional[Path]:
    """
    Find workspace root by looking for namel3ss.toml upward from start_path.
    
    Returns None if no workspace root is found.
    """
    current = Path(start_path).resolve()
    
    # If start_path is a file, start from its directory
    if current.is_file():
        current = current.parent
    
    # Walk up the directory tree looking for namel3ss.toml
    while current != current.parent:
        if (current / "namel3ss.toml").exists():
            return current
        current = current.parent
    
    return None


def extract_single_app(program: Program) -> App:
    resolved = resolve_program(program)
    return resolved.app


__all__ = [
    "load_program", 
    "load_workspace_program",
    "get_workspace_info",
    "find_workspace_root", 
    "extract_single_app"
]
