"""
Multi-File Module System for Namel3ss

This module implements a robust module system that allows larger applications
to be split across multiple .ai files. Features include:

- Module declarations: module "app.main"
- Import statements: import "app.shared.types"
- Cross-module symbol resolution
- Circular dependency detection
- Project-local module resolution (no complex package management)

This elevates Namel3ss from a single-file toy to a real development platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import os

from namel3ss.ast import Module, App
from namel3ss.ast.modules import Import, ModuleDeclaration
from namel3ss.lang.parser import parse_module, N3SyntaxError
from namel3ss.lang.parser.errors import N3Error, N3SemanticError


# ==================== Module Information ====================


@dataclass
class ModuleInfo:
    """Information about a parsed module."""
    name: str
    path: str
    module_ast: Module
    imports: List[str] = field(default_factory=list)
    exports: Dict[str, any] = field(default_factory=dict)


@dataclass
class ResolvedImport:
    """A resolved import statement."""
    import_name: str  # e.g. "app.shared.types"
    resolved_path: str  # e.g. "src/app/shared/types.ai"
    module_info: Optional[ModuleInfo] = None


# ==================== Module Resolver ====================


class ModuleSystemError(N3SemanticError):
    """Raised when module system operations fail."""
    pass


class CircularDependencyError(ModuleSystemError):
    """Raised when circular dependencies are detected."""
    pass


class ModuleResolver:
    """
    Resolves module imports and manages multi-file project structure.
    
    The resolver:
    - Maps module names to file paths (e.g. "app.shared" -> "app/shared.ai")
    - Loads and parses modules
    - Detects circular dependencies
    - Builds a dependency graph
    - Validates cross-module references
    """
    
    def __init__(self, project_root: Optional[str] = None, search_paths: Optional[List[str]] = None):
        """
        Initialize the module resolver.
        
        Args:
            project_root: Root directory for module resolution (defaults to current directory)
            search_paths: Additional directories to search for modules
        """
        self.project_root = Path(project_root or os.getcwd())
        self.search_paths = [self.project_root]
        if search_paths:
            self.search_paths.extend([Path(p) for p in search_paths])
        
        self.loaded_modules: Dict[str, ModuleInfo] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.loading_stack: List[str] = []  # For circular dependency detection
    
    def resolve_module_path(self, module_name: str) -> Optional[str]:
        """
        Resolve a module name to a file path.
        
        Converts module name like "app.shared.types" to:
        - app/shared/types.ai
        - app/shared/types.n3
        
        Args:
            module_name: Dotted module name
        
        Returns:
            Absolute path to module file, or None if not found
        """
        # Convert dotted name to path: "app.shared.types" -> "app/shared/types"
        relative_path = module_name.replace(".", os.sep)
        
        # Try with different extensions
        for ext in [".ai", ".n3"]:
            for search_path in self.search_paths:
                candidate = search_path / f"{relative_path}{ext}"
                if candidate.exists():
                    return str(candidate.resolve())
        
        return None
    
    def load_module(self, module_name: str, source_path: Optional[str] = None) -> ModuleInfo:
        """
        Load and parse a module by name.
        
        Args:
            module_name: Dotted module name to load
            source_path: Optional path to the file requesting this module (for error messages)
        
        Returns:
            ModuleInfo for the loaded module
        
        Raises:
            ModuleSystemError: If module cannot be found or loaded
            CircularDependencyError: If circular dependency is detected
        """
        # Check if already loaded
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        
        # Check for circular dependencies
        if module_name in self.loading_stack:
            cycle = " -> ".join(self.loading_stack + [module_name])
            raise CircularDependencyError(
                f"Circular dependency detected: {cycle}",
                path=source_path,
                code="CIRCULAR_DEPENDENCY"
            )
        
        # Resolve module path
        module_path = self.resolve_module_path(module_name)
        if module_path is None:
            raise ModuleSystemError(
                f"Cannot import module '{module_name}' (file not found)",
                path=source_path,
                code="MODULE_NOT_FOUND"
            )
        
        # Load and parse module
        self.loading_stack.append(module_name)
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            module_ast = parse_module(source, path=module_path, module_name=module_name)
            
            # Extract imports from the module
            imports = []
            if hasattr(module_ast, 'imports'):
                for import_stmt in module_ast.imports:
                    if hasattr(import_stmt, 'module'):
                        imports.append(import_stmt.module)
            
            # Create module info
            module_info = ModuleInfo(
                name=module_name,
                path=module_path,
                module_ast=module_ast,
                imports=imports
            )
            
            # Extract exports
            module_info.exports = self._extract_exports(module_ast)
            
            # Update dependency graph (but don't add to loaded_modules yet)
            self.dependency_graph[module_name] = set(imports)
            
            # Recursively load imports (while still on the stack to detect cycles)
            for import_name in imports:
                self.load_module(import_name, source_path=module_path)
            
            # Only after all imports succeed, mark this module as loaded
            self.loaded_modules[module_name] = module_info
            
            # Only pop from stack after all imports are loaded
            self.loading_stack.pop()
            
            return module_info
        
        except N3SyntaxError as e:
            if module_name in self.loading_stack:
                self.loading_stack.pop()
            raise ModuleSystemError(
                f"Syntax error in module '{module_name}': {e.message}",
                path=module_path,
                line=e.line,
                column=e.column,
                code="MODULE_SYNTAX_ERROR"
            )
        
        except CircularDependencyError:
            # Don't wrap circular dependency errors, just re-raise
            if module_name in self.loading_stack:
                self.loading_stack.pop()
            raise
        
        except Exception as e:
            if module_name in self.loading_stack:
                self.loading_stack.pop()
            raise ModuleSystemError(
                f"Failed to load module '{module_name}': {str(e)}",
                path=module_path,
                code="MODULE_LOAD_ERROR"
            )
    
    @property
    def dependency_order(self) -> List[str]:
        """
        Get modules in dependency order (topological sort).
        
        Modules that are depended upon come before modules that depend on them.
        For example, if A imports B, then B comes before A.
        
        Returns:
            List of module names in dependency order
        """
        # Calculate in-degree (number of modules depending on each module)
        in_degree = {mod: 0 for mod in self.loaded_modules}
        
        # Count dependencies: if A depends on B, increase B's in-degree
        for module, deps in self.dependency_graph.items():
            if module not in in_degree:
                in_degree[module] = 0
            for dep in deps:
                if dep not in in_degree:
                    in_degree[dep] = 0
        
        # Count how many modules each module depends on
        for module, deps in self.dependency_graph.items():
            in_degree[module] = len([d for d in deps if d in self.loaded_modules])
        
        # Start with modules that have no dependencies
        queue = [mod for mod, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)
            
            # For each module that depends on current, decrease its in-degree
            for module, deps in self.dependency_graph.items():
                if current in deps and module in in_degree and module not in result:
                    in_degree[module] -= 1
                    if in_degree[module] == 0:
                        queue.append(module)
        
        return result
    
    def resolve_import(self, import_stmt: Import, source_module: str) -> ResolvedImport:
        """
        Resolve an import statement.
        
        Args:
            import_stmt: Import AST node
            source_module: Name of the module containing this import
        
        Returns:
            ResolvedImport with module information
        """
        import_name = import_stmt.module_path
        
        try:
            module_info = self.load_module(import_name, source_path=source_module)
            module_path = self.resolve_module_path(import_name)
            
            return ResolvedImport(
                import_name=import_name,
                resolved_path=module_path or "",
                module_info=module_info
            )
        
        except ModuleSystemError as e:
            raise ModuleSystemError(
                f"Failed to resolve import '{import_name}' in module '{source_module}': {e.message}",
                path=e.path,
                line=e.line,
                column=e.column,
                code=e.code
            )
    
    def check_circular_dependencies(self) -> List[List[str]]:
        """
        Check for circular dependencies in the dependency graph.
        
        Returns:
            List of cycles found (each cycle is a list of module names)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(module: str, path: List[str]) -> None:
            visited.add(module)
            rec_stack.add(module)
            path.append(module)
            
            for dep in self.dependency_graph.get(module, set()):
                if dep not in visited:
                    dfs(dep, path[:])
                elif dep in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    cycles.append(cycle)
            
            rec_stack.remove(module)
        
        for module in self.dependency_graph:
            if module not in visited:
                dfs(module, [])
        
        return cycles
    
    def get_import_order(self) -> List[str]:
        """
        Get modules in topological order (dependencies first).
        
        Returns:
            List of module names in import order
        
        Raises:
            CircularDependencyError: If circular dependencies exist
        """
        cycles = self.check_circular_dependencies()
        if cycles:
            cycle_str = ", ".join(" -> ".join(cycle) for cycle in cycles)
            raise CircularDependencyError(
                f"Circular dependencies detected: {cycle_str}",
                code="CIRCULAR_DEPENDENCY"
            )
        
        # Topological sort
        result = []
        visited = set()
        
        def visit(module: str) -> None:
            if module in visited:
                return
            visited.add(module)
            
            for dep in self.dependency_graph.get(module, set()):
                visit(dep)
            
            result.append(module)
        
        for module in self.dependency_graph:
            visit(module)
        
        return result
    
    def validate_cross_module_references(self) -> List[N3Error]:
        """
        Validate that all cross-module references are valid.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        for module_name, module_info in self.loaded_modules.items():
            # Check each import
            for import_name in module_info.imports:
                if import_name not in self.loaded_modules:
                    errors.append(N3SemanticError(
                        f"Module '{module_name}' imports undefined module '{import_name}'",
                        path=module_info.path,
                        code="UNDEFINED_MODULE"
                    ))
                    continue
                
                # TODO: Validate that imported symbols are actually exported
                # This would require analyzing the AST to find symbol references
        
        return errors
    
    def _extract_exports(self, module_ast: Module) -> Dict[str, any]:
        """
        Extract exported symbols from a module.
        
        Args:
            module_ast: Parsed module AST
        
        Returns:
            Dictionary of exported symbols
        """
        exports = {}
        
        # Extract declarations from module body
        for decl in module_ast.body:
            if isinstance(decl, App):
                # Export app and its nested declarations
                exports["app"] = decl
                
                # Export datasets
                for dataset in decl.datasets:
                    exports[dataset.name] = dataset
                
                # Export prompts
                for prompt in decl.prompts:
                    exports[prompt.name] = prompt
                
                # Export chains
                for chain in decl.chains:
                    exports[chain.name] = chain
                
                # Export pages
                for page in decl.pages:
                    exports[page.name] = page
                
                # Export AI models
                for model in decl.ai_models:
                    exports[model.name] = model
        
        return exports
    
    def get_symbol(self, module_name: str, symbol_name: str) -> Optional[any]:
        """
        Get a symbol from a module's exports.
        
        Args:
            module_name: Module to get symbol from
            symbol_name: Name of the symbol
        
        Returns:
            Symbol value, or None if not found
        """
        module_info = self.loaded_modules.get(module_name)
        if module_info is None:
            return None
        
        return module_info.exports.get(symbol_name)


# ==================== Module System Builder ====================


class ModuleSystemBuilder:
    """
    Builds a complete multi-module project.
    
    Orchestrates:
    - Module loading
    - Dependency resolution
    - Cross-module validation
    - Combined AST generation
    """
    
    def __init__(self, project_root: Optional[str] = None):
        self.resolver = ModuleResolver(project_root=project_root)
        self.errors: List[N3Error] = []
    
    def build_project(self, entry_module: str) -> Tuple[List[ModuleInfo], List[N3Error]]:
        """
        Build a complete project starting from an entry module.
        
        Args:
            entry_module: Name of the entry point module
        
        Returns:
            Tuple of (loaded modules in order, errors)
        """
        self.errors = []
        
        try:
            # Load entry module and all its dependencies
            self.resolver.load_module(entry_module)
            
            # Get modules in dependency order
            module_order = self.resolver.dependency_order
            
            # Return modules in order
            modules = [self.resolver.loaded_modules[name] for name in module_order]
            return modules, self.errors
        
        except (ModuleSystemError, CircularDependencyError) as e:
            self.errors.append(e)
            return [], self.errors


# ==================== Public API ====================


def load_multi_module_project(entry_module: str, project_root: Optional[str] = None) -> Tuple[List[ModuleInfo], List[N3Error]]:
    """
    Load a multi-module Namel3ss project.
    
    Args:
        entry_module: Name of the entry point module (e.g. "app.main")
        project_root: Root directory of the project (defaults to current directory)
    
    Returns:
        Tuple of (loaded modules in dependency order, errors)
    
    Example:
        ```python
        modules, errors = load_multi_module_project("app.main", project_root="./my_project")
        
        if errors:
            for error in errors:
                print(f"Error: {error.message}")
        else:
            for module in modules:
                print(f"Loaded module: {module.name} from {module.path}")
        ```
    """
    builder = ModuleSystemBuilder(project_root=project_root)
    return builder.build_project(entry_module)


def resolve_module(module_name: str, project_root: Optional[str] = None) -> Optional[str]:
    """
    Resolve a module name to its file path.
    
    Args:
        module_name: Dotted module name (e.g. "app.shared.types")
        project_root: Root directory for resolution
    
    Returns:
        Absolute path to module file, or None if not found
    
    Example:
        ```python
        path = resolve_module("app.shared.types", project_root="./my_project")
        if path:
            print(f"Module found at: {path}")
        ```
    """
    resolver = ModuleResolver(project_root=project_root)
    return resolver.resolve_module_path(module_name)


__all__ = [
    # Core classes
    "ModuleInfo",
    "ResolvedImport",
    "ModuleResolver",
    "ModuleSystemBuilder",
    
    # Errors
    "ModuleSystemError",
    "CircularDependencyError",
    
    # Public API
    "load_multi_module_project",
    "resolve_module",
]
