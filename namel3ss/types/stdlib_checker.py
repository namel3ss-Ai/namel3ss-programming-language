"""Enhanced type checker with standard library integration."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Any, TYPE_CHECKING

from namel3ss.ast import App, Module
from namel3ss.types.checker_original_backup import AppTypeChecker as BaseAppTypeChecker
from namel3ss.stdlib.typing_enhanced import (
    get_stdlib_checker,
    StandardLibraryTypeChecker as StdLibValidator, 
    StdLibImport
)
from namel3ss.errors import N3TypeError

if TYPE_CHECKING:
    from namel3ss.resolver import ResolvedProgram, ResolvedImport


class EnhancedAppTypeChecker(BaseAppTypeChecker):
    """Type checker with standard library validation support."""
    
    def __init__(self, *, path: Optional[str] = None):
        super().__init__(path=path)
        self.stdlib_checker = get_stdlib_checker()
        self.stdlib_imports: Dict[str, StdLibImport] = {}
        self.imported_stdlib_symbols: Dict[str, str] = {}  # symbol_name -> module_name
    
    def check_program(self, program) -> None:
        """Check a resolved program including stdlib imports."""
        # First collect all stdlib imports
        self._collect_stdlib_imports(program)
        
        # Then check the app as usual
        self.check_app(program.app, path=program.root.module.path)
        
        # Finally validate stdlib usage
        self._validate_stdlib_usage(program.app)
    
    def _collect_stdlib_imports(self, program) -> None:
        """Collect all stdlib imports from the program."""
        for module in program.modules.values():
            for import_stmt in module.imports:
                if import_stmt.stdlib_import:
                    stdlib_import = import_stmt.stdlib_import
                    self.stdlib_imports[import_stmt.target_module] = stdlib_import
                    
                    # Record imported symbols
                    for symbol_name, symbol in import_stmt.stdlib_symbols.items():
                        self.imported_stdlib_symbols[symbol_name] = import_stmt.target_module
    
    def _validate_stdlib_usage(self, app: App) -> None:
        """Validate standard library component usage throughout the app."""
        errors = []
        
        # Validate LLM configurations
        for llm in app.llms:
            if hasattr(llm, 'config') and llm.config:
                config_errors = self.stdlib_checker.validate_llm_config(
                    llm.config, 
                    context=f"LLM '{llm.name}'"
                )
                errors.extend(config_errors)
        
        # Validate tool configurations
        for tool in app.tools:
            if hasattr(tool, 'config') and tool.config:
                config_errors = self.stdlib_checker.validate_tool_config(
                    tool.config,
                    context=f"Tool '{tool.name}'"
                )
                errors.extend(config_errors)
        
        # Validate memory configurations (if they exist in memories)
        for memory in app.memories:
            if hasattr(memory, 'config') and memory.config:
                config_errors = self.stdlib_checker.validate_memory_config(
                    memory.config,
                    context=f"Memory '{memory.name}'"
                )
                errors.extend(config_errors)
        
        # Check that stdlib components are properly imported
        for symbol_name in self._find_stdlib_symbol_usage(app):
            if symbol_name not in self.imported_stdlib_symbols:
                stdlib_symbol = self.stdlib_checker.get_symbol(symbol_name)
                if stdlib_symbol:
                    errors.append(f"Standard library component '{symbol_name}' used but not imported")
        
        if errors:
            raise N3TypeError(f"Standard library validation errors:\n" + "\n".join(f"  - {err}" for err in errors))
    
    def _find_stdlib_symbol_usage(self, app: App) -> List[str]:
        """Find all stdlib symbols used in the app configuration."""
        used_symbols = []
        
        # Check LLM configs for stdlib field names
        for llm in app.llms:
            if hasattr(llm, 'config') and llm.config:
                for field_name in llm.config.keys():
                    if self.stdlib_checker.get_symbol(field_name):
                        used_symbols.append(field_name)
        
        # Check tool configs for stdlib categories
        for tool in app.tools:
            if hasattr(tool, 'config') and tool.config:
                category = tool.config.get('category')
                if category and self.stdlib_checker.get_symbol(category):
                    used_symbols.append(category)
        
        # Check memory configs for stdlib policies
        for memory in app.memories:
            if hasattr(memory, 'config') and memory.config:
                policy = memory.config.get('policy')
                if policy and self.stdlib_checker.get_symbol(policy):
                    used_symbols.append(policy)
        
        return used_symbols
    
    def check_app(self, app: App, *, path: Optional[str] = None) -> None:
        """Override to add stdlib-aware validation."""
        # Call parent implementation first
        super().check_app(app, path=path)
        
        # Don't validate stdlib usage here if called from check_program
        # (it will be done separately with full import context)
        if not hasattr(self, '_in_program_check'):
            self._validate_stdlib_usage(app)


def check_app_with_stdlib(app: App, *, path: Optional[str] = None) -> None:
    """Check an app with standard library validation."""
    checker = EnhancedAppTypeChecker(path=path)
    checker.check_app(app, path=path)


def check_program_with_stdlib(program) -> None:
    """Check a resolved program with standard library validation."""
    checker = EnhancedAppTypeChecker(path=program.root.module.path)
    checker._in_program_check = True  # Flag to avoid duplicate validation
    checker.check_program(program)


def check_module_with_stdlib(module: Module) -> None:
    """Check a module with standard library validation."""
    checker = EnhancedAppTypeChecker()
    checker.check_module(module)


# Export functions for backward compatibility and new stdlib support
__all__ = [
    "EnhancedAppTypeChecker",
    "check_app_with_stdlib", 
    "check_program_with_stdlib",
    "check_module_with_stdlib"
]