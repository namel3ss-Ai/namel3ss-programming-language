"""Standard library integration with type system and module resolution."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

from namel3ss.stdlib.registry import StandardLibraryRegistry
from namel3ss.stdlib.memory.policies import MemoryPolicy, get_memory_policy_spec
from namel3ss.stdlib.memory import list_memory_policies, validate_memory_config
from namel3ss.stdlib.llm.config import LLMConfigField, get_llm_config_spec
from namel3ss.stdlib.llm import list_llm_config_fields, validate_llm_config
from namel3ss.stdlib.tools import ToolCategory, get_tool_spec
from namel3ss.stdlib.tools import list_tool_categories, validate_tool_config
from namel3ss.stdlib.memory.policies import MemoryPolicy
from namel3ss.stdlib.llm.config import LLMConfigField
from namel3ss.stdlib.tools import ToolCategory
from namel3ss.errors import N3TypeError, N3ResolutionError


class ComponentType(Enum):
    """Standard library component types."""
    
    MEMORY = "memory"
    LLM = "llm" 
    TOOL = "tool"


class StdLibResolutionError(N3ResolutionError):
    """Raised when standard library module resolution fails."""


class StdLibTypeError(N3TypeError):
    """Raised when standard library component usage has type errors."""


@dataclass
class StdLibImport:
    """Represents an import from the standard library."""
    
    module: str  # e.g., "stdlib.memory", "stdlib.llm", "stdlib.tools"
    component: Optional[str] = None  # e.g., "conversation_window", "temperature", "http"
    alias: Optional[str] = None
    
    @property
    def component_type(self) -> ComponentType:
        """Extract component type from module path."""
        if self.module == "stdlib.memory":
            return ComponentType.MEMORY
        elif self.module == "stdlib.llm":
            return ComponentType.LLM
        elif self.module == "stdlib.tools":
            return ComponentType.TOOL
        else:
            raise StdLibResolutionError(f"Unknown stdlib module: {self.module}")


@dataclass 
class StdLibSymbol:
    """Represents a symbol from the standard library."""
    
    name: str
    component_type: ComponentType
    value: Any  # MemoryPolicySpec, LLMConfigSpec, or ToolSpec
    description: str


class StdLibTypeRegistry:
    """Registry for standard library types and validation."""
    
    def __init__(self):
        self.stdlib_registry = StandardLibraryRegistry()
        self._symbol_cache: Dict[str, StdLibSymbol] = {}
        self._build_symbol_cache()
    
    def _build_symbol_cache(self) -> None:
        """Build cache of all standard library symbols."""
        # Memory policies
        for policy_name in list_memory_policies():
            try:
                policy_spec = get_memory_policy(policy_name)
                self._symbol_cache[policy_name] = StdLibSymbol(
                    name=policy_name,
                    component_type=ComponentType.MEMORY,
                    value=policy_spec,
                    description=policy_spec.description
                )
            except (ValueError, AttributeError):
                # Skip invalid policies
                pass
        
        # LLM config fields
        for field_name in list_llm_config_fields():
            try:
                field_spec = get_llm_config_field(field_name)
                self._symbol_cache[field_name] = StdLibSymbol(
                    name=field_name,
                    component_type=ComponentType.LLM,
                    value=field_spec,
                    description=field_spec.description
                )
            except (ValueError, AttributeError):
                # Skip invalid fields
                pass
        
        # Tool categories
        for tool_name in list_tool_categories():
            try:
                tool_spec = get_tool_spec(tool_name)
                self._symbol_cache[tool_name] = StdLibSymbol(
                    name=tool_name,
                    component_type=ComponentType.TOOL,
                    value=tool_spec,
                    description=tool_spec.description
                )
            except (ValueError, AttributeError):
                # Skip tools without standard specifications
                pass
    
    def resolve_stdlib_import(self, import_stmt: StdLibImport) -> Dict[str, StdLibSymbol]:
        """Resolve a standard library import statement."""
        if import_stmt.component:
            # Specific component import: import stdlib.memory: conversation_window
            return self._resolve_specific_component(import_stmt)
        else:
            # Module import: import stdlib.memory as mem
            return self._resolve_module_import(import_stmt)
    
    def _resolve_specific_component(self, import_stmt: StdLibImport) -> Dict[str, StdLibSymbol]:
        """Resolve import of a specific stdlib component."""
        component_name = import_stmt.component
        if not component_name:
            raise StdLibResolutionError("Component name is required")
        
        if component_name not in self._symbol_cache:
            raise StdLibResolutionError(f"Unknown stdlib component: {component_name}")
        
        symbol = self._symbol_cache[component_name]
        
        # Validate component type matches module
        expected_type = import_stmt.component_type
        if symbol.component_type != expected_type:
            raise StdLibResolutionError(
                f"Component '{component_name}' is a {symbol.component_type.value}, "
                f"not a {expected_type.value} (wrong module)"
            )
        
        # Use alias if provided
        symbol_name = import_stmt.alias or component_name
        return {symbol_name: symbol}
    
    def _resolve_module_import(self, import_stmt: StdLibImport) -> Dict[str, StdLibSymbol]:
        """Resolve import of an entire stdlib module."""
        component_type = import_stmt.component_type
        
        # Get all symbols of this type
        symbols = {}
        for name, symbol in self._symbol_cache.items():
            if symbol.component_type == component_type:
                symbols[name] = symbol
        
        if not symbols:
            raise StdLibResolutionError(f"No components found in {import_stmt.module}")
        
        return symbols
    
    def validate_stdlib_usage(self, component_type: str, config: Dict[str, Any]) -> List[str]:
        """Validate usage of stdlib components with type checking."""
        try:
            return validate_stdlib_config(component_type, config)
        except Exception as e:
            return [str(e)]
    
    def get_stdlib_symbol(self, name: str) -> Optional[StdLibSymbol]:
        """Get stdlib symbol by name."""
        return self._symbol_cache.get(name)
    
    def list_stdlib_symbols(self, component_type: Optional[ComponentType] = None) -> List[StdLibSymbol]:
        """List all stdlib symbols, optionally filtered by type."""
        if component_type:
            return [
                symbol for symbol in self._symbol_cache.values()
                if symbol.component_type == component_type
            ]
        return list(self._symbol_cache.values())


class StdLibValidator:
    """Validates standard library component configurations at compile time."""
    
    def __init__(self, type_registry: StdLibTypeRegistry):
        self.type_registry = type_registry
    
    def validate_memory_config(self, config: Dict[str, Any], context: str = "") -> List[str]:
        """Validate memory policy configuration."""
        errors = []
        
        if "policy" not in config:
            errors.append(f"{context}: Memory config missing required 'policy' field")
            return errors
        
        policy_name = config["policy"]
        if not isinstance(policy_name, str):
            errors.append(f"{context}: Memory policy must be a string, got {type(policy_name).__name__}")
            return errors
        
        # Check if policy exists in stdlib
        symbol = self.type_registry.get_stdlib_symbol(policy_name)
        if not symbol or symbol.component_type != ComponentType.MEMORY:
            errors.append(f"{context}: Unknown memory policy '{policy_name}'")
            return errors
        
        # Validate specific policy constraints
        policy_spec = symbol.value
        if hasattr(policy_spec, 'validate_config'):
            try:
                policy_errors = policy_spec.validate_config(config)
                errors.extend([f"{context}: {err}" for err in policy_errors])
            except Exception as e:
                errors.append(f"{context}: Policy validation failed: {e}")
        
        return errors
    
    def validate_llm_config(self, config: Dict[str, Any], context: str = "") -> List[str]:
        """Validate LLM configuration against stdlib standards."""
        errors = []
        
        for field_name, field_value in config.items():
            symbol = self.type_registry.get_stdlib_symbol(field_name)
            
            if symbol and symbol.component_type == ComponentType.LLM:
                # This is a standard LLM config field - validate it
                field_spec = symbol.value
                
                # Type validation
                expected_type = field_spec.value_type
                if expected_type == "float":
                    if not isinstance(field_value, (int, float)):
                        errors.append(f"{context}: LLM field '{field_name}' must be numeric, got {type(field_value).__name__}")
                        continue
                    
                    # Range validation
                    if hasattr(field_spec, 'min_value') and field_value < field_spec.min_value:
                        errors.append(f"{context}: LLM field '{field_name}' ({field_value}) below minimum ({field_spec.min_value})")
                    if hasattr(field_spec, 'max_value') and field_value > field_spec.max_value:
                        errors.append(f"{context}: LLM field '{field_name}' ({field_value}) above maximum ({field_spec.max_value})")
                
                elif expected_type == "int":
                    if not isinstance(field_value, int):
                        errors.append(f"{context}: LLM field '{field_name}' must be integer, got {type(field_value).__name__}")
                        continue
                    
                    # Range validation for integers
                    if hasattr(field_spec, 'min_value') and field_value < field_spec.min_value:
                        errors.append(f"{context}: LLM field '{field_name}' ({field_value}) below minimum ({field_spec.min_value})")
                    if hasattr(field_spec, 'max_value') and field_value > field_spec.max_value:
                        errors.append(f"{context}: LLM field '{field_name}' ({field_value}) above maximum ({field_spec.max_value})")
                
                elif expected_type == "string":
                    if not isinstance(field_value, str):
                        errors.append(f"{context}: LLM field '{field_name}' must be string, got {type(field_value).__name__}")
                
                elif expected_type == "bool":
                    if not isinstance(field_value, bool):
                        errors.append(f"{context}: LLM field '{field_name}' must be boolean, got {type(field_value).__name__}")
        
        return errors
    
    def validate_tool_config(self, config: Dict[str, Any], context: str = "") -> List[str]:
        """Validate tool configuration against stdlib specifications."""
        errors = []
        
        if "category" not in config:
            errors.append(f"{context}: Tool config missing required 'category' field")
            return errors
        
        category = config["category"]
        if not isinstance(category, str):
            errors.append(f"{context}: Tool category must be string, got {type(category).__name__}")
            return errors
        
        # Check if category exists in stdlib
        symbol = self.type_registry.get_stdlib_symbol(category)
        if not symbol or symbol.component_type != ComponentType.TOOL:
            # Allow non-stdlib tool categories but skip validation
            return errors
        
        # Use stdlib validation
        stdlib_errors = self.type_registry.validate_stdlib_usage("tool", config)
        errors.extend([f"{context}: {err}" for err in stdlib_errors])
        
        return errors
    
    def validate_stdlib_component_usage(self, component_name: str, usage_context: str) -> List[str]:
        """Validate that a stdlib component is being used correctly."""
        errors = []
        
        symbol = self.type_registry.get_stdlib_symbol(component_name)
        if not symbol:
            errors.append(f"Unknown stdlib component '{component_name}' used in {usage_context}")
            return errors
        
        # Additional context-specific validation could go here
        # For example, checking that memory policies are only used in memory contexts
        
        return errors
    
    def check_stdlib_compatibility(self, imports: List[StdLibImport], usage: Dict[str, Any]) -> List[str]:
        """Check that stdlib imports are compatible with their usage."""
        errors = []
        
        # Build set of imported stdlib symbols
        imported_symbols: Set[str] = set()
        for import_stmt in imports:
            try:
                resolved = self.type_registry.resolve_stdlib_import(import_stmt)
                imported_symbols.update(resolved.keys())
            except StdLibResolutionError as e:
                errors.append(str(e))
        
        # Check that used stdlib components were imported
        for component_name, _ in usage.items():
            if self.type_registry.get_stdlib_symbol(component_name):
                # This is a stdlib component
                if component_name not in imported_symbols:
                    errors.append(f"Stdlib component '{component_name}' used but not imported")
        
        return errors


# Global registry instance for type checking
_stdlib_type_registry: Optional[StdLibTypeRegistry] = None


def get_stdlib_type_registry() -> StdLibTypeRegistry:
    """Get the global stdlib type registry."""
    global _stdlib_type_registry
    if _stdlib_type_registry is None:
        _stdlib_type_registry = StdLibTypeRegistry()
    return _stdlib_type_registry


def get_memory_policy(policy_name: str):
    """Get memory policy specification by name."""
    if isinstance(policy_name, MemoryPolicy):
        return get_memory_policy_spec(policy_name)
    else:
        # Convert string to enum
        policy_enum = MemoryPolicy(policy_name)
        return get_memory_policy_spec(policy_enum)


def get_llm_config_field(field_name: str):
    """Get LLM config field specification by name."""
    if isinstance(field_name, LLMConfigField):
        return get_llm_config_spec(field_name)
    else:
        # Convert string to enum
        field_enum = LLMConfigField(field_name)
        return get_llm_config_spec(field_enum)


def validate_stdlib_config(component_type: str, config: Dict[str, Any]) -> List[str]:
    """Validate stdlib configuration by component type."""
    if component_type == "memory":
        return validate_memory_config(config.get("policy", ""), config)
    elif component_type == "llm":
        return validate_llm_config(config)
    elif component_type == "tool":
        return validate_tool_config(config.get("category", ""), config)
    else:
        return [f"Unknown component type: {component_type}"]


def get_stdlib_component(component_type: str, name: str):
    """Get stdlib component by type and name."""
    if component_type == "memory":
        return get_memory_policy(name)
    elif component_type == "llm":
        return get_llm_config_field(name)
    elif component_type == "tool":
        return get_tool_spec(name)
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def is_stdlib_import(module_name: str) -> bool:
    """Check if a module name refers to the standard library."""
    return module_name.startswith("stdlib.")


def parse_stdlib_import(module_name: str, imported_names: Optional[List[str]] = None, alias: Optional[str] = None) -> StdLibImport:
    """Parse an import statement targeting the standard library."""
    if not is_stdlib_import(module_name):
        raise StdLibResolutionError(f"Not a stdlib import: {module_name}")
    
    if imported_names and len(imported_names) == 1:
        return StdLibImport(
            module=module_name,
            component=imported_names[0], 
            alias=alias
        )
    elif not imported_names:
        return StdLibImport(
            module=module_name,
            alias=alias
        )
    else:
        raise StdLibResolutionError(f"Multiple component imports not yet supported: {imported_names}")


def validate_stdlib_import(import_stmt: StdLibImport) -> List[str]:
    """Validate a stdlib import statement."""
    registry = get_stdlib_type_registry()
    try:
        registry.resolve_stdlib_import(import_stmt)
        return []
    except StdLibResolutionError as e:
        return [str(e)]