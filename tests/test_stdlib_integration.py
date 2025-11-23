"""Integration tests for standard library with type system."""

import pytest

from namel3ss.ast import (
    App, Module, Program,
    LLMDefinition, ToolDefinition, Memory
)
from namel3ss.ast.modules import Import, ImportedName
from namel3ss.resolver import resolve_program
from namel3ss.types import check_program_with_stdlib, N3TypeError
from namel3ss.stdlib.typing import (
    get_stdlib_type_registry,
    StdLibValidator,
    is_stdlib_import,
    parse_stdlib_import
)


def test_stdlib_import_detection():
    """Test detection of stdlib imports."""
    assert is_stdlib_import("stdlib.memory")
    assert is_stdlib_import("stdlib.llm")
    assert is_stdlib_import("stdlib.tools")
    assert not is_stdlib_import("regular.module")
    assert not is_stdlib_import("user_module")


def test_stdlib_import_parsing():
    """Test parsing of stdlib import statements."""
    # Module import
    import_stmt = parse_stdlib_import("stdlib.memory")
    assert import_stmt.module == "stdlib.memory"
    assert import_stmt.component is None
    assert import_stmt.alias is None
    
    # Component import
    import_stmt = parse_stdlib_import("stdlib.memory", ["conversation_window"])
    assert import_stmt.module == "stdlib.memory"
    assert import_stmt.component == "conversation_window"
    
    # With alias
    import_stmt = parse_stdlib_import("stdlib.llm", alias="llm_config")
    assert import_stmt.module == "stdlib.llm"
    assert import_stmt.alias == "llm_config"


def test_stdlib_type_registry():
    """Test stdlib type registry functionality."""
    registry = get_stdlib_type_registry()
    
    # Test symbol lookup
    temp_symbol = registry.get_stdlib_symbol("temperature")
    assert temp_symbol is not None
    assert temp_symbol.name == "temperature"
    
    # Test component-specific symbols
    memory_symbols = registry.list_stdlib_symbols(
        registry.stdlib_registry._get_component_type("memory")
    )
    assert len(memory_symbols) > 0
    assert any(symbol.name == "conversation_window" for symbol in memory_symbols)


def test_stdlib_validator():
    """Test stdlib configuration validation."""
    registry = get_stdlib_type_registry()
    validator = StdLibValidator(registry)
    
    # Valid memory config
    memory_config = {
        "policy": "conversation_window",
        "window_size": 10
    }
    errors = validator.validate_memory_config(memory_config)
    assert not errors
    
    # Invalid memory config
    invalid_config = {
        "policy": "nonexistent_policy"
    }
    errors = validator.validate_memory_config(invalid_config)
    assert len(errors) > 0
    
    # Valid LLM config
    llm_config = {
        "temperature": 0.7,
        "max_tokens": 1024
    }
    errors = validator.validate_llm_config(llm_config)
    assert not errors
    
    # Invalid LLM config
    invalid_llm_config = {
        "temperature": -1.0  # Out of range
    }
    errors = validator.validate_llm_config(invalid_llm_config)
    assert len(errors) > 0


def test_stdlib_import_resolution():
    """Test integration of stdlib imports with module resolution."""
    # Create module with stdlib imports
    module = Module(
        name="test_module",
        imports=[
            Import(module="stdlib.memory", names=[ImportedName(name="conversation_window")]),
            Import(module="stdlib.llm", names=[ImportedName(name="temperature")]),
            Import(module="stdlib.tools", names=[ImportedName(name="http")])
        ],
        body=[
            App(
                name="TestApp",
                llms=[
                    LLMDefinition(
                        name="test_llm",
                        provider="openai", 
                        model="gpt-3.5-turbo",
                        config={"temperature": 0.7, "max_tokens": 1024}
                    )
                ],
                tools=[
                    ToolDefinition(
                        name="test_tool",
                        category="http",
                        config={
                            "category": "http",
                            "method": "GET", 
                            "url": "https://api.example.com",
                            "description": "Test API"
                        }
                    )
                ],
                memories=[
                    Memory(
                        name="test_memory",
                        config={
                            "policy": "conversation_window",
                            "window_size": 5
                        }
                    )
                ]
            )
        ],
        has_explicit_app=True
    )
    
    program = Program(modules=[module])
    
    # Should resolve without errors
    resolved = resolve_program(program)
    assert resolved is not None
    
    # Should have stdlib imports
    root_imports = resolved.root.imports
    stdlib_imports = [imp for imp in root_imports if imp.stdlib_import]
    assert len(stdlib_imports) == 3
    
    # Check specific imports
    memory_import = next((imp for imp in stdlib_imports if imp.target_module == "stdlib.memory"), None)
    assert memory_import is not None
    assert "conversation_window" in memory_import.stdlib_symbols


def test_stdlib_validation_in_type_checker():
    """Test that type checker validates stdlib usage."""
    # Valid configuration - should pass
    valid_module = Module(
        name="valid_module",
        imports=[
            Import(module="stdlib.llm", names=[ImportedName(name="temperature")]),
        ],
        body=[
            App(
                name="ValidApp",
                llms=[
                    LLMDefinition(
                        name="valid_llm",
                        provider="openai",
                        model="gpt-3.5-turbo", 
                        config={"temperature": 0.7}  # Valid temperature
                    )
                ]
            )
        ],
        has_explicit_app=True
    )
    
    valid_program = Program(modules=[valid_module])
    resolved_valid = resolve_program(valid_program)
    
    # Should not raise an error
    check_program_with_stdlib(resolved_valid)
    
    # Invalid configuration - should fail
    invalid_module = Module(
        name="invalid_module", 
        imports=[
            # Missing import for temperature
        ],
        body=[
            App(
                name="InvalidApp",
                llms=[
                    LLMDefinition(
                        name="invalid_llm",
                        provider="openai",
                        model="gpt-3.5-turbo",
                        config={"temperature": 3.0}  # Out of range + not imported
                    )
                ]
            )
        ],
        has_explicit_app=True
    )
    
    invalid_program = Program(modules=[invalid_module])
    resolved_invalid = resolve_program(invalid_program)
    
    # Should raise N3TypeError due to stdlib validation
    with pytest.raises(N3TypeError):
        check_program_with_stdlib(resolved_invalid)


def test_stdlib_import_error_handling():
    """Test error handling for invalid stdlib imports."""
    # Invalid stdlib module
    with pytest.raises(Exception):  # Should be caught during parsing
        parse_stdlib_import("stdlib.nonexistent")
    
    # Invalid component
    import_stmt = parse_stdlib_import("stdlib.memory", ["nonexistent_policy"])
    registry = get_stdlib_type_registry()
    
    with pytest.raises(Exception):  # Should fail during resolution
        registry.resolve_stdlib_import(import_stmt)


def test_stdlib_symbol_aliasing():
    """Test aliasing of stdlib symbols."""
    # Import with alias
    import_stmt = parse_stdlib_import("stdlib.memory", ["conversation_window"], alias="window")
    registry = get_stdlib_type_registry()
    
    symbols = registry.resolve_stdlib_import(import_stmt)
    assert "window" in symbols  # Should use alias
    assert "conversation_window" not in symbols  # Original name not available


def test_complete_stdlib_integration():
    """Test complete integration with realistic stdlib usage."""
    module = Module(
        name="complete_test",
        imports=[
            Import(module="stdlib.memory", names=[ImportedName(name="conversation_window")]),
            Import(module="stdlib.llm", names=[
                ImportedName(name="temperature"),
                ImportedName(name="max_tokens"),
                ImportedName(name="top_p")
            ]),
            Import(module="stdlib.tools", names=[ImportedName(name="http")])
        ],
        body=[
            App(
                name="CompleteApp",
                llms=[
                    LLMDefinition(
                        name="production_llm",
                        provider="openai",
                        model="gpt-4",
                        config={
                            "temperature": 0.8,
                            "max_tokens": 2048, 
                            "top_p": 0.9,
                            "frequency_penalty": 0.1
                        }
                    )
                ],
                tools=[
                    ToolDefinition(
                        name="api_client",
                        category="http",
                        config={
                            "category": "http",
                            "method": "POST",
                            "url": "https://api.production.com/endpoint",
                            "description": "Production API client",
                            "headers": {"Authorization": "Bearer token"},
                            "timeout": 30
                        }
                    )
                ],
                memories=[
                    Memory(
                        name="conversation_memory",
                        config={
                            "policy": "conversation_window",
                            "window_size": 20,
                            "include_system_messages": True
                        }
                    )
                ]
            )
        ],
        has_explicit_app=True
    )
    
    program = Program(modules=[module])
    
    # Should resolve and validate successfully
    resolved = resolve_program(program)
    check_program_with_stdlib(resolved)
    
    # Verify stdlib symbols were resolved
    root_imports = resolved.root.imports
    stdlib_imports = [imp for imp in root_imports if imp.stdlib_import]
    
    total_symbols = sum(len(imp.stdlib_symbols) for imp in stdlib_imports)
    assert total_symbols >= 4  # temperature, max_tokens, top_p, http, conversation_window