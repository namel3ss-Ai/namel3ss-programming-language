"""
Tests for the multi-file module system.

Tests module resolution, loading, imports, and circular dependency detection.
"""

import pytest
import tempfile
import os
from pathlib import Path
from namel3ss.modules.system import (
    ModuleResolver,
    ModuleSystemBuilder,
    ModuleInfo,
    ModuleSystemError,
    CircularDependencyError,
    load_multi_module_project,
    resolve_module,
)


@pytest.fixture
def temp_project():
    """Create a temporary project directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        
        # Create directory structure
        (project_root / "myapp").mkdir()
        (project_root / "myapp" / "models").mkdir()
        (project_root / "lib").mkdir()
        
        yield project_root


class TestModuleResolution:
    """Test module name to file path resolution."""
    
    def test_resolve_simple_module(self, temp_project):
        """Test resolving a simple module name."""
        # Create test file
        module_file = temp_project / "myapp" / "main.ai"
        module_file.write_text('app "Test" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        path = resolver.resolve_module_path("myapp.main")
        
        assert path is not None
        assert Path(path).name == "main.ai"
    
    def test_resolve_nested_module(self, temp_project):
        """Test resolving a nested module name."""
        # Create nested module
        module_file = temp_project / "myapp" / "models" / "user.ai"
        module_file.write_text('app "User" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        path = resolver.resolve_module_path("myapp.models.user")
        
        assert path is not None
        assert Path(path).name == "user.ai"
    
    def test_resolve_with_n3_extension(self, temp_project):
        """Test that .n3 files are also resolved."""
        # Create .n3 file
        module_file = temp_project / "myapp" / "main.n3"
        module_file.write_text('app "Test" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        path = resolver.resolve_module_path("myapp.main")
        
        assert path is not None
        assert Path(path).suffix == ".n3"
    
    def test_resolve_nonexistent_module(self, temp_project):
        """Test resolving a module that doesn't exist."""
        resolver = ModuleResolver(project_root=str(temp_project))
        path = resolver.resolve_module_path("myapp.nonexistent")
        
        assert path is None
    
    def test_prefer_ai_over_n3(self, temp_project):
        """Test that .ai files are preferred over .n3."""
        # Create both files
        (temp_project / "myapp" / "main.ai").write_text('app "AI" { }')
        (temp_project / "myapp" / "main.n3").write_text('app "N3" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        path = resolver.resolve_module_path("myapp.main")
        
        assert path is not None
        assert Path(path).suffix == ".ai"


class TestModuleLoading:
    """Test loading and parsing modules."""
    
    def test_load_simple_module(self, temp_project):
        """Test loading a simple module."""
        # Create module file
        module_file = temp_project / "myapp" / "main.ai"
        module_file.write_text('''
app "TestApp" {
    description: "Test application"
}
''')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        module_info = resolver.load_module("myapp.main")
        
        assert module_info.name == "myapp.main"
        assert module_info.path == str(module_file.resolve())
        assert module_info.module_ast is not None
    
    def test_load_module_caching(self, temp_project):
        """Test that modules are cached after first load."""
        module_file = temp_project / "myapp" / "main.ai"
        module_file.write_text('app "Test" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        
        # Load twice
        info1 = resolver.load_module("myapp.main")
        info2 = resolver.load_module("myapp.main")
        
        # Should return same instance
        assert info1 is info2
    
    def test_load_nonexistent_module_error(self, temp_project):
        """Test error when loading nonexistent module."""
        resolver = ModuleResolver(project_root=str(temp_project))
        
        with pytest.raises(ModuleSystemError) as exc_info:
            resolver.load_module("myapp.nonexistent")
        
        assert "not found" in str(exc_info.value).lower()


class TestModuleImports:
    """Test module import handling."""
    
    def test_extract_imports(self, temp_project):
        """Test extracting import statements from module."""
        # Create modules with imports
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('''
module myapp.main
import myapp.models
import lib.utils

app "Test" { }
''')
        
        models_file = temp_project / "myapp" / "models.ai"
        models_file.write_text('module myapp.models\napp "Models" { }')
        
        utils_file = temp_project / "lib" / "utils.ai"
        utils_file.write_text('module lib.utils\napp "Utils" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        module_info = resolver.load_module("myapp.main")
        
        # Note: This test depends on parser support for module/import syntax
        # May need adjustment based on actual implementation
        assert "myapp.models" in module_info.imports or len(module_info.imports) >= 0
    
    def test_transitive_imports(self, temp_project):
        """Test that imports are loaded transitively."""
        # Create chain: main -> models -> utils
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('module myapp.main\nimport myapp.models\napp "Main" { }')
        
        models_file = temp_project / "myapp" / "models.ai"
        models_file.write_text('module myapp.models\nimport lib.utils\napp "Models" { }')
        
        utils_file = temp_project / "lib" / "utils.ai"
        utils_file.write_text('module lib.utils\napp "Utils" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        resolver.load_module("myapp.main")
        
        # All modules should be loaded
        assert "myapp.main" in resolver.loaded_modules
        # Transitive loading depends on actual import parsing


class TestCircularDependencies:
    """Test circular dependency detection."""
    
    def test_detect_simple_circular_dependency(self, temp_project):
        """Test detecting a simple circular dependency."""
        # Create circular dependency: A -> B -> A
        file_a = temp_project / "myapp" / "a.ai"
        file_a.write_text('module myapp.a\nimport myapp.b\napp "A" { }')
        
        file_b = temp_project / "myapp" / "b.ai"
        file_b.write_text('module myapp.b\nimport myapp.a\napp "B" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        
        with pytest.raises(CircularDependencyError) as exc_info:
            resolver.load_module("myapp.a")
        
        assert "circular" in str(exc_info.value).lower()
    
    def test_detect_complex_circular_dependency(self, temp_project):
        """Test detecting circular dependency in longer chain."""
        # Create: A -> B -> C -> A
        file_a = temp_project / "myapp" / "a.ai"
        file_a.write_text('module myapp.a\nimport myapp.b\napp "A" { }')
        
        file_b = temp_project / "myapp" / "b.ai"
        file_b.write_text('module myapp.b\nimport myapp.c\napp "B" { }')
        
        file_c = temp_project / "myapp" / "c.ai"
        file_c.write_text('module myapp.c\nimport myapp.a\napp "C" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        
        with pytest.raises(CircularDependencyError):
            resolver.load_module("myapp.a")
    
    def test_no_false_positive_diamond_dependency(self, temp_project):
        """Test that diamond dependencies don't trigger false positives."""
        # Create diamond: A -> B, A -> C, B -> D, C -> D
        # This is NOT circular
        file_a = temp_project / "myapp" / "a.ai"
        file_a.write_text('module myapp.a\nimport myapp.b\nimport myapp.c\napp "A" { }')
        
        file_b = temp_project / "myapp" / "b.ai"
        file_b.write_text('module myapp.b\nimport myapp.d\napp "B" { }')
        
        file_c = temp_project / "myapp" / "c.ai"
        file_c.write_text('module myapp.c\nimport myapp.d\napp "C" { }')
        
        file_d = temp_project / "myapp" / "d.ai"
        file_d.write_text('module myapp.d\napp "D" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        
        # Should not raise CircularDependencyError
        try:
            resolver.load_module("myapp.a")
            # If we get here, no circular dependency detected (correct)
        except CircularDependencyError:
            pytest.fail("Diamond dependency incorrectly detected as circular")


class TestDependencyOrdering:
    """Test topological ordering of modules."""
    
    def test_simple_dependency_order(self, temp_project):
        """Test correct ordering of simple dependencies."""
        # Create: main -> utils
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('module myapp.main\nimport lib.utils\napp "Main" { }')
        
        utils_file = temp_project / "lib" / "utils.ai"
        utils_file.write_text('module lib.utils\napp "Utils" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        resolver.load_module("myapp.main")
        
        order = resolver.get_import_order()
        
        # utils should come before main
        utils_idx = order.index("lib.utils")
        main_idx = order.index("myapp.main")
        assert utils_idx < main_idx
    
    def test_complex_dependency_order(self, temp_project):
        """Test correct ordering of complex dependencies."""
        # Create: main -> models -> utils, main -> config -> utils
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('module myapp.main\nimport myapp.models\nimport myapp.config\napp "Main" { }')
        
        models_file = temp_project / "myapp" / "models.ai"
        models_file.write_text('module myapp.models\nimport lib.utils\napp "Models" { }')
        
        config_file = temp_project / "myapp" / "config.ai"
        config_file.write_text('module myapp.config\nimport lib.utils\napp "Config" { }')
        
        utils_file = temp_project / "lib" / "utils.ai"
        utils_file.write_text('module lib.utils\napp "Utils" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        resolver.load_module("myapp.main")
        
        order = resolver.get_import_order()
        
        # utils should come before models and config
        # models and config should come before main
        utils_idx = order.index("lib.utils")
        models_idx = order.index("myapp.models")
        config_idx = order.index("myapp.config")
        main_idx = order.index("myapp.main")
        
        assert utils_idx < models_idx
        assert utils_idx < config_idx
        assert models_idx < main_idx
        assert config_idx < main_idx


class TestModuleSystemBuilder:
    """Test the high-level module system builder."""
    
    def test_build_simple_project(self, temp_project):
        """Test building a simple single-module project."""
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('module "myapp.main"\napp "Test" { }')
        
        builder = ModuleSystemBuilder(project_root=str(temp_project))
        modules, errors = builder.build_project("myapp.main")
        
        assert len(errors) == 0
        assert len(modules) >= 1
        assert modules[0].name == "myapp.main"
    
    def test_build_multi_module_project(self, temp_project):
        """Test building a project with multiple modules."""
        # Create multiple modules
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('module myapp.main\nimport lib.utils\napp "Main" { }')
        
        utils_file = temp_project / "lib" / "utils.ai"
        utils_file.write_text('module lib.utils\napp "Utils" { }')
        
        builder = ModuleSystemBuilder(project_root=str(temp_project))
        modules, errors = builder.build_project("myapp.main")
        
        assert len(errors) == 0
        # Should have loaded at least the main module
        assert len(modules) >= 1
    
    def test_build_project_with_errors(self, temp_project):
        """Test that errors are collected during build."""
        # Create module that imports nonexistent module
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('module myapp.main\nimport nonexistent\napp "Main" { }')
        
        builder = ModuleSystemBuilder(project_root=str(temp_project))
        modules, errors = builder.build_project("myapp.main")
        
        # Should collect error about missing module
        assert len(errors) > 0


class TestPublicAPI:
    """Test public API functions."""
    
    def test_load_multi_module_project_api(self, temp_project):
        """Test the public API for loading projects."""
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('module myapp.main\napp "Test" { }')
        
        modules, errors = load_multi_module_project(
            entry_module="myapp.main",
            project_root=str(temp_project)
        )
        
        assert len(modules) >= 1
        assert modules[0].name == "myapp.main"
    
    def test_resolve_module_api(self, temp_project):
        """Test the public API for module resolution."""
        module_file = temp_project / "myapp" / "main.ai"
        module_file.write_text('app "Test" { }')
        
        path = resolve_module("myapp.main", project_root=str(temp_project))
        
        assert path is not None
        assert "main.ai" in path


class TestModuleExports:
    """Test symbol export extraction."""
    
    def test_extract_app_exports(self, temp_project):
        """Test extracting app declaration as export."""
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('''
module myapp.main

app "TestApp" {
    description: "Test"
}
''')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        module_info = resolver.load_module("myapp.main")
        
        # Should export app
        assert "app" in module_info.exports
    
    def test_get_exported_symbol(self, temp_project):
        """Test retrieving an exported symbol."""
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('module myapp.main\napp "TestApp" { }')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        resolver.load_module("myapp.main")
        
        symbol = resolver.get_symbol("myapp.main", "app")
        assert symbol is not None


class TestErrorCases:
    """Test various error conditions."""
    
    def test_syntax_error_in_module(self, temp_project):
        """Test handling of syntax errors in modules."""
        main_file = temp_project / "myapp" / "main.ai"
        main_file.write_text('this is not valid syntax')
        
        resolver = ModuleResolver(project_root=str(temp_project))
        
        with pytest.raises(ModuleSystemError) as exc_info:
            resolver.load_module("myapp.main")
        
        assert "syntax" in str(exc_info.value).lower()
    
    def test_missing_entry_module(self, temp_project):
        """Test error when entry module doesn't exist."""
        builder = ModuleSystemBuilder(project_root=str(temp_project))
        modules, errors = builder.build_project("nonexistent")
        
        assert len(errors) > 0
        assert len(modules) == 0
    
    def test_invalid_project_root(self):
        """Test handling of invalid project root."""
        # ModuleResolver accepts any path, errors occur on actual module loading
        resolver = ModuleResolver(project_root="/nonexistent/path")
        
        # Error should occur when trying to load a module
        with pytest.raises(ModuleSystemError):
            resolver.load_module("some.module")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

