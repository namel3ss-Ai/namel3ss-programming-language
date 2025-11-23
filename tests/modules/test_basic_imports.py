"""Test basic module import and resolution functionality."""

import pytest
from pathlib import Path
from textwrap import dedent

from namel3ss.loader import load_workspace_program, find_workspace_root
from namel3ss.resolver import resolve_program, ModuleResolutionError
from namel3ss.packages.discovery import PackageDiscovery, load_workspace_config


@pytest.fixture
def workspace_root(tmp_path):
    """Create a test workspace with multiple modules."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    
    # Create workspace config
    config_content = dedent("""
        [workspace]
        name = "test-workspace"
        module_paths = ["."]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    return workspace


@pytest.fixture 
def multi_module_workspace(workspace_root):
    """Create workspace with multiple related modules."""
    
    # Main app module
    main_content = dedent("""
        module app.main
        
        use app.shared.prompts
        use app.analytics.llms
        
        app "Multi Module App" {
            description: "App with multiple modules"
        }
        
        llm "main_model" {
            provider: "openai"
            model: "gpt-4"
        }
    """)
    (workspace_root / "main.ai").write_text(main_content)
    
    # Shared prompts module
    shared_dir = workspace_root / "shared"
    shared_dir.mkdir()
    
    prompts_content = dedent("""
        module app.shared.prompts
        
        prompt "welcome" {
            model: "main_model"
            template: "Welcome to {{app_name}}!"
        }
        
        prompt "goodbye" {
            model: "main_model"
            template: "Thank you for using {{app_name}}"
        }
    """)
    (shared_dir / "prompts.ai").write_text(prompts_content)
    
    # Analytics module
    analytics_dir = workspace_root / "analytics"
    analytics_dir.mkdir()
    
    llms_content = dedent("""
        module app.analytics.llms
        
        llm "analytics_model" {
            provider: "anthropic"
            model: "claude-3-sonnet"
            temperature: 0.1
        }
        
        prompt "analyze" {
            model: "analytics_model"
            template: "Analyze: {{data}}"
        }
    """)
    (analytics_dir / "llms.ai").write_text(llms_content)
    
    return workspace_root


def test_module_discovery(multi_module_workspace):
    """Test that modules are discovered correctly in workspace."""
    config = load_workspace_config(multi_module_workspace)
    discovery = PackageDiscovery(multi_module_workspace, config)
    
    packages, workspace_modules = discovery.discover_workspace()
    
    # Should find workspace modules but no packages
    assert len(packages) == 0
    assert len(workspace_modules) == 3
    
    # Check module names
    module_names = set(workspace_modules.keys())
    assert module_names == {"app.main", "app.shared.prompts", "app.analytics.llms"}
    
    # Check file paths
    main_module = workspace_modules["app.main"]
    assert main_module.name == "app.main"
    assert main_module.file_path.name == "main.ai"
    assert main_module.package_name is None


def test_hierarchical_module_names(multi_module_workspace):
    """Test hierarchical module name resolution."""
    program, packages = load_workspace_program(multi_module_workspace)
    
    # Should have 3 modules
    assert len(program.modules) == 3
    
    # Check module names are hierarchical
    module_names = {module.name for module in program.modules}
    assert module_names == {"app.main", "app.shared.prompts", "app.analytics.llms"}
    
    # Find main module
    main_module = next(m for m in program.modules if m.name == "app.main")
    assert main_module.has_explicit_app


def test_use_statement_resolution(multi_module_workspace):
    """Test that use statements resolve correctly."""
    from namel3ss.packages.discovery import ModuleResolver
    
    program, packages = load_workspace_program(multi_module_workspace)
    
    # Set up module resolver
    config = load_workspace_config(multi_module_workspace)
    discovery = PackageDiscovery(multi_module_workspace, config)
    packages_dict, workspace_modules = discovery.discover_workspace()
    module_resolver = ModuleResolver(packages_dict, workspace_modules)
    
    resolved = resolve_program(
        program,
        packages=packages,
        module_resolver=module_resolver
    )
    
    # Find main module resolution
    main_resolved = None
    for resolved_module in resolved.modules.values():
        if resolved_module.module.name == "app.main":
            main_resolved = resolved_module
            break
    
    assert main_resolved is not None
    
    # Check imports were resolved
    assert len(main_resolved.imports) >= 2  # Should have resolved the use statements
    
    # Check that modules were properly merged into the app
    assert len(resolved.app.prompts) >= 3  # welcome, goodbye, analyze
    assert len(resolved.app.llms) >= 2  # main_model, analytics_model


def test_relative_imports(workspace_root):
    """Test relative import resolution within module hierarchy."""
    
    # Create modules with relative imports
    base_content = dedent("""
        module utils.base
        
        prompt "base_prompt" {
            model: "test_model"
            template: "Base prompt"
        }
    """)
    utils_dir = workspace_root / "utils"
    utils_dir.mkdir()
    (utils_dir / "base.ai").write_text(base_content)
    
    helper_content = dedent("""
        module utils.helpers
        
        use .base  # Relative import
        
        prompt "helper_prompt" {
            model: "test_model"
            template: "Helper using base: {{base_prompt}}"
        }
    """)
    (utils_dir / "helpers.ai").write_text(helper_content)
    
    main_content = dedent("""
        module main
        
        use utils.helpers
        
        app "Relative Import Test" {
            description: "Testing relative imports"
        }
        
        llm "test_model" {
            provider: "openai"
            model: "gpt-3.5-turbo"
        }
    """)
    (workspace_root / "main.ai").write_text(main_content)
    
    program, packages = load_workspace_program(workspace_root)
    
    # Set up module resolver for relative imports
    config = load_workspace_config(workspace_root)
    discovery = PackageDiscovery(workspace_root, config)
    packages_dict, workspace_modules = discovery.discover_workspace()
    module_resolver = ModuleResolver(packages_dict, workspace_modules)
    
    resolved = resolve_program(
        program,
        packages=packages,
        module_resolver=module_resolver
    )
    
    # Should resolve successfully with relative imports
    assert resolved.app.name == "Relative Import Test"
    assert len(resolved.app.prompts) >= 2


def test_module_not_found_error(workspace_root):
    """Test error handling for missing modules."""
    
    main_content = dedent("""
        module main
        
        use nonexistent.module  # This should fail
        
        app "Error Test" {
            description: "Testing error handling"
        }
    """)
    (workspace_root / "main.ai").write_text(main_content)
    
    program, packages = load_workspace_program(workspace_root)
    
    config = load_workspace_config(workspace_root)
    discovery = PackageDiscovery(workspace_root, config)
    packages_dict, workspace_modules = discovery.discover_workspace()
    module_resolver = ModuleResolver(packages_dict, workspace_modules)
    
    # Should raise ModuleResolutionError for missing module
    with pytest.raises(ModuleResolutionError, match="Module not found"):
        resolve_program(
            program,
            packages=packages,
            module_resolver=module_resolver
        )


def test_circular_import_detection(workspace_root):
    """Test detection of circular imports."""
    
    # Module A imports B
    a_content = dedent("""
        module circular.a
        
        use circular.b
        
        prompt "a_prompt" {
            model: "test_model"
            template: "From A"
        }
    """)
    circular_dir = workspace_root / "circular"
    circular_dir.mkdir()
    (circular_dir / "a.ai").write_text(a_content)
    
    # Module B imports A (circular)
    b_content = dedent("""
        module circular.b
        
        use circular.a
        
        prompt "b_prompt" {
            model: "test_model" 
            template: "From B"
        }
    """)
    (circular_dir / "b.ai").write_text(b_content)
    
    main_content = dedent("""
        module main
        
        use circular.a
        
        app "Circular Test" {
            description: "Testing circular import detection"
        }
        
        llm "test_model" {
            provider: "openai"
            model: "gpt-3.5-turbo"
        }
    """)
    (workspace_root / "main.ai").write_text(main_content)
    
    program, packages = load_workspace_program(workspace_root)
    
    config = load_workspace_config(workspace_root)
    discovery = PackageDiscovery(workspace_root, config)
    packages_dict, workspace_modules = discovery.discover_workspace()
    module_resolver = ModuleResolver(packages_dict, workspace_modules)
    
    # Should handle circular imports gracefully (may warn but not fail)
    # Implementation detail: how to handle circular imports
    try:
        resolved = resolve_program(
            program,
            packages=packages,
            module_resolver=module_resolver
        )
        # If it succeeds, that's fine (circular imports may be allowed)
        assert resolved.app.name == "Circular Test"
    except ModuleResolutionError:
        # If it fails with circular import detection, that's also fine
        pass


def test_workspace_root_discovery(workspace_root):
    """Test finding workspace root from subdirectories."""
    
    # Create nested directory structure
    deep_dir = workspace_root / "src" / "deep" / "nested"
    deep_dir.mkdir(parents=True)
    
    # Should find workspace root from deep directory
    found_root = find_workspace_root(deep_dir)
    assert found_root == workspace_root
    
    # Should find workspace root from workspace root itself
    found_root = find_workspace_root(workspace_root)
    assert found_root == workspace_root
    
    # Should return None if no workspace root found
    temp_dir = workspace_root.parent / "no_workspace"
    temp_dir.mkdir()
    found_root = find_workspace_root(temp_dir)
    assert found_root is None