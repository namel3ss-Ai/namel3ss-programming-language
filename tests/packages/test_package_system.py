"""Test package system functionality including manifests and dependencies."""

import pytest
from pathlib import Path
from textwrap import dedent
import tempfile

from namel3ss.loader import load_workspace_program
from namel3ss.packages import PackageManifest, PackageDependency, VersionConstraint
from namel3ss.packages.discovery import PackageDiscovery, DependencyResolver, load_workspace_config
from namel3ss.packages import PackageNotFoundError, DependencyCycleError, PackageVersionConflictError


@pytest.fixture
def package_workspace(tmp_path):
    """Create a workspace with packages."""
    workspace = tmp_path / "package_workspace"
    workspace.mkdir()
    
    # Create workspace config  
    config_content = dedent("""
        [workspace]
        name = "package-test-workspace"
        module_paths = ["."]
        package_paths = ["packages", "libs"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    return workspace


@pytest.fixture
def prompt_package(package_workspace):
    """Create a sample prompt package."""
    packages_dir = package_workspace / "packages"
    packages_dir.mkdir()
    
    # Create prompt package
    prompt_pkg_dir = packages_dir / "prompt_pack"
    prompt_pkg_dir.mkdir()
    
    # Package manifest
    manifest_content = dedent("""
        [package]
        name = "acme.prompts"
        version = "1.0.0"
        description = "Collection of reusable prompts"
        authors = ["Acme Corp"]
        
        [dependencies]
        
        [dev-dependencies]
        
        modules = [
            "customer_support",
            "analytics.reports"
        ]
        
        [module-exports]
        
        [namel3ss]
        version = ">=1.0.0"
    """)
    (prompt_pkg_dir / "namel3ss.toml").write_text(manifest_content)
    
    # Customer support prompts
    customer_content = dedent("""
        module customer_support
        
        prompt "greet_customer" {
            template: "Hello {{customer_name}}, how can I help you today?"
        }
        
        prompt "escalate_issue" {
            template: "I understand your concern. Let me escalate this to our specialist team."
        }
    """)
    (prompt_pkg_dir / "customer_support.ai").write_text(customer_content)
    
    # Analytics reports module
    analytics_dir = prompt_pkg_dir / "analytics"
    analytics_dir.mkdir()
    
    reports_content = dedent("""
        module analytics.reports
        
        prompt "daily_summary" {
            template: "Daily Report for {{date}}: {{metrics}}"
        }
        
        prompt "performance_analysis" {
            template: "Performance Analysis: {{data}}"
        }
    """)
    (analytics_dir / "reports.ai").write_text(reports_content)
    
    return prompt_pkg_dir


@pytest.fixture
def tool_package(package_workspace):
    """Create a sample tool package."""
    packages_dir = package_workspace / "packages"
    if not packages_dir.exists():
        packages_dir.mkdir()
        
    tool_pkg_dir = packages_dir / "common_tools"
    tool_pkg_dir.mkdir()
    
    # Package manifest with dependency
    manifest_content = dedent("""
        [package]
        name = "acme.tools"
        version = "2.1.0"
        description = "Common tools and utilities"
        
        [dependencies]
        "acme.prompts" = ">=1.0.0"
        
        modules = ["http_client", "database"]
    """)
    (tool_pkg_dir / "namel3ss.toml").write_text(manifest_content)
    
    # HTTP client module
    http_content = dedent("""
        module http_client
        
        use acme.prompts::customer_support
        
        # Tool that uses prompts from dependency
        tool "api_client" {
            type: "http"
            base_url: "https://api.example.com"
        }
    """)
    (tool_pkg_dir / "http_client.ai").write_text(http_content)
    
    # Database module
    db_content = dedent("""
        module database
        
        tool "user_db" {
            type: "database"
            connection: "postgresql://localhost/users"
        }
    """)
    (tool_pkg_dir / "database.ai").write_text(db_content)
    
    return tool_pkg_dir


def test_package_manifest_parsing():
    """Test parsing of package manifest files."""
    manifest_data = {
        'package': {
            'name': 'test.package',
            'version': '1.2.3',
            'description': 'Test package',
            'authors': ['Test Author']
        },
        'dependencies': {
            'other.package': '>=1.0.0'
        },
        'modules': ['main', 'utils.helpers']
    }
    
    manifest = PackageManifest.from_dict(manifest_data)
    
    assert manifest.name == 'test.package'
    assert manifest.version == '1.2.3'
    assert manifest.description == 'Test package'
    assert 'other.package' in manifest.dependencies
    
    dep = manifest.dependencies['other.package']
    assert dep.name == 'other.package'
    assert str(dep.constraint) == '>=1.2.3'


def test_version_constraints():
    """Test version constraint parsing and matching."""
    # Test various constraint formats
    constraints = [
        ('>=1.0.0', '1.5.0', True),
        ('>=1.0.0', '0.9.0', False),
        ('^1.2.0', '1.3.0', True),
        ('^1.2.0', '2.0.0', False),
        ('~=1.2.0', '1.2.5', True),
        ('~=1.2.0', '1.3.0', False),
        ('==1.0.0', '1.0.0', True),
        ('==1.0.0', '1.0.1', False),
    ]
    
    for constraint_str, version, should_match in constraints:
        constraint = VersionConstraint.parse(constraint_str)
        result = constraint.matches(version)
        assert result == should_match, f"{constraint_str} vs {version} should be {should_match}"


def test_package_discovery(package_workspace, prompt_package, tool_package):
    """Test package discovery in workspace."""
    config = load_workspace_config(package_workspace)
    discovery = PackageDiscovery(package_workspace, config)
    
    packages, workspace_modules = discovery.discover_workspace()
    
    # Should find both packages
    assert len(packages) == 2
    assert 'acme.prompts' in packages
    assert 'acme.tools' in packages
    
    # Check prompt package
    prompt_pkg = packages['acme.prompts']
    assert prompt_pkg.manifest.name == 'acme.prompts'
    assert prompt_pkg.manifest.version == '1.0.0'
    assert len(prompt_pkg.modules) == 2
    assert 'customer_support' in prompt_pkg.modules
    assert 'analytics.reports' in prompt_pkg.modules
    
    # Check tool package
    tool_pkg = packages['acme.tools']
    assert tool_pkg.manifest.name == 'acme.tools'
    assert 'acme.prompts' in tool_pkg.manifest.dependencies


def test_dependency_resolution(package_workspace, prompt_package, tool_package):
    """Test package dependency resolution."""
    config = load_workspace_config(package_workspace)
    discovery = PackageDiscovery(package_workspace, config)
    packages, _ = discovery.discover_workspace()
    
    resolver = DependencyResolver(packages)
    
    # Resolve dependencies starting from tool package
    resolved = resolver.resolve_dependencies(['acme.tools'])
    
    # Should resolve both packages in dependency order
    assert len(resolved) == 2
    resolved_names = [pkg.manifest.name for pkg in resolved]
    
    # acme.prompts should come before acme.tools (dependency order)
    prompt_idx = resolved_names.index('acme.prompts')
    tools_idx = resolved_names.index('acme.tools')
    assert prompt_idx < tools_idx


def test_cross_package_imports(package_workspace, prompt_package, tool_package):
    """Test imports across packages."""
    # Create main app that uses both packages
    main_content = dedent("""
        module main
        
        use acme.prompts::customer_support
        use acme.tools::http_client
        
        app "Cross Package App" {
            description: "App using multiple packages"
        }
        
        llm "main_model" {
            provider: "openai"
            model: "gpt-4"
        }
    """)
    (package_workspace / "main.ai").write_text(main_content)
    
    # Load workspace with packages
    program, packages = load_workspace_program(package_workspace)
    
    # Should have modules from workspace and packages
    module_names = {module.name for module in program.modules}
    expected = {
        'main',  # workspace module
        'acme.prompts::customer_support',  # package modules
        'acme.prompts::analytics.reports',
        'acme.tools::http_client',
        'acme.tools::database'
    }
    assert module_names == expected


def test_package_not_found_error():
    """Test error when required package is not found."""
    packages = {}  # Empty packages
    resolver = DependencyResolver(packages)
    
    with pytest.raises(PackageNotFoundError, match="Package not found: nonexistent"):
        resolver.resolve_dependencies(['nonexistent'])


def test_circular_dependency_detection(tmp_path):
    """Test detection of circular dependencies between packages."""
    workspace = tmp_path / "circular_test"
    workspace.mkdir()
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Package A depends on B
    pkg_a_dir = packages_dir / "pkg_a"
    pkg_a_dir.mkdir()
    
    manifest_a = dedent("""
        [package]
        name = "test.a"
        version = "1.0.0"
        
        [dependencies]
        "test.b" = "1.0.0"
    """)
    (pkg_a_dir / "namel3ss.toml").write_text(manifest_a)
    
    (pkg_a_dir / "main.ai").write_text('module main\nprompt "a" { template: "A" }')
    
    # Package B depends on A (circular)
    pkg_b_dir = packages_dir / "pkg_b"
    pkg_b_dir.mkdir()
    
    manifest_b = dedent("""
        [package]
        name = "test.b"
        version = "1.0.0"
        
        [dependencies]
        "test.a" = "1.0.0"
    """)
    (pkg_b_dir / "namel3ss.toml").write_text(manifest_b)
    
    (pkg_b_dir / "main.ai").write_text('module main\nprompt "b" { template: "B" }')
    
    # Create workspace config
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    packages, _ = discovery.discover_workspace()
    
    resolver = DependencyResolver(packages)
    
    # Should detect circular dependency
    with pytest.raises(DependencyCycleError, match="Circular dependency"):
        resolver.resolve_dependencies(['test.a'])


def test_package_version_constraints(tmp_path):
    """Test version constraint validation."""
    workspace = tmp_path / "version_test"
    workspace.mkdir()
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Create package with specific version
    pkg_dir = packages_dir / "versioned_pkg"
    pkg_dir.mkdir()
    
    manifest = dedent("""
        [package]
        name = "test.versioned"
        version = "2.0.0"
    """)
    (pkg_dir / "namel3ss.toml").write_text(manifest)
    (pkg_dir / "main.ai").write_text('module main\nprompt "test" { template: "Test" }')
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    packages, _ = discovery.discover_workspace()
    
    resolver = DependencyResolver(packages)
    
    # Test version constraint matching
    versioned_pkg = packages['test.versioned']
    assert versioned_pkg.manifest.version == '2.0.0'
    
    # Create dependency with constraint
    constraint = VersionConstraint.parse('>=1.0.0')
    assert constraint.matches('2.0.0')
    
    constraint_strict = VersionConstraint.parse('==1.0.0')
    assert not constraint_strict.matches('2.0.0')


def test_workspace_info_with_packages(package_workspace, prompt_package, tool_package):
    """Test workspace info includes package information."""
    from namel3ss.loader import get_workspace_info
    
    info = get_workspace_info(package_workspace)
    
    assert info['total_packages'] == 2
    assert 'acme.prompts' in info['packages']
    assert 'acme.tools' in info['packages']
    
    prompt_info = info['packages']['acme.prompts']
    assert prompt_info['version'] == '1.0.0'
    assert prompt_info['module_count'] == 2
    assert 'customer_support' in prompt_info['modules']
    assert 'analytics.reports' in prompt_info['modules']