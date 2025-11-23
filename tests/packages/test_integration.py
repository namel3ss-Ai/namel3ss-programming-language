"""Integration tests for complete module and package workflows."""

import pytest
from pathlib import Path
from textwrap import dedent
import tempfile

from namel3ss.loader import load_workspace_program, get_workspace_info
from namel3ss.resolver import resolve_program
from namel3ss.packages.discovery import PackageDiscovery, load_workspace_config


@pytest.fixture
def complex_workspace(tmp_path):
    """Create a complex workspace with multiple packages and dependencies."""
    workspace = tmp_path / "complex_workspace"
    workspace.mkdir()
    
    # Workspace configuration
    config_content = dedent("""
        [workspace]
        name = "complex-workspace"
        module_paths = ["src", "apps"]
        package_paths = ["packages", "external"]
        
        [dependencies]
        # Global workspace dependencies
        
        [dev-dependencies]
        
        [workspace.settings]
        default_llm_provider = "openai"
        enable_type_checking = true
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    return workspace


@pytest.fixture
def core_utilities_package(complex_workspace):
    """Create core utilities package."""
    packages_dir = complex_workspace / "packages"
    packages_dir.mkdir(exist_ok=True)
    
    core_pkg = packages_dir / "core_utils"
    core_pkg.mkdir()
    
    # Package manifest
    manifest_content = dedent("""
        [package]
        name = "company.core"
        version = "2.1.0"
        description = "Core utilities and shared components"
        authors = ["Company DevTeam <dev@company.com>"]
        license = "MIT"
        
        [dependencies]
        # No external dependencies for core package
        
        [module-exports]
        # Export specific modules for external use
        "utils.validation" = "utils.validation"
        "utils.formatting" = "utils.formatting"
        "logging" = "logging"
        
        modules = [
            "utils.validation",
            "utils.formatting", 
            "logging",
            "config"
        ]
        
        [namel3ss]
        version = ">=1.0.0"
    """)
    (core_pkg / "namel3ss.toml").write_text(manifest_content)
    
    # Utils validation module
    utils_dir = core_pkg / "utils"
    utils_dir.mkdir()
    
    validation_content = dedent("""
        module utils.validation
        
        prompt "validate_email" {
            template: "Validate this email address: {{email}}"
            validation: {
                required: true,
                type: "email"
            }
        }
        
        prompt "validate_input" {
            template: "Validate input: {{input}} against rules: {{rules}}"
        }
        
        tool "email_validator" {
            type: "validation"
            regex: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        }
    """)
    (utils_dir / "validation.ai").write_text(validation_content)
    
    formatting_content = dedent("""
        module utils.formatting
        
        prompt "format_json" {
            template: "Format this JSON data: {{data}}"
        }
        
        prompt "format_table" {
            template: "Create a table from this data: {{data}}"
        }
    """)
    (utils_dir / "formatting.ai").write_text(formatting_content)
    
    # Logging module
    logging_content = dedent("""
        module logging
        
        prompt "log_info" {
            template: "[INFO] {{timestamp}}: {{message}}"
        }
        
        prompt "log_error" {
            template: "[ERROR] {{timestamp}}: {{error}} - {{context}}"
        }
        
        tool "logger" {
            type: "logging"
            level: "info"
            output: "console"
        }
    """)
    (core_pkg / "logging.ai").write_text(logging_content)
    
    # Config module (internal to package)
    config_content = dedent("""
        module config
        
        # Internal configuration - not exported
        prompt "load_config" {
            template: "Load configuration from: {{source}}"
        }
    """)
    (core_pkg / "config.ai").write_text(config_content)
    
    return core_pkg


@pytest.fixture 
def analytics_package(complex_workspace, core_utilities_package):
    """Create analytics package that depends on core utilities."""
    packages_dir = complex_workspace / "packages"
    
    analytics_pkg = packages_dir / "analytics"
    analytics_pkg.mkdir()
    
    # Package manifest with dependency
    manifest_content = dedent("""
        [package]
        name = "company.analytics"
        version = "1.5.0"
        description = "Analytics and reporting tools"
        
        [dependencies]
        "company.core" = ">=2.0.0"
        
        modules = [
            "reports.daily",
            "reports.monthly", 
            "metrics.performance",
            "visualizations"
        ]
        
        [module-exports]
        "reports" = ["reports.daily", "reports.monthly"]
        "metrics" = "metrics.performance"
    """)
    (analytics_pkg / "namel3ss.toml").write_text(manifest_content)
    
    # Reports modules
    reports_dir = analytics_pkg / "reports"
    reports_dir.mkdir()
    
    daily_content = dedent("""
        module reports.daily
        
        use company.core::utils.formatting
        use company.core::logging
        
        prompt "daily_report" {
            template: "Daily Report for {{date}}"
        }
        
        prompt "daily_metrics" {
            template: "Metrics: {{metrics}}"
        }
        
        app "DailyReporter" {
            description: "Generate daily reports"
            uses: ["utils.formatting", "logging"]
        }
    """)
    (reports_dir / "daily.ai").write_text(daily_content)
    
    monthly_content = dedent("""
        module reports.monthly
        
        use company.core::logging
        use company.analytics::reports.daily as daily
        
        prompt "monthly_summary" {
            template: "Monthly Summary for {{month}}"
        }
        
        prompt "trend_analysis" {
            template: "Trend analysis: {{trends}}"
        }
    """)
    (reports_dir / "monthly.ai").write_text(monthly_content)
    
    # Metrics module
    metrics_dir = analytics_pkg / "metrics"
    metrics_dir.mkdir()
    
    performance_content = dedent("""
        module metrics.performance
        
        use company.core::utils.validation
        
        prompt "measure_performance" {
            template: "Performance metrics: {{data}}"
        }
        
        tool "performance_tracker" {
            type: "metrics"
            interval: "1m"
        }
    """)
    (metrics_dir / "performance.ai").write_text(performance_content)
    
    # Visualizations module  
    viz_content = dedent("""
        module visualizations
        
        use company.core::utils.formatting
        
        prompt "create_chart" {
            template: "Create chart: {{chart_type}} with data: {{data}}"
        }
        
        tool "chart_generator" {
            type: "visualization"
            formats: ["png", "svg", "html"]
        }
    """)
    (analytics_pkg / "visualizations.ai").write_text(viz_content)
    
    return analytics_pkg


@pytest.fixture
def customer_app(complex_workspace, analytics_package):
    """Create customer-facing application using packages."""
    apps_dir = complex_workspace / "apps"
    apps_dir.mkdir()
    
    # Customer service app
    customer_app_content = dedent("""
        module customer_service
        
        use company.core::utils.validation
        use company.core::logging
        use company.analytics::reports.daily as analytics
        
        app "CustomerServiceApp" {
            name: "Customer Service Dashboard"
            description: "Customer support with analytics"
            version: "1.0.0"
        }
        
        llm "customer_assistant" {
            provider: "openai" 
            model: "gpt-4"
            temperature: 0.3
        }
        
        prompt "greet_customer" {
            template: "Hello {{customer_name}}, how can I help you today?"
        }
        
        prompt "escalate_issue" {
            template: "Escalating issue: {{issue}} to {{department}}"
        }
        
        workflow "handle_customer_inquiry" {
            steps: [
                "validate_input",
                "greet_customer", 
                "process_inquiry",
                "log_interaction"
            ]
        }
    """)
    (apps_dir / "customer_service.ai").write_text(customer_app_content)
    
    # Main workspace module
    main_content = dedent("""
        module main
        
        use company.core::logging
        use company.analytics::metrics
        use apps.customer_service as customer
        
        app "CompanyMainApp" {
            description: "Main company application"
            modules: ["customer_service", "analytics"]
        }
        
        llm "main_model" {
            provider: "openai"
            model: "gpt-4"
        }
    """)
    (complex_workspace / "main.ai").write_text(main_content)
    
    return apps_dir


def test_complex_workspace_discovery(complex_workspace, core_utilities_package, analytics_package, customer_app):
    """Test discovery of complex workspace structure."""
    config = load_workspace_config(complex_workspace)
    discovery = PackageDiscovery(complex_workspace, config)
    
    packages, workspace_modules = discovery.discover_workspace()
    
    # Should find both packages
    assert len(packages) == 2
    assert 'company.core' in packages
    assert 'company.analytics' in packages
    
    # Check workspace modules
    expected_workspace_modules = {
        'main',
        'apps.customer_service'
    }
    workspace_module_names = {mod.name for mod in workspace_modules}
    assert workspace_module_names == expected_workspace_modules
    
    # Verify package structure
    core_pkg = packages['company.core']
    assert len(core_pkg.modules) == 4  # validation, formatting, logging, config
    assert 'utils.validation' in core_pkg.modules
    assert 'utils.formatting' in core_pkg.modules
    
    analytics_pkg = packages['company.analytics']
    assert len(analytics_pkg.modules) == 4  # daily, monthly, performance, visualizations
    assert 'reports.daily' in analytics_pkg.modules
    assert 'metrics.performance' in analytics_pkg.modules


def test_cross_package_dependency_resolution(complex_workspace, core_utilities_package, analytics_package, customer_app):
    """Test dependency resolution across packages."""
    program, packages = load_workspace_program(complex_workspace)
    
    # Should load all modules from workspace and packages
    all_module_names = {module.name for module in program.modules}
    
    expected_modules = {
        # Workspace modules
        'main',
        'apps.customer_service',
        
        # Core package modules  
        'company.core::utils.validation',
        'company.core::utils.formatting',
        'company.core::logging', 
        'company.core::config',
        
        # Analytics package modules
        'company.analytics::reports.daily',
        'company.analytics::reports.monthly',
        'company.analytics::metrics.performance',
        'company.analytics::visualizations'
    }
    
    assert all_module_names == expected_modules
    
    # Verify dependencies are correctly resolved
    assert len(packages) == 2
    assert packages['company.analytics'].dependencies['company.core']


def test_hierarchical_module_imports(complex_workspace, core_utilities_package, analytics_package, customer_app):
    """Test hierarchical module import resolution."""
    program, packages = load_workspace_program(complex_workspace)
    
    # Resolve all cross-references
    resolved_program = resolve_program(program, packages)
    
    # Find modules that import from other packages
    daily_reports = None
    customer_service = None
    
    for module in resolved_program.modules:
        if module.name == 'company.analytics::reports.daily':
            daily_reports = module
        elif module.name == 'apps.customer_service':
            customer_service = module
    
    assert daily_reports is not None
    assert customer_service is not None
    
    # Check that use statements are properly resolved
    # Note: This would require examining the resolved import statements
    # in the actual AST, which depends on the resolver implementation


def test_workspace_info_complex(complex_workspace, core_utilities_package, analytics_package, customer_app):
    """Test workspace info for complex structure."""
    info = get_workspace_info(complex_workspace)
    
    # Workspace should have correct counts
    assert info['total_modules'] == 6  # 2 workspace + 4 per package
    assert info['total_packages'] == 2
    
    # Check package info
    assert 'company.core' in info['packages']
    assert 'company.analytics' in info['packages']
    
    core_info = info['packages']['company.core']
    assert core_info['version'] == '2.1.0'
    assert core_info['module_count'] == 4
    
    analytics_info = info['packages']['company.analytics'] 
    assert analytics_info['version'] == '1.5.0'
    assert analytics_info['module_count'] == 4
    assert analytics_info['dependency_count'] == 1


def test_module_export_system(complex_workspace, core_utilities_package, analytics_package):
    """Test module export visibility and access control."""
    config = load_workspace_config(complex_workspace)
    discovery = PackageDiscovery(complex_workspace, config)
    packages, _ = discovery.discover_workspace()
    
    core_pkg = packages['company.core']
    analytics_pkg = packages['company.analytics']
    
    # Core package exports specific modules
    core_manifest = core_pkg.manifest
    exports = core_manifest.module_exports
    
    assert 'utils.validation' in exports
    assert 'utils.formatting' in exports  
    assert 'logging' in exports
    # config module should not be exported (internal only)
    assert 'config' not in exports
    
    # Analytics package exports organized by category
    analytics_exports = analytics_pkg.manifest.module_exports
    assert 'reports' in analytics_exports
    assert 'metrics' in analytics_exports


def test_version_compatibility_checking(complex_workspace, core_utilities_package, analytics_package):
    """Test that version constraints are properly validated."""
    config = load_workspace_config(complex_workspace)
    discovery = PackageDiscovery(complex_workspace, config)
    packages, _ = discovery.discover_workspace()
    
    analytics_pkg = packages['company.analytics']
    core_dependency = analytics_pkg.manifest.dependencies['company.core']
    
    # Check constraint is parsed correctly
    assert str(core_dependency.constraint).startswith('>=2.0.0')
    
    # Core package version should satisfy constraint
    core_pkg = packages['company.core'] 
    constraint_satisfied = core_dependency.constraint.matches(core_pkg.manifest.version)
    assert constraint_satisfied


def test_circular_import_detection_complex(tmp_path):
    """Test circular import detection in complex scenarios."""
    workspace = tmp_path / "circular_complex"
    workspace.mkdir()
    
    # Create workspace with circular package dependencies  
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Package A depends on B
    pkg_a = packages_dir / "pkg_a"
    pkg_a.mkdir()
    
    manifest_a = dedent("""
        [package]
        name = "circular.a"
        version = "1.0.0"
        
        [dependencies]
        "circular.b" = "1.0.0"
    """)
    (pkg_a / "namel3ss.toml").write_text(manifest_a)
    (pkg_a / "main.ai").write_text('module main\nprompt "a" { template: "A" }')
    
    # Package B depends on C
    pkg_b = packages_dir / "pkg_b"
    pkg_b.mkdir()
    
    manifest_b = dedent("""
        [package]
        name = "circular.b" 
        version = "1.0.0"
        
        [dependencies]
        "circular.c" = "1.0.0"
    """)
    (pkg_b / "namel3ss.toml").write_text(manifest_b)
    (pkg_b / "main.ai").write_text('module main\nprompt "b" { template: "B" }')
    
    # Package C depends on A (creates cycle)
    pkg_c = packages_dir / "pkg_c"
    pkg_c.mkdir()
    
    manifest_c = dedent("""
        [package]
        name = "circular.c"
        version = "1.0.0"
        
        [dependencies]
        "circular.a" = "1.0.0"
    """)
    (pkg_c / "namel3ss.toml").write_text(manifest_c)
    (pkg_c / "main.ai").write_text('module main\nprompt "c" { template: "C" }')
    
    # Should detect circular dependency
    from namel3ss.packages.discovery import DependencyResolver
    from namel3ss.packages import DependencyCycleError
    
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    packages, _ = discovery.discover_workspace()
    
    resolver = DependencyResolver(packages)
    
    with pytest.raises(DependencyCycleError):
        resolver.resolve_dependencies(['circular.a'])


def test_full_workflow_integration(complex_workspace, core_utilities_package, analytics_package, customer_app):
    """Test complete workflow from workspace discovery to program execution."""
    # Load complete workspace
    program, packages = load_workspace_program(complex_workspace)
    
    # Resolve all imports and dependencies
    resolved_program = resolve_program(program, packages)
    
    # Verify the complete program structure
    assert resolved_program is not None
    assert len(resolved_program.modules) == 8  # 2 workspace + 6 package modules
    
    # All modules should be properly resolved
    for module in resolved_program.modules:
        assert module.name is not None
        assert len(module.name) > 0
    
    # Package system should be fully functional
    assert len(packages) == 2
    
    # Dependencies should be resolved in correct order
    from namel3ss.packages.discovery import DependencyResolver
    resolver = DependencyResolver(packages)
    resolved_deps = resolver.resolve_dependencies(['company.analytics'])
    
    # Core should come before analytics in dependency order
    resolved_names = [pkg.manifest.name for pkg in resolved_deps]
    core_idx = resolved_names.index('company.core')
    analytics_idx = resolved_names.index('company.analytics')
    assert core_idx < analytics_idx