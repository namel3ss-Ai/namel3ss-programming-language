"""Test configuration and shared fixtures for the package system tests."""

import pytest
from pathlib import Path


def pytest_configure(config):
    """Configure pytest for package system tests."""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (deselect with '-m \"not performance\"')"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace for testing."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Add performance marker to performance tests
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add edge_cases marker to edge case tests
        if "edge_cases" in item.nodeid:
            item.add_marker(pytest.mark.edge_cases)


# Shared test utilities
class TestHelpers:
    """Helper functions for package system tests."""
    
    @staticmethod
    def create_simple_package(workspace_path, package_name, version="1.0.0", dependencies=None):
        """Create a simple test package."""
        from textwrap import dedent
        
        packages_dir = workspace_path / "packages"
        packages_dir.mkdir(exist_ok=True)
        
        pkg_dir = packages_dir / package_name.replace(".", "_")
        pkg_dir.mkdir()
        
        deps_section = ""
        if dependencies:
            deps_section = "[dependencies]\n"
            for dep_name, dep_version in dependencies.items():
                deps_section += f'"{dep_name}" = "{dep_version}"\n'
        
        manifest_content = dedent(f"""
            [package]
            name = "{package_name}"
            version = "{version}"
            description = "Test package {package_name}"
            
            {deps_section}
            
            modules = ["main"]
        """)
        (pkg_dir / "namel3ss.toml").write_text(manifest_content)
        
        module_content = dedent(f"""
            module main
            
            prompt "test_prompt" {{
                template: "Test prompt for {package_name}"
            }}
        """)
        (pkg_dir / "main.ai").write_text(module_content)
        
        return pkg_dir
    
    @staticmethod
    def create_workspace_config(workspace_path, **kwargs):
        """Create a workspace configuration file."""
        from textwrap import dedent
        
        name = kwargs.get('name', 'test-workspace')
        module_paths = kwargs.get('module_paths', ['.'])
        package_paths = kwargs.get('package_paths', ['packages'])
        
        config_content = dedent(f"""
            [workspace]
            name = "{name}"
            module_paths = {module_paths}
            package_paths = {package_paths}
        """)
        (workspace_path / "namel3ss.toml").write_text(config_content)
        return workspace_path / "namel3ss.toml"


@pytest.fixture
def test_helpers():
    """Provide test helper utilities."""
    return TestHelpers


# Performance test configuration
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        'max_discovery_time': 10.0,
        'max_resolution_time': 5.0,
        'max_loading_time': 15.0,
        'max_memory_usage_mb': 500.0
    }


# Integration test fixtures
@pytest.fixture
def sample_workspace_structure():
    """Define a standard workspace structure for testing."""
    return {
        'workspace_modules': ['main', 'utils', 'config'],
        'packages': {
            'core.package': {
                'version': '1.0.0',
                'modules': ['validation', 'formatting'],
                'dependencies': {}
            },
            'analytics.package': {
                'version': '1.5.0',
                'modules': ['reports', 'metrics'],
                'dependencies': {'core.package': '>=1.0.0'}
            }
        }
    }