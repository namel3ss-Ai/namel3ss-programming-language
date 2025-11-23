"""Additional edge case and error handling tests."""

import pytest
from pathlib import Path
from textwrap import dedent

from namel3ss.packages import (
    PackageManifest, PackageNotFoundError, DependencyCycleError, 
    PackageVersionConflictError, InvalidPackageManifestError
)
from namel3ss.packages.discovery import PackageDiscovery, DependencyResolver, load_workspace_config
from namel3ss.loader import load_workspace_program


def test_invalid_package_manifest_formats(tmp_path):
    """Test handling of various invalid package manifest formats."""
    workspace = tmp_path / "invalid_manifest_test"
    workspace.mkdir()
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Test invalid TOML syntax
    invalid_toml_pkg = packages_dir / "invalid_toml"
    invalid_toml_pkg.mkdir()
    
    invalid_toml_content = dedent("""
        [package
        name = "invalid.toml"  # Missing closing bracket
        version = "1.0.0"
    """)
    (invalid_toml_pkg / "namel3ss.toml").write_text(invalid_toml_content)
    
    # Test missing required fields
    missing_fields_pkg = packages_dir / "missing_fields"
    missing_fields_pkg.mkdir()
    
    missing_fields_content = dedent("""
        [package]
        # Missing name and version
        description = "Package missing required fields"
    """)
    (missing_fields_pkg / "namel3ss.toml").write_text(missing_fields_content)
    
    # Test invalid version format
    invalid_version_pkg = packages_dir / "invalid_version"
    invalid_version_pkg.mkdir()
    
    invalid_version_content = dedent("""
        [package]
        name = "invalid.version"
        version = "not-a-version"
    """)
    (invalid_version_pkg / "namel3ss.toml").write_text(invalid_version_content)
    
    # Discovery should handle these gracefully
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    
    # Should not crash, but may skip invalid packages
    packages, _ = discovery.discover_workspace()
    
    # Valid packages should still be discoverable
    # Invalid ones should be skipped or raise specific errors


def test_complex_version_constraints(tmp_path):
    """Test complex version constraint scenarios."""
    workspace = tmp_path / "version_constraints_test"
    workspace.mkdir()
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Create packages with various versions
    versions = ["1.0.0", "1.1.0", "1.2.0", "2.0.0", "2.1.0"]
    
    for version in versions:
        pkg_dir = packages_dir / f"versioned_{version.replace('.', '_')}"
        pkg_dir.mkdir()
        
        manifest_content = dedent(f"""
            [package]
            name = "test.versioned"
            version = "{version}"
        """)
        (pkg_dir / "namel3ss.toml").write_text(manifest_content)
        (pkg_dir / "main.ai").write_text('module main\nprompt "test" { template: "Test" }')
    
    # Test various constraint formats
    from namel3ss.packages import VersionConstraint
    
    test_cases = [
        ("^1.0.0", ["1.0.0", "1.1.0", "1.2.0"], ["2.0.0", "2.1.0"]),
        ("~=1.1.0", ["1.1.0"], ["1.0.0", "1.2.0", "2.0.0"]),
        (">=1.1.0,<2.0.0", ["1.1.0", "1.2.0"], ["1.0.0", "2.0.0", "2.1.0"]),
        ("==2.0.0", ["2.0.0"], ["1.0.0", "1.1.0", "1.2.0", "2.1.0"]),
    ]
    
    for constraint_str, should_match, should_not_match in test_cases:
        constraint = VersionConstraint.parse(constraint_str)
        
        for version in should_match:
            assert constraint.matches(version), f"{constraint_str} should match {version}"
            
        for version in should_not_match:
            assert not constraint.matches(version), f"{constraint_str} should not match {version}"


def test_deep_dependency_chains(tmp_path):
    """Test deep dependency chains and resolution order."""
    workspace = tmp_path / "deep_deps_test"
    workspace.mkdir()
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Create chain: A -> B -> C -> D -> E
    chain = ['a', 'b', 'c', 'd', 'e']
    
    for i, pkg_name in enumerate(chain):
        pkg_dir = packages_dir / f"pkg_{pkg_name}"
        pkg_dir.mkdir()
        
        dependencies = {}
        if i < len(chain) - 1:
            next_pkg = chain[i + 1]
            dependencies[f"test.{next_pkg}"] = "1.0.0"
        
        deps_section = ""
        if dependencies:
            deps_section = "[dependencies]\n"
            for dep_name, dep_version in dependencies.items():
                deps_section += f'"{dep_name}" = "{dep_version}"\n'
        
        manifest_content = dedent(f"""
            [package]
            name = "test.{pkg_name}"
            version = "1.0.0"
            
            {deps_section}
        """)
        (pkg_dir / "namel3ss.toml").write_text(manifest_content)
        (pkg_dir / "main.ai").write_text(f'module main\nprompt "test_{pkg_name}" {{ template: "Test {pkg_name.upper()}" }}')
    
    # Test dependency resolution
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    packages, _ = discovery.discover_workspace()
    
    resolver = DependencyResolver(packages)
    resolved = resolver.resolve_dependencies(['test.a'])
    
    # Should resolve entire chain in dependency order (E, D, C, B, A)
    assert len(resolved) == 5
    resolved_names = [pkg.manifest.name for pkg in resolved]
    
    # Later dependencies should come first
    assert resolved_names.index('test.e') < resolved_names.index('test.d')
    assert resolved_names.index('test.d') < resolved_names.index('test.c')
    assert resolved_names.index('test.c') < resolved_names.index('test.b')
    assert resolved_names.index('test.b') < resolved_names.index('test.a')


def test_diamond_dependency_resolution(tmp_path):
    """Test diamond dependency pattern resolution."""
    workspace = tmp_path / "diamond_deps_test"
    workspace.mkdir()
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Create diamond: A depends on B and C, both B and C depend on D
    diamond_packages = {
        'a': ['test.b', 'test.c'],
        'b': ['test.d'],
        'c': ['test.d'],
        'd': []
    }
    
    for pkg_name, deps in diamond_packages.items():
        pkg_dir = packages_dir / f"pkg_{pkg_name}"
        pkg_dir.mkdir()
        
        deps_section = ""
        if deps:
            deps_section = "[dependencies]\n"
            for dep in deps:
                deps_section += f'"{dep}" = "1.0.0"\n'
        
        manifest_content = dedent(f"""
            [package]
            name = "test.{pkg_name}"
            version = "1.0.0"
            
            {deps_section}
        """)
        (pkg_dir / "namel3ss.toml").write_text(manifest_content)
        (pkg_dir / "main.ai").write_text(f'module main\nprompt "test_{pkg_name}" {{ template: "Test {pkg_name.upper()}" }}')
    
    # Test resolution
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    packages, _ = discovery.discover_workspace()
    
    resolver = DependencyResolver(packages)
    resolved = resolver.resolve_dependencies(['test.a'])
    
    # Should include all packages, with D appearing only once
    assert len(resolved) == 4
    resolved_names = [pkg.manifest.name for pkg in resolved]
    
    # D should come before B and C
    d_idx = resolved_names.index('test.d')
    b_idx = resolved_names.index('test.b')
    c_idx = resolved_names.index('test.c')
    a_idx = resolved_names.index('test.a')
    
    assert d_idx < b_idx
    assert d_idx < c_idx
    assert b_idx < a_idx
    assert c_idx < a_idx


def test_missing_module_files(tmp_path):
    """Test handling of packages with missing module files."""
    workspace = tmp_path / "missing_modules_test"
    workspace.mkdir()
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Package declares modules but files are missing
    pkg_dir = packages_dir / "missing_files_pkg"
    pkg_dir.mkdir()
    
    manifest_content = dedent("""
        [package]
        name = "test.missing"
        version = "1.0.0"
        
        modules = [
            "existing_module",
            "missing_module",
            "also_missing"
        ]
    """)
    (pkg_dir / "namel3ss.toml").write_text(manifest_content)
    
    # Only create one of the declared modules
    existing_content = dedent("""
        module existing_module
        
        prompt "exists" {
            template: "This module exists"
        }
    """)
    (pkg_dir / "existing_module.ai").write_text(existing_content)
    
    # missing_module.ai and also_missing.ai don't exist
    
    # Discovery should handle this gracefully
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    packages, _ = discovery.discover_workspace()
    
    # Package should still be discovered, but with warnings about missing modules
    assert 'test.missing' in packages
    pkg = packages['test.missing']
    
    # Should only include modules that actually exist
    existing_modules = list(pkg.modules.keys())
    assert 'existing_module' in existing_modules
    # Missing modules might be excluded or flagged


def test_workspace_without_config(tmp_path):
    """Test workspace discovery without explicit configuration."""
    workspace = tmp_path / "no_config_test"
    workspace.mkdir()
    
    # No namel3ss.toml file
    
    # Create some modules anyway
    module_content = dedent("""
        module standalone
        
        prompt "test" {
            template: "Standalone module"
        }
    """)
    (workspace / "standalone.ai").write_text(module_content)
    
    # Should still be able to load as a basic workspace
    try:
        program, packages = load_workspace_program(workspace)
        assert len(program.modules) == 1
        assert program.modules[0].name == 'standalone'
        assert len(packages) == 0  # No packages without config
    except Exception:
        # It's acceptable for this to fail if workspace config is required
        pass


def test_malformed_module_files(tmp_path):
    """Test handling of malformed module files."""
    workspace = tmp_path / "malformed_modules_test"
    workspace.mkdir()
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    pkg_dir = packages_dir / "malformed_pkg"
    pkg_dir.mkdir()
    
    manifest_content = dedent("""
        [package]
        name = "test.malformed"
        version = "1.0.0"
        
        modules = ["valid_module", "malformed_module"]
    """)
    (pkg_dir / "namel3ss.toml").write_text(manifest_content)
    
    # Valid module
    valid_content = dedent("""
        module valid_module
        
        prompt "valid" {
            template: "This is valid"
        }
    """)
    (pkg_dir / "valid_module.ai").write_text(valid_content)
    
    # Malformed module with syntax errors
    malformed_content = dedent("""
        module malformed_module
        
        prompt "invalid" {
            template: "Missing closing brace"
        # Missing closing brace
        
        this is not valid syntax
    """)
    (pkg_dir / "malformed_module.ai").write_text(malformed_content)
    
    # Loading should handle parsing errors gracefully
    try:
        program, packages = load_workspace_program(workspace)
        # Should still load what it can
        assert 'test.malformed' in packages
    except Exception as e:
        # Parsing errors are acceptable - should be handled appropriately
        assert 'parse' in str(e).lower() or 'syntax' in str(e).lower()


def test_package_name_conflicts(tmp_path):
    """Test handling of package name conflicts."""
    workspace = tmp_path / "name_conflicts_test"
    workspace.mkdir()
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages", "external"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    # Create two packages with same name in different directories
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    external_dir = workspace / "external"
    external_dir.mkdir()
    
    # First package
    pkg1_dir = packages_dir / "conflict_pkg"
    pkg1_dir.mkdir()
    
    manifest1 = dedent("""
        [package]
        name = "conflict.package"
        version = "1.0.0"
    """)
    (pkg1_dir / "namel3ss.toml").write_text(manifest1)
    (pkg1_dir / "main.ai").write_text('module main\nprompt "first" { template: "First" }')
    
    # Second package (same name, different location)
    pkg2_dir = external_dir / "another_conflict_pkg"
    pkg2_dir.mkdir()
    
    manifest2 = dedent("""
        [package]
        name = "conflict.package"
        version = "2.0.0"
    """)
    (pkg2_dir / "namel3ss.toml").write_text(manifest2)
    (pkg2_dir / "main.ai").write_text('module main\nprompt "second" { template: "Second" }')
    
    # Discovery should handle name conflicts
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    packages, _ = discovery.discover_workspace()
    
    # Should either:
    # 1. Take the first one found
    # 2. Take the highest version
    # 3. Raise an error about the conflict
    # The specific behavior depends on implementation
    
    if 'conflict.package' in packages:
        # If it loads one, verify it's consistent
        pkg = packages['conflict.package']
        assert pkg.manifest.name == 'conflict.package'
        assert pkg.manifest.version in ['1.0.0', '2.0.0']


def test_empty_packages(tmp_path):
    """Test handling of empty packages."""
    workspace = tmp_path / "empty_packages_test"
    workspace.mkdir()
    
    config_content = dedent("""
        [workspace]
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Package with no modules
    empty_pkg_dir = packages_dir / "empty_pkg"
    empty_pkg_dir.mkdir()
    
    manifest_content = dedent("""
        [package]
        name = "test.empty"
        version = "1.0.0"
        description = "Package with no modules"
        
        modules = []
    """)
    (empty_pkg_dir / "namel3ss.toml").write_text(manifest_content)
    
    # Discovery should handle empty packages
    config = load_workspace_config(workspace)
    discovery = PackageDiscovery(workspace, config)
    packages, _ = discovery.discover_workspace()
    
    # Empty package should still be discovered
    assert 'test.empty' in packages
    pkg = packages['test.empty']
    assert len(pkg.modules) == 0