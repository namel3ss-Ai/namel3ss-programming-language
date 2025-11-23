"""Test CLI package and module commands."""

import pytest
import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock, patch
from click.testing import CliRunner

from namel3ss.cli.commands.packages import package_cmd
from namel3ss.cli.commands.modules import module_cmd


@pytest.fixture
def cli_test_workspace(tmp_path):
    """Create workspace for CLI testing."""
    workspace = tmp_path / "cli_test"
    workspace.mkdir()
    
    # Create workspace config
    config_content = dedent("""
        [workspace]
        name = "cli-test"
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    # Create a sample package
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    pkg_dir = packages_dir / "sample_pkg"
    pkg_dir.mkdir()
    
    manifest_content = dedent("""
        [package]
        name = "sample.package"
        version = "1.0.0"
        description = "Sample package for testing"
        
        modules = ["main", "utils"]
    """)
    (pkg_dir / "namel3ss.toml").write_text(manifest_content)
    
    # Create modules
    main_content = dedent("""
        module main
        
        prompt "hello" {
            template: "Hello {{name}}"
        }
    """)
    (pkg_dir / "main.ai").write_text(main_content)
    
    utils_content = dedent("""
        module utils
        
        prompt "helper" {
            template: "Helper: {{text}}"
        }
    """)
    (pkg_dir / "utils.ai").write_text(utils_content)
    
    # Create workspace module
    workspace_content = dedent("""
        module app
        
        use sample.package::main
        
        app "CLI Test App" {
            description: "Test application"
        }
    """)
    (workspace / "app.ai").write_text(workspace_content)
    
    return workspace


def test_package_list_command(cli_test_workspace):
    """Test package list command."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        # Copy workspace to isolated filesystem
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        # Run package list command
        result = runner.invoke(package_cmd, ['list'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        assert 'sample.package' in result.output
        assert '1.0.0' in result.output


def test_package_list_json_output(cli_test_workspace):
    """Test package list with JSON output."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(package_cmd, ['list', '--json'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output_data = json.loads(result.output)
        assert 'packages' in output_data
        assert len(output_data['packages']) == 1
        
        package_info = output_data['packages'][0]
        assert package_info['name'] == 'sample.package'
        assert package_info['version'] == '1.0.0'


def test_package_info_command(cli_test_workspace):
    """Test package info command."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(package_cmd, ['info', 'sample.package'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        assert 'sample.package' in result.output
        assert '1.0.0' in result.output
        assert 'Sample package for testing' in result.output


def test_package_info_not_found(cli_test_workspace):
    """Test package info for non-existent package."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(package_cmd, ['info', 'nonexistent.package'], cwd=str(workspace_copy))
        
        assert result.exit_code != 0
        assert 'Package not found' in result.output


def test_package_deps_command(cli_test_workspace):
    """Test package deps command."""
    runner = CliRunner()
    
    # Create package with dependencies
    packages_dir = cli_test_workspace / "packages"
    dependent_pkg = packages_dir / "dependent_pkg"
    dependent_pkg.mkdir()
    
    manifest_content = dedent("""
        [package]
        name = "dependent.package"
        version = "1.0.0"
        
        [dependencies]
        "sample.package" = ">=1.0.0"
    """)
    (dependent_pkg / "namel3ss.toml").write_text(manifest_content)
    (dependent_pkg / "main.ai").write_text('module main\nprompt "test" { template: "Test" }')
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(package_cmd, ['deps', 'dependent.package'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        assert 'sample.package' in result.output


def test_package_check_command(cli_test_workspace):
    """Test package check command."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(package_cmd, ['check'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        assert 'Dependencies satisfied' in result.output or 'No issues found' in result.output


def test_module_list_command(cli_test_workspace):
    """Test module list command."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(module_cmd, ['list'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        assert 'app' in result.output  # workspace module
        assert 'sample.package::main' in result.output  # package module


def test_module_list_package_filter(cli_test_workspace):
    """Test module list with package filter."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(module_cmd, ['list', '--package', 'sample.package'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        assert 'sample.package::main' in result.output
        assert 'sample.package::utils' in result.output
        assert 'app' not in result.output  # workspace module should be filtered out


def test_module_info_command(cli_test_workspace):
    """Test module info command."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(module_cmd, ['info', 'sample.package::main'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        assert 'sample.package::main' in result.output
        assert 'prompt' in result.output


def test_module_deps_command(cli_test_workspace):
    """Test module deps command."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(module_cmd, ['deps', 'app'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        assert 'sample.package::main' in result.output


def test_module_graph_command(cli_test_workspace):
    """Test module graph command."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "test_workspace"
        shutil.copytree(cli_test_workspace, workspace_copy)
        
        result = runner.invoke(module_cmd, ['graph'], cwd=str(workspace_copy))
        
        assert result.exit_code == 0
        # Should show module dependency relationships


def test_package_command_no_workspace():
    """Test package commands outside workspace."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        # No namel3ss.toml file
        result = runner.invoke(package_cmd, ['list'])
        
        assert result.exit_code != 0
        assert 'No workspace found' in result.output or 'workspace root' in result.output


def test_cli_error_handling():
    """Test CLI error handling for various scenarios."""
    runner = CliRunner()
    
    # Test invalid package name
    result = runner.invoke(package_cmd, ['info', ''])
    assert result.exit_code != 0
    
    # Test invalid module name
    result = runner.invoke(module_cmd, ['info', ''])
    assert result.exit_code != 0


@pytest.fixture
def broken_workspace(tmp_path):
    """Create workspace with invalid package manifests."""
    workspace = tmp_path / "broken_test"
    workspace.mkdir()
    
    # Valid workspace config
    config_content = dedent("""
        [workspace]
        name = "broken-test"
        package_paths = ["packages"]
    """)
    (workspace / "namel3ss.toml").write_text(config_content)
    
    packages_dir = workspace / "packages"
    packages_dir.mkdir()
    
    # Create package with invalid manifest
    broken_pkg = packages_dir / "broken_pkg"
    broken_pkg.mkdir()
    
    # Invalid TOML syntax
    invalid_content = dedent("""
        [package
        name = "broken.package"  # Missing closing bracket
        version = "1.0.0"
    """)
    (broken_pkg / "namel3ss.toml").write_text(invalid_content)
    
    return workspace


def test_cli_broken_package_handling(broken_workspace):
    """Test CLI handling of broken package manifests."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        import shutil
        workspace_copy = Path.cwd() / "broken_workspace"
        shutil.copytree(broken_workspace, workspace_copy)
        
        result = runner.invoke(package_cmd, ['list'], cwd=str(workspace_copy))
        
        # Should handle errors gracefully
        assert result.exit_code != 0 or 'Error' in result.output or 'Failed' in result.output