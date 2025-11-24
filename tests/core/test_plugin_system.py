"""
Tests for the Namel3ss plugin system.

Tests cover:
- Plugin manifest parsing and validation  
- Plugin AST node functionality
- Basic plugin manager operations
"""

import asyncio
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from typing import Any, Dict

import pytest
import toml

# Test plugin manifest functionality
def test_plugin_manifest_basic():
    """Test basic plugin manifest creation and validation."""
    try:
        from namel3ss.plugins.manifest import (
            PluginManifest, PluginType, SemVer, 
            PluginEntryPoint, PluginEntryPointType, PluginAuthor
        )
        
        # Test data with all required fields
        manifest_data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "plugin_types": {"connector"},
            "authors": [{"name": "Test Author"}],
            "license": "MIT",
            "entry_points": {
                "main": {
                    "type": "factory",
                    "module": "test_plugin.main",
                }
            }
        }
        
        manifest = PluginManifest(**manifest_data)
        
        assert manifest.name == "test-plugin"
        assert manifest.version == "1.0.0"
        assert PluginType.CONNECTOR in manifest.plugin_types
        assert manifest.description == "A test plugin"
        assert manifest.license == "MIT"
        assert len(manifest.authors) == 1
        assert manifest.authors[0].name == "Test Author"
        assert "main" in manifest.entry_points
        
    except ImportError as e:
        pytest.skip(f"Plugin manifest module not available: {e}")


def test_semver_functionality():
    """Test semantic versioning functionality."""
    try:
        from namel3ss.plugins.manifest import SemVer
        
        v1 = SemVer.parse("1.0.0")
        v2 = SemVer.parse("1.0.1") 
        v3 = SemVer.parse("1.1.0")
        
        assert v1 < v2
        assert v2 < v3
        assert v1.satisfies(">=1.0.0")
        assert v2.satisfies("^1.0.0")
        
    except ImportError:
        pytest.skip("SemVer module not available")


def test_plugin_ast_nodes():
    """Test plugin AST node creation."""
    try:
        from namel3ss.ast.plugins import PluginReference, PluginUsage
        
        # Test PluginReference
        plugin_ref = PluginReference(
            name="test-plugin",
            version_constraint="^1.0.0",
            alias="tp"
        )
        
        assert plugin_ref.name == "test-plugin"
        assert plugin_ref.version_constraint == "^1.0.0"
        assert plugin_ref.alias == "tp"
        
        # Test PluginUsage
        usage = PluginUsage(
            plugin_ref=plugin_ref,  # Use plugin_ref instead of plugin
        )
        
        assert usage.plugin_ref.name == "test-plugin"
        assert usage.scope == "global"  # Default value
        assert usage.required == True   # Default value
        
    except ImportError:
        pytest.skip("Plugin AST modules not available")


def test_plugin_ir_specs():
    """Test plugin IR specification creation.""" 
    try:
        from namel3ss.ir.plugins import PluginSpec, PluginType as IRPluginType
        
        spec = PluginSpec(
            name="ir-plugin",
            version_constraint=">=1.0.0",
            plugin_types={IRPluginType.TOOL},
            required_capabilities={"database.read"}
        )
        
        assert spec.name == "ir-plugin"
        assert spec.version_constraint == ">=1.0.0"
        assert IRPluginType.TOOL in spec.plugin_types
        assert "database.read" in spec.required_capabilities
        
    except ImportError:
        pytest.skip("Plugin IR modules not available")


def test_plugin_manifest_from_toml(tmp_path):
    """Test loading plugin manifest from TOML file."""
    try:
        from namel3ss.plugins.manifest import load_plugin_manifest
        import toml
        
        # Create test manifest file with required fields in [plugin] section
        manifest_data = {
            "plugin": {
                "name": "file-plugin",
                "version": "0.1.0", 
                "description": "Plugin from file",
                "plugin_types": ["tool"],
                "authors": [{"name": "Test Author"}],
                "license": "MIT",
                "entry_points": {
                    "main": {
                        "type": "factory",
                        "module": "file_plugin.main"
                    }
                }
            }
        }
        
        manifest_file = tmp_path / "n3-plugin.toml"
        with manifest_file.open("w") as f:
            toml.dump(manifest_data, f)
        
        manifest = load_plugin_manifest(manifest_file)
        assert manifest.name == "file-plugin"
        assert manifest.version == "0.1.0"
        assert manifest.description == "Plugin from file"
        
    except ImportError:
        pytest.skip("Plugin manifest loading not available")


def test_plugin_manager_basic():
    """Test basic plugin manager functionality."""
    try:
        from namel3ss.plugins.manager import PluginManager
        
        manager = PluginManager()
        
        # Test basic manager creation
        assert manager is not None
        assert hasattr(manager, 'discover_plugins')
        assert hasattr(manager, 'load_plugin')
        
        # Test empty state
        plugins = manager.get_all_plugins()
        assert isinstance(plugins, list)
        
    except ImportError:
        pytest.skip("Plugin manager not available")


def test_enhanced_tool_with_plugin():
    """Test enhanced tool AST node with plugin support."""
    try:
        from namel3ss.ast.plugins import PluginReference, EnhancedTool
        
        plugin_ref = PluginReference(name="tool-plugin")
        
        tool = EnhancedTool(
            name="enhanced_tool",
            tool_type="plugin",
            plugin_ref=plugin_ref,  # Use plugin_ref instead of plugin_reference
            config={"timeout": 30}   # Use config instead of plugin_config
        )
        
        assert tool.name == "enhanced_tool"
        assert tool.plugin_ref.name == "tool-plugin"
        assert tool.config["timeout"] == 30
        
    except ImportError:
        pytest.skip("Enhanced tool AST not available")


@pytest.mark.asyncio
async def test_plugin_registry_client_basic():
    """Test basic registry client functionality."""
    try:
        from namel3ss.plugins.registry_client import RegistryClient
        
        # Test client creation
        client = RegistryClient(backends={})
        
        assert client is not None
        assert hasattr(client, 'search_plugins')
        assert hasattr(client, 'download_plugin')
        
    except ImportError:
        pytest.skip("Registry client not available")


def test_plugin_codegen_basic():
    """Test basic plugin codegen functionality."""
    try:
        from namel3ss.codegen.plugins import PluginAwareCodegen
        from namel3ss.plugins.manager import PluginManager
        
        # Mock plugin manager
        mock_manager = Mock(spec=PluginManager)
        
        # Test codegen creation
        codegen = PluginAwareCodegen(mock_manager)
        
        assert codegen is not None
        assert hasattr(codegen, 'generate_backend')
        assert codegen.plugin_manager is mock_manager
        
    except ImportError:
        pytest.skip("Plugin codegen not available")


def test_plugin_manifest_validation():
    """Test plugin manifest validation with invalid data."""
    try:
        from namel3ss.plugins.manifest import PluginManifest
        from pydantic import ValidationError
        
        # Test invalid version
        with pytest.raises(ValidationError):
            PluginManifest(
                name="test-plugin",
                version="not-a-version",  # Invalid semantic version
                description="Test plugin",
                plugin_type="connector"
            )
            
        # Test invalid plugin type  
        with pytest.raises(ValidationError):
            PluginManifest(
                name="test-plugin",
                version="1.0.0",
                description="Test plugin", 
                plugin_type="invalid_type"  # Invalid plugin type
            )
            
    except ImportError:
        pytest.skip("Plugin manifest validation not available")


def test_backend_ir_with_plugins():
    """Test BackendIR with plugin extensions."""
    try:
        from namel3ss.ir.plugins import BackendIRWithPlugins, PluginRequirementSpec, PluginSpec
        
        # Create plugin requirements
        requirements = PluginRequirementSpec(
            required_plugins=[
                PluginSpec(name="required-plugin")
            ],
            optional_plugins=[
                PluginSpec(name="optional-plugin") 
            ]
        )
        
        # Create backend IR
        ir = BackendIRWithPlugins(plugin_requirements=requirements)
        
        assert len(ir.get_required_plugins()) == 1
        assert "required-plugin" in ir.get_required_plugins()
        assert ir.has_plugin_dependencies()
        
    except ImportError:
        pytest.skip("Plugin IR not available")


def test_plugin_ir_validation():
    """Test plugin IR validation functionality."""
    try:
        from namel3ss.ir.plugins import (
            BackendIRWithPlugins, PluginRequirementSpec, PluginSpec,
            EnhancedToolSpec, validate_plugin_ir
        )
        from namel3ss.ir.spec import TypeSpec
        
        # Create IR with undeclared plugin reference
        plugin_spec = PluginSpec(name="undeclared-plugin")
        tool_spec = EnhancedToolSpec(
            name="test_tool",
            input_schema=TypeSpec(kind="object"),
            output_schema=TypeSpec(kind="object"),
            plugin_spec=plugin_spec
        )
        
        ir = BackendIRWithPlugins(
            enhanced_tools={"test_tool": tool_spec},
            plugin_requirements=PluginRequirementSpec()  # Empty requirements
        )
        
        # Should have validation errors
        errors = validate_plugin_ir(ir)
        assert len(errors) > 0
        assert any("undeclared plugin" in error.lower() for error in errors)
        
    except ImportError:
        pytest.skip("Plugin IR validation not available")


def test_complete_plugin_workflow():
    """Test a complete plugin workflow from manifest to usage."""
    try:
        from namel3ss.plugins.manifest import PluginManifest, PluginType
        from namel3ss.ast.plugins import PluginReference, PluginUsage
        from namel3ss.ir.plugins import PluginSpec
        
        # 1. Create plugin manifest with all required fields
        manifest = PluginManifest(
            name="workflow-plugin",
            version="1.0.0",
            description="Workflow test plugin", 
            plugin_types={PluginType.TOOL},
            authors=[{"name": "Test Author"}],
            license="MIT",
            entry_points={
                "main": {
                    "type": "factory",
                    "module": "workflow_plugin.main"
                }
            }
        )
        
        # 2. Create AST nodes
        plugin_ref = PluginReference(
            name="workflow-plugin",
            version_constraint="^1.0.0"
        )
        
        plugin_usage = PluginUsage(
            plugin_ref=plugin_ref,
            scope="global",
            required=True
        )
        
        # 3. Create IR spec
        plugin_spec = PluginSpec(
            name="workflow-plugin",
            version_constraint="^1.0.0"
        )
        
        # Verify the workflow
        assert manifest.name == plugin_ref.name == plugin_spec.name
        assert plugin_usage.scope == "global"
        
    except ImportError:
        pytest.skip("Complete plugin workflow modules not available")


# Additional integration tests can be added here as needed
if __name__ == "__main__":
    pytest.main([__file__, "-v"])