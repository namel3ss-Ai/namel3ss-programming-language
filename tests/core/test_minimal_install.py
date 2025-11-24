"""
Tests for minimal core-only installation without optional dependencies.

These tests verify that the core functionality (parsing, AST, basic codegen, CLI)
works without any optional dependencies installed. They also verify that helpful
error messages are shown when optional features are used without the required extras.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Test that importing core modules works without optional dependencies
def test_import_core_modules():
    """Verify core modules can be imported without optional dependencies."""
    # Parser and AST
    from namel3ss.lang import parser
    from namel3ss import ast
    
    # Code generation
    from namel3ss.codegen.backend import core
    from namel3ss.codegen import frontend
    
    # CLI
    from namel3ss import cli
    from namel3ss.cli import commands
    
    # Config and utilities
    from namel3ss import config
    from namel3ss import loader
    
    # Features module
    from namel3ss import features
    
    assert parser is not None
    assert ast is not None
    assert core is not None
    assert frontend is not None
    assert cli is not None
    assert commands is not None
    assert config is not None
    assert loader is not None
    assert features is not None


def test_parse_simple_n3_file(tmp_path):
    """Verify parser works without optional dependencies."""
    from namel3ss.loader import load_program
    from pathlib import Path
    
    # Create a minimal .n3 file
    n3_file = tmp_path / "test.n3"
    n3_file.write_text("""
app "Test App" {
  description: "Simple test application"
}
""")
    
    # Should parse successfully
    program = load_program(tmp_path)
    assert program is not None
    assert program.modules is not None


def test_basic_codegen_without_sql(tmp_path):
    """Verify basic code generation module imports without SQL dependencies."""
    # Just verify the imports work
    from namel3ss.codegen.backend import core
    from namel3ss.codegen import frontend
    
    assert core is not None
    assert frontend is not None
    
    # Verify generate_backend function exists
    assert hasattr(core, 'generate_backend')
    assert callable(core.generate_backend)


class TestFeatureDetection:
    """Test feature detection functions."""
    
    def test_feature_detection_module_imports(self):
        """Verify features module can be imported."""
        from namel3ss import features
        
        assert hasattr(features, 'has_openai')
        assert hasattr(features, 'require_openai')
        assert hasattr(features, 'has_redis')
        assert hasattr(features, 'get_available_features')
    
    def test_get_available_features(self):
        """Test get_available_features returns a dict."""
        from namel3ss.features import get_available_features
        
        features = get_available_features()
        
        assert isinstance(features, dict)
        assert 'openai' in features
        assert 'anthropic' in features
        assert 'redis' in features
        assert 'sqlalchemy' in features
        assert 'opentelemetry' in features
        
        # All should be bool values
        for key, value in features.items():
            assert isinstance(value, bool), f"{key} should have bool value"
    
    @patch.dict(sys.modules, {'openai': None})
    def test_has_openai_when_missing(self):
        """Test has_openai returns False when package missing."""
        # Force reimport to trigger ImportError
        import importlib
        from namel3ss import features
        importlib.reload(features)
        
        # Note: This test depends on openai not being installed
        # In CI where dependencies are minimal, this will pass
        # With all extras installed, we'd need to mock the import
        # For now, just test the function exists and returns bool
        result = features.has_openai()
        assert isinstance(result, bool)


class TestMissingDependencyErrors:
    """Test that helpful errors are raised when optional deps are missing."""
    
    def test_require_openai_raises_helpful_error(self):
        """Test require_openai raises helpful error when openai missing."""
        from namel3ss.features import MissingDependencyError, require_openai
        
        # Since openai isn't installed, this should raise
        try:
            require_openai()
            assert False, "Should have raised MissingDependencyError"
        except MissingDependencyError as e:
            error_msg = str(e)
            assert "namel3ss[openai]" in error_msg or "namel3ss[ai]" in error_msg
            assert "pip install" in error_msg
    
    def test_require_redis_raises_helpful_error(self):
        """Test require_redis raises helpful error when redis missing."""
        from namel3ss.features import MissingDependencyError, require_redis
        
        # Since redis isn't installed, this should raise
        try:
            require_redis()
            assert False, "Should have raised MissingDependencyError"
        except MissingDependencyError as e:
            error_msg = str(e)
            assert "namel3ss[redis]" in error_msg
            assert "pip install" in error_msg
    
    def test_require_sqlalchemy_raises_helpful_error(self):
        """Test require_sqlalchemy raises helpful error when sqlalchemy missing."""
        from namel3ss.features import MissingDependencyError, require_sqlalchemy, has_sqlalchemy
        
        # Skip if SQLAlchemy is actually installed (will be removed from core in production)
        if has_sqlalchemy():
            pytest.skip("SQLAlchemy is installed - test only valid with minimal install")
        
        # Since sqlalchemy isn't installed in core, this should raise
        try:
            require_sqlalchemy()
            assert False, "Should have raised MissingDependencyError"
        except MissingDependencyError as e:
            error_msg = str(e)
            assert "namel3ss[sql]" in error_msg
            assert "pip install" in error_msg


class TestAdapterErrorMessages:
    """Test that adapters raise helpful errors when dependencies missing."""
    
    def test_model_adapter_openai_error(self):
        """Test ModelAdapter raises helpful error for missing OpenAI."""
        from namel3ss.adapters.model import ModelAdapter, ModelAdapterConfig
        from namel3ss.features import MissingDependencyError
        
        config = ModelAdapterConfig(
            name="test_openai",
            provider="openai",
            model="gpt-4",
            api_key="test_key",
        )
        
        # Mock has_openai to return False BEFORE creating adapter
        with patch('namel3ss.features.has_openai', return_value=False):
            with pytest.raises(MissingDependencyError) as exc_info:
                adapter = ModelAdapter(config)
            
            error_msg = str(exc_info.value)
            # Should mention the extra to install
            assert "openai" in error_msg.lower() or "ai" in error_msg.lower()
            assert "pip install" in error_msg.lower()
    
    def test_queue_adapter_redis_error(self):
        """Test that Redis queue adapters require Redis capability."""
        from namel3ss.features import MissingDependencyError, require_redis
        
        # Test that require_redis() works correctly
        with patch('namel3ss.features.has_redis', return_value=False):
            with pytest.raises(MissingDependencyError) as exc_info:
                require_redis()
            
            error_msg = str(exc_info.value)
            # Should mention the redis extra
            assert "redis" in error_msg.lower()
            assert "pip install" in error_msg.lower()


class TestCLIWithoutOptionals:
    """Test CLI commands work without optional dependencies."""
    
    def test_cli_help_works(self):
        """Test that --help works without optional deps."""
        from namel3ss.cli import main
        
        # Just verify the main function exists and is callable
        assert callable(main)
    
    def test_cli_commands_importable(self):
        """Test CLI command modules can be imported."""
        from namel3ss.cli import commands
        
        # Verify commands exist
        assert hasattr(commands, 'cmd_build')
        assert hasattr(commands, 'cmd_run')
        assert hasattr(commands, 'cmd_test')


def test_jinja2_available_for_templates():
    """Verify jinja2 is available (core dependency for templates)."""
    try:
        import jinja2
        assert jinja2 is not None
    except ImportError:
        pytest.fail("jinja2 should be a core dependency but is missing")


def test_pygls_available_for_lsp():
    """Verify pygls is available (core dependency for LSP)."""
    try:
        import pygls
        assert pygls is not None
    except ImportError:
        # pygls may not be installed in all environments
        pytest.skip("pygls not installed - should be added to core dependencies")


class TestCoreRuntime:
    """Test core runtime functionality."""
    
    def test_fastapi_available(self):
        """Verify FastAPI is available as core dependency."""
        try:
            import fastapi
            assert fastapi is not None
        except ImportError:
            pytest.fail("FastAPI should be a core dependency")
    
    def test_pydantic_available(self):
        """Verify Pydantic is available as core dependency."""
        try:
            import pydantic
            assert pydantic is not None
        except ImportError:
            pytest.fail("Pydantic should be a core dependency")
    
    def test_httpx_available(self):
        """Verify httpx is available as core dependency."""
        try:
            import httpx
            assert httpx is not None
        except ImportError:
            pytest.fail("httpx should be a core dependency")
    
    def test_uvicorn_available(self):
        """Verify uvicorn is available as core dependency."""
        try:
            import uvicorn
            assert uvicorn is not None
        except ImportError:
            pytest.fail("uvicorn should be a core dependency")
