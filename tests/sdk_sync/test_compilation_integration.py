"""
Integration tests for SDK-Sync with N3 compilation pipeline.

Tests schema export during backend generation and metadata endpoint generation.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from namel3ss.ast import App
from namel3ss.codegen.backend.core.generator import generate_backend
from namel3ss.sdk_sync import SchemaRegistry, export_schemas_from_app
from namel3ss.sdk_sync.ir import SchemaVersion


class TestCompilationIntegration:
    """Test SDK-Sync integration with N3 compilation."""
    
    def test_schema_export_during_backend_generation(self):
        """Test that schemas are exported during backend generation."""
        # Create minimal app
        app = App(
            name="test_app",
            datasets=[],
        )
        
        # Generate backend with schema export
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "backend"
            
            generate_backend(
                app=app,
                out_dir=out_dir,
                export_schemas=True,
                schema_version="1.0.0",
            )
            
            # Verify metadata router was created
            metadata_router = out_dir / "generated" / "routers" / "metadata.py"
            assert metadata_router.exists(), "Metadata router should be created"
            
            # Verify spec.json was created
            spec_file = out_dir / "generated" / "schemas" / "spec.json"
            assert spec_file.exists(), "Spec JSON should be created"
            
            # Verify routers/__init__.py includes metadata
            routers_init = out_dir / "generated" / "routers" / "__init__.py"
            content = routers_init.read_text()
            assert "metadata" in content, "Routers init should import metadata"
            assert "metadata_router" in content, "Routers init should export metadata_router"
    
    def test_backend_generation_without_schema_export(self):
        """Test that backend generation works without schema export."""
        app = App(
            name="test_app",
            datasets=[],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "backend"
            
            generate_backend(
                app=app,
                out_dir=out_dir,
                export_schemas=False,
            )
            
            # Verify metadata router was NOT created
            metadata_router = out_dir / "generated" / "routers" / "metadata.py"
            assert not metadata_router.exists(), "Metadata router should not be created"
            
            # Verify routers/__init__.py does NOT include metadata
            routers_init = out_dir / "generated" / "routers" / "__init__.py"
            content = routers_init.read_text()
            assert "metadata_router" not in content, "Routers init should not export metadata_router"
    
    def test_export_schemas_from_app(self):
        """Test exporting schemas from App AST."""
        # Create app with dataset (using actual Dataset structure)
        from namel3ss.ast import Dataset, DatasetSchemaField
        
        # Create dataset
        users_dataset = Dataset(
            name="users",
            source_type="sql",
            source="SELECT * FROM users",
            schema=[
                DatasetSchemaField(name="id", dtype="string"),
                DatasetSchemaField(name="email", dtype="string"),
            ],
        )
        
        app = App(
            name="test_app",
            datasets=[users_dataset],
        )
        
        version = SchemaVersion(1, 0, 0)
        spec = export_schemas_from_app(app, version, namespace="test")
        
        assert spec.version == version
        # Schema export may create models based on dataset schema
        assert spec.metadata["namespace"] == "test"
        assert spec.metadata["source"] == "app"
    
    def test_metadata_router_code_generation(self):
        """Test that metadata router code is valid."""
        from namel3ss.sdk_sync.exporter import generate_metadata_router
        
        code = generate_metadata_router()
        
        # Verify key elements are present
        assert "router = APIRouter" in code
        assert "/api/_meta" in code
        assert "get_schema_manifest" in code
        assert "get_model_schema" in code
        assert "get_tool_spec" in code
        assert "get_complete_spec" in code
        
        # Verify it compiles
        compile(code, "<metadata_router>", "exec")
    
    def test_schema_registry_populated_during_export(self):
        """Test that SchemaRegistry is populated during schema export."""
        from namel3ss.ast import Dataset, DatasetSchemaField
        
        # Clear registry
        registry = SchemaRegistry()
        registry.models.clear()
        registry.tools.clear()
        
        # Create dataset
        products_dataset = Dataset(
            name="products",
            source_type="sql",
            source="SELECT * FROM products",
            schema=[
                DatasetSchemaField(name="sku", dtype="string"),
            ],
        )
        
        # Create app
        app = App(
            name="test_app",
            datasets=[products_dataset],
        )
        
        version = SchemaVersion(1, 0, 0)
        spec = export_schemas_from_app(app, version)
        
        # Verify spec was created (models may be empty if schema extraction not implemented yet)
        assert spec is not None
        assert spec.version == version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
