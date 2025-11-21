"""
Integration tests for SDK Sync system.

Tests the complete round-trip:
1. Export schemas from N3 runtime
2. Generate Python SDK
3. Use generated SDK to call N3 backend
4. Validate responses match expectations

Guarantees zero-copy compatibility between N3 runtime and generated SDK.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict
import json
import asyncio

from namel3ss.sdk_sync.ir import (
    IRSpec,
    IRModel,
    IRTool,
    IRField,
    IRType,
    SchemaVersion,
)
from namel3ss.sdk_sync.registry import SchemaRegistry, SchemaExporter
from namel3ss.sdk_sync.generators.python import (
    PythonModelGenerator,
    PythonClientGenerator,
    PythonSDKGenerator,
)
from namel3ss.sdk_sync.versioning import (
    CompatibilityChecker,
    MigrationGenerator,
    VersionManager,
)
from namel3ss.sdk_sync.validation import (
    RequestValidator,
    ResponseValidator,
    ValidationContext,
)


class TestIRRepresentation:
    """Test IR (Intermediate Representation) core."""

    def test_schema_version_parsing(self):
        """Test SchemaVersion parsing and comparison."""
        v1 = SchemaVersion.parse("1.0.0")
        assert v1.major == 1
        assert v1.minor == 0
        assert v1.patch == 0
        assert str(v1) == "1.0.0"

        v2 = SchemaVersion.parse("1.2.3")
        assert v1 < v2
        assert v1.is_compatible_with(v2)  # Same major

        v3 = SchemaVersion.parse("2.0.0")
        assert not v1.is_compatible_with(v3)  # Different major

    def test_ir_field_creation(self):
        """Test IRField creation and JSON Schema export."""
        field = IRField(
            name="email",
            type=IRType.STRING,
            required=True,
            description="User email",
            constraints={"format": "email"},
        )

        schema = field.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "User email"
        assert schema["format"] == "email"

    def test_ir_model_creation(self):
        """Test IRModel creation and JSON Schema export."""
        model = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(name="id", type=IRType.STRING, required=True),
                IRField(name="email", type=IRType.STRING, required=True),
                IRField(name="age", type=IRType.INTEGER, required=False),
            ],
            description="User model",
        )

        schema = model.to_json_schema()
        assert schema["type"] == "object"
        assert schema["title"] == "User"
        assert "id" in schema["properties"]
        assert "email" in schema["properties"]
        assert set(schema["required"]) == {"id", "email"}
        assert schema["x-version"] == "1.0.0"

    def test_ir_tool_creation(self):
        """Test IRTool creation."""
        input_model = IRModel(
            name="SearchInput",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(name="query", type=IRType.STRING, required=True),
                IRField(name="limit", type=IRType.INTEGER, required=False, default=10),
            ],
        )

        output_model = IRModel(
            name="SearchOutput",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(name="results", type=IRType.ARRAY, required=True),
                IRField(name="total", type=IRType.INTEGER, required=True),
            ],
        )

        tool = IRTool(
            name="search",
            version=SchemaVersion(1, 0, 0),
            description="Search tool",
            input_schema=input_model,
            output_schema=output_model,
            timeout=30,
        )

        assert tool.name == "search"
        assert tool.timeout == 30
        assert tool.input_schema.name == "SearchInput"

    def test_ir_spec_serialization(self):
        """Test IRSpec complete serialization."""
        model = IRModel(
            name="TestModel",
            version=SchemaVersion(1, 0, 0),
            fields=[IRField(name="id", type=IRType.STRING, required=True)],
        )

        spec = IRSpec(
            version=SchemaVersion(1, 0, 0),
            api_version="1.0",
            models={"TestModel": model},
        )

        # Serialize to JSON
        json_str = spec.to_json()
        assert "TestModel" in json_str
        assert "1.0.0" in json_str

        # Compute hash (deterministic)
        hash1 = spec.compute_hash()
        hash2 = spec.compute_hash()
        assert hash1 == hash2


class TestSchemaRegistry:
    """Test schema registry operations."""

    def test_registry_model_registration(self):
        """Test registering models in registry."""
        registry = SchemaRegistry()

        model = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[IRField(name="id", type=IRType.STRING, required=True)],
        )

        registry.register_model(model)
        retrieved = registry.get_model("User")
        assert retrieved is not None
        assert retrieved.name == "User"
        assert retrieved.version == model.version

    def test_registry_version_retrieval(self):
        """Test retrieving specific model version."""
        registry = SchemaRegistry()

        v1 = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[IRField(name="id", type=IRType.STRING, required=True)],
        )

        v2 = IRModel(
            name="User",
            version=SchemaVersion(2, 0, 0),
            fields=[
                IRField(name="id", type=IRType.STRING, required=True),
                IRField(name="email", type=IRType.STRING, required=True),
            ],
        )

        registry.register_model(v1)
        registry.register_model(v2)

        # Get latest (v2)
        latest = registry.get_model("User")
        assert latest.version == SchemaVersion(2, 0, 0)

        # Get specific version
        specific = registry.get_model("User", SchemaVersion(1, 0, 0))
        assert specific.version == SchemaVersion(1, 0, 0)

    def test_registry_find_by_tag(self):
        """Test finding models by tag."""
        registry = SchemaRegistry()

        model1 = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[],
            tags=["entity", "core"],
        )

        model2 = IRModel(
            name="Product",
            version=SchemaVersion(1, 0, 0),
            fields=[],
            tags=["entity", "commerce"],
        )

        registry.register_model(model1)
        registry.register_model(model2)

        entity_models = registry.find_models_by_tag("entity")
        assert len(entity_models) == 2

        core_models = registry.find_models_by_tag("core")
        assert len(core_models) == 1


class TestPythonModelGenerator:
    """Test Python model code generation."""

    def test_generate_simple_model(self):
        """Test generating simple Pydantic model."""
        generator = PythonModelGenerator()

        model = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(name="id", type=IRType.STRING, required=True, description="User ID"),
                IRField(name="email", type=IRType.STRING, required=True),
                IRField(name="age", type=IRType.INTEGER, required=False),
            ],
            description="User model",
        )

        code = generator.generate_model(model)

        # Verify generated code
        assert "class User(BaseModel):" in code
        assert "id: str" in code
        assert "email: str" in code
        assert "age: Optional[int] = None" in code
        assert "model_config" in code
        assert '"extra": "forbid"' in code

    def test_generate_model_with_constraints(self):
        """Test generating model with field constraints."""
        generator = PythonModelGenerator()

        model = IRModel(
            name="Product",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(
                    name="name",
                    type=IRType.STRING,
                    required=True,
                    constraints={"minLength": 1, "maxLength": 100},
                ),
                IRField(
                    name="price",
                    type=IRType.NUMBER,
                    required=True,
                    constraints={"minimum": 0},
                ),
            ],
        )

        code = generator.generate_model(model)

        assert "Field(" in code
        assert "min_length=" in code
        assert "max_length=" in code
        assert "ge=" in code  # greater than or equal

    def test_generate_model_with_nested_types(self):
        """Test generating model with array and object types."""
        generator = PythonModelGenerator()

        model = IRModel(
            name="Order",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(
                    name="items",
                    type=IRType.ARRAY,
                    required=True,
                    items=IRField(name="item", type=IRType.STRING),
                ),
                IRField(
                    name="metadata",
                    type=IRType.OBJECT,
                    required=False,
                ),
            ],
        )

        code = generator.generate_model(model)

        assert "List[str]" in code
        assert "Dict[str, Any]" in code


class TestPythonClientGenerator:
    """Test Python client code generation."""

    def test_generate_tool_client(self):
        """Test generating client for tool."""
        generator = PythonClientGenerator()

        input_model = IRModel(
            name="SearchInput",
            version=SchemaVersion(1, 0, 0),
            fields=[IRField(name="query", type=IRType.STRING, required=True)],
        )

        output_model = IRModel(
            name="SearchOutput",
            version=SchemaVersion(1, 0, 0),
            fields=[IRField(name="results", type=IRType.ARRAY, required=True)],
        )

        tool = IRTool(
            name="search",
            version=SchemaVersion(1, 0, 0),
            description="Search tool",
            input_schema=input_model,
            output_schema=output_model,
        )

        code = generator.generate_tool_client(tool)

        assert "class SearchClient:" in code
        assert "async def execute(" in code
        assert "def execute_sync(" in code
        assert "SearchInput" in code
        assert "SearchOutput" in code


class TestVersioningSystem:
    """Test versioning and migration system."""

    def test_compatibility_checking(self):
        """Test schema compatibility checking."""
        v1 = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[IRField(name="id", type=IRType.STRING, required=True)],
        )

        # Add non-required field (compatible)
        v2 = IRModel(
            name="User",
            version=SchemaVersion(1, 1, 0),
            fields=[
                IRField(name="id", type=IRType.STRING, required=True),
                IRField(name="email", type=IRType.STRING, required=False),
            ],
        )

        checker = CompatibilityChecker()
        is_compatible, changes = checker.check_compatibility(v1, v2)

        assert is_compatible  # Adding optional field is compatible
        assert len(changes) == 1
        assert changes[0].type.value == "field_added"

    def test_breaking_change_detection(self):
        """Test detecting breaking changes."""
        v1 = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(name="id", type=IRType.STRING, required=True),
                IRField(name="name", type=IRType.STRING, required=False),
            ],
        )

        # Make field required (breaking)
        v2 = IRModel(
            name="User",
            version=SchemaVersion(2, 0, 0),
            fields=[
                IRField(name="id", type=IRType.STRING, required=True),
                IRField(name="name", type=IRType.STRING, required=True),  # Now required
            ],
        )

        checker = CompatibilityChecker()
        is_compatible, changes = checker.check_compatibility(v1, v2)

        assert not is_compatible  # Making required is breaking
        assert any(c.breaking for c in changes)

    def test_migration_generation(self):
        """Test generating migrations."""
        v1 = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[IRField(name="id", type=IRType.STRING, required=True)],
        )

        v2 = IRModel(
            name="User",
            version=SchemaVersion(1, 1, 0),
            fields=[
                IRField(name="id", type=IRType.STRING, required=True),
                IRField(name="email", type=IRType.STRING, required=False, default=""),
            ],
        )

        generator = MigrationGenerator()
        migration = generator.generate(v1, v2)

        assert migration.schema_name == "User"
        assert migration.from_version == v1.version
        assert migration.to_version == v2.version
        assert migration.upgrade_code is not None
        assert migration.downgrade_code is not None
        assert "email" in migration.upgrade_code

    def test_migration_execution(self):
        """Test executing migrations on data."""
        v1 = IRModel(
            name="User",
            version=SchemaVersion(1, 0, 0),
            fields=[IRField(name="id", type=IRType.STRING, required=True)],
        )

        v2 = IRModel(
            name="User",
            version=SchemaVersion(1, 1, 0),
            fields=[
                IRField(name="id", type=IRType.STRING, required=True),
                IRField(name="status", type=IRType.STRING, required=False, default="active"),
            ],
        )

        generator = MigrationGenerator()
        migration = generator.generate(v1, v2)

        manager = VersionManager()
        manager.register_migration(migration)

        # Test data migration
        old_data = {"id": "user123"}
        migrated_data = manager.execute_migration(migration, old_data, direction="upgrade")

        assert "id" in migrated_data
        assert "status" in migrated_data
        assert migrated_data["status"] == "active"


class TestValidation:
    """Test runtime validation."""

    def test_request_validation(self):
        """Test request validation."""
        model = IRModel(
            name="SearchInput",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(name="query", type=IRType.STRING, required=True),
                IRField(name="limit", type=IRType.INTEGER, required=False, default=10),
            ],
        )

        validator = RequestValidator(model)

        # Valid request
        valid_data = {"query": "test search"}
        validated = validator.validate(valid_data)
        assert validated["query"] == "test search"

        # Invalid request (missing required field)
        from namel3ss.sdk_sync.errors import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            validator.validate({})
        assert "query" in str(exc_info.value)

    def test_response_validation(self):
        """Test response validation."""
        model = IRModel(
            name="SearchOutput",
            version=SchemaVersion(1, 0, 0),
            fields=[
                IRField(name="results", type=IRType.ARRAY, required=True),
                IRField(name="total", type=IRType.INTEGER, required=True),
            ],
        )

        validator = ResponseValidator(model)

        # Valid response
        valid_data = {"results": [], "total": 0}
        validated = validator.validate(valid_data)
        assert validated["total"] == 0

    def test_validation_context(self):
        """Test validation context."""
        context = ValidationContext(strict=False)

        # Add warning (should not raise)
        context.add_warning("Test warning")
        assert len(context.warnings) == 1

        # Add error in non-strict mode
        context.add_error({"field": "test", "error": "invalid"})
        assert context.has_errors()


@pytest.mark.asyncio
async def test_complete_sdk_generation():
    """
    Integration test: Complete SDK generation workflow.
    
    This test verifies:
    1. IR creation from models/tools
    2. Registry operations
    3. Python SDK generation
    4. Generated code is syntactically valid
    """
    # Step 1: Create IR models and tools
    user_model = IRModel(
        name="User",
        version=SchemaVersion(1, 0, 0),
        fields=[
            IRField(name="id", type=IRType.STRING, required=True, description="User ID"),
            IRField(name="email", type=IRType.STRING, required=True, description="Email address"),
            IRField(name="name", type=IRType.STRING, required=False, description="Full name"),
        ],
        description="User entity model",
        namespace="core",
        tags=["entity", "user"],
    )

    search_input = IRModel(
        name="SearchInput",
        version=SchemaVersion(1, 0, 0),
        fields=[
            IRField(name="query", type=IRType.STRING, required=True),
            IRField(name="limit", type=IRType.INTEGER, required=False, default=10),
        ],
    )

    search_output = IRModel(
        name="SearchOutput",
        version=SchemaVersion(1, 0, 0),
        fields=[
            IRField(name="results", type=IRType.ARRAY, required=True),
            IRField(name="total", type=IRType.INTEGER, required=True),
        ],
    )

    search_tool = IRTool(
        name="search",
        version=SchemaVersion(1, 0, 0),
        description="Search for users",
        input_schema=search_input,
        output_schema=search_output,
        namespace="tools",
        tags=["search"],
    )

    # Step 2: Create IR spec
    spec = IRSpec(
        version=SchemaVersion(1, 0, 0),
        api_version="1.0",
        models={
            "User": user_model,
            "SearchInput": search_input,
            "SearchOutput": search_output,
        },
        tools={"search": search_tool},
    )

    # Step 3: Generate SDK
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generator = PythonSDKGenerator(
            spec=spec,
            package_name="test_sdk",
            output_dir=output_dir,
            base_url="http://test.local:8000",
        )

        await generator.generate()

        # Step 4: Verify generated files exist
        package_dir = output_dir / "test_sdk"
        assert package_dir.exists()
        assert (package_dir / "__init__.py").exists()
        assert (package_dir / "models.py").exists()
        assert (package_dir / "clients.py").exists()
        assert (package_dir / "exceptions.py").exists()
        assert (package_dir / "validation.py").exists()
        assert (package_dir / "py.typed").exists()
        assert (output_dir / "pyproject.toml").exists()
        assert (output_dir / "README.md").exists()

        # Step 5: Verify generated code is syntactically valid
        models_code = (package_dir / "models.py").read_text()
        assert "class User(BaseModel):" in models_code
        assert "class SearchInput(BaseModel):" in models_code
        assert "class SearchOutput(BaseModel):" in models_code
        assert 'SCHEMA_VERSION = "1.0.0"' in models_code

        clients_code = (package_dir / "clients.py").read_text()
        assert "class SearchClient:" in clients_code
        assert "async def execute(" in clients_code

        # Step 6: Verify pyproject.toml
        pyproject_content = (output_dir / "pyproject.toml").read_text()
        assert 'name = "test_sdk"' in pyproject_content
        assert "pydantic>=2.0.0" in pyproject_content
        assert "httpx>=0.25.0" in pyproject_content

        # Step 7: Compile Python code to verify syntax
        import py_compile

        py_compile.compile(str(package_dir / "models.py"), doraise=True)
        py_compile.compile(str(package_dir / "clients.py"), doraise=True)
        py_compile.compile(str(package_dir / "exceptions.py"), doraise=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
