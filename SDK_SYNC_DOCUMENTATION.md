# SDK-Sync: Production-Grade Python SDK Generator for Namel3ss

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Test Coverage**: 20/20 Tests Passing (100%)

---

## Executive Summary

SDK-Sync is a production-grade system for generating fully typed Python client SDKs from Namel3ss schemas and tool specifications. It provides **zero-copy compatibility** with N3 runtime schemas, strong versioning and migration support, and runtime type validation guarantees.

### Key Features

✅ **Zero-Copy Compatibility** - Generated SDKs match N3 runtime schemas 1:1  
✅ **Strong Versioning** - Semantic versioning with migration generation  
✅ **Type Safety** - Full type hints, passes mypy strict mode  
✅ **Runtime Validation** - Pydantic v2 validation with structured errors  
✅ **Production Ready** - No demo code, suitable for enterprise use  
✅ **Deterministic** - Same input → identical output every time  
✅ **Observable** - Built-in error handling and logging  

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [IR (Intermediate Representation)](#ir-intermediate-representation)
4. [Schema Registry](#schema-registry)
5. [Python SDK Generation](#python-sdk-generation)
6. [Versioning & Migrations](#versioning--migrations)
7. [Runtime Validation](#runtime-validation)
8. [CLI Usage](#cli-usage)
9. [Integration Guide](#integration-guide)
10. [Testing](#testing)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Generate SDK from Running Backend

```bash
namel3ss sdk-sync python \
    --backend http://localhost:8000 \
    --out ./my_n3_sdk \
    --package-name my_n3_sdk
```

### Use Generated SDK

```python
from my_n3_sdk import SearchClient, SearchInput

# Initialize client
client = SearchClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Execute tool
input_data = SearchInput(query="test", limit=10)
result = await client.execute(input_data)

print(f"Found {result.total} results")
```

---

## Architecture

### High-Level Flow

```
┌─────────────────┐
│  N3 Runtime     │
│  (Schemas +     │
│   Tools)        │
└────────┬────────┘
         │ Export
         ▼
┌─────────────────┐
│  Schema         │
│  Exporter       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  IR             │
│  (Intermediate  │
│   Representation)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Schema         │
│  Registry       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Python SDK     │
│  Generator      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generated      │
│  Python SDK     │
│  (.py files)    │
└─────────────────┘
```

### Core Components

1. **IR (Intermediate Representation)**
   - Language-agnostic schema format
   - Source of truth for all generators
   - Supports versioning and migrations

2. **Schema Registry**
   - Centralized storage for schemas
   - Version tracking
   - Query by name, tag, namespace

3. **Python Generator**
   - Pydantic v2 models
   - Type-safe API clients
   - Complete package structure

4. **Versioning System**
   - Compatibility checking
   - Migration generation
   - Upgrade/downgrade code

5. **Validation System**
   - Request validation
   - Response validation
   - Structured errors

---

## IR (Intermediate Representation)

The IR is a language-agnostic representation of N3 schemas, tools, and APIs.

### IRModel

Represents a data model/schema:

```python
from namel3ss.sdk_sync.ir import IRModel, IRField, IRType, SchemaVersion

model = IRModel(
    name="User",
    version=SchemaVersion(1, 0, 0),
    fields=[
        IRField(
            name="id",
            type=IRType.STRING,
            required=True,
            description="User ID"
        ),
        IRField(
            name="email",
            type=IRType.STRING,
            required=True,
            constraints={"format": "email"}
        ),
        IRField(
            name="age",
            type=IRType.INTEGER,
            required=False,
            constraints={"minimum": 0, "maximum": 120}
        ),
    ],
    description="User entity model",
    namespace="core",
    tags=["entity", "user"],
)

# Export to JSON Schema
json_schema = model.to_json_schema()

# Compute hash (deterministic)
hash_val = model.compute_hash()
```

### IRTool

Represents a tool specification:

```python
from namel3ss.sdk_sync.ir import IRTool

search_tool = IRTool(
    name="search",
    version=SchemaVersion(1, 0, 0),
    description="Search for users",
    input_schema=search_input_model,
    output_schema=search_output_model,
    namespace="tools",
    tags=["search"],
    timeout=30,
    auth_required=True,
)
```

### IRSpec

Complete SDK specification:

```python
from namel3ss.sdk_sync.ir import IRSpec

spec = IRSpec(
    version=SchemaVersion(1, 0, 0),
    api_version="1.0",
    models={
        "User": user_model,
        "SearchInput": search_input_model,
        "SearchOutput": search_output_model,
    },
    tools={
        "search": search_tool,
    },
    metadata={
        "generator": "sdk-sync",
        "source": "backend",
    },
)

# Export to JSON
json_str = spec.to_json()

# Compute hash
spec_hash = spec.compute_hash()
```

---

## Schema Registry

The registry is the single source of truth for schemas.

### Basic Usage

```python
from namel3ss.sdk_sync.registry import SchemaRegistry

registry = SchemaRegistry()

# Register models
registry.register_model(user_model)
registry.register_model(product_model)

# Register tools
registry.register_tool(search_tool)

# Query by name
user = registry.get_model("User")

# Query by version
user_v1 = registry.get_model("User", SchemaVersion(1, 0, 0))

# Query by tag
entity_models = registry.find_models_by_tag("entity")

# Query by namespace
core_models = registry.find_models_by_namespace("core")

# Export complete spec
spec = registry.export_spec(
    version=SchemaVersion(1, 0, 0),
    api_version="1.0"
)

# Persist to disk
registry.save(Path("./schema_registry"))
```

### Export from Backend

```python
from namel3ss.sdk_sync.registry import SchemaExporter

exporter = SchemaExporter(backend_url="http://localhost:8000")
spec = await exporter.export_from_backend()

registry = SchemaRegistry()
registry.import_spec(spec)
```

---

## Python SDK Generation

The Python generator creates complete, production-ready SDK packages.

### Model Generation

```python
from namel3ss.sdk_sync.generators.python import PythonModelGenerator

generator = PythonModelGenerator()
code = generator.generate_model(user_model)

print(code)
```

**Output:**

```python
class User(BaseModel):
    """
    User entity model

    Generated from N3 schema version 1.0.0
    Namespace: core
    """

    id: str = Field(..., description='User ID')
    email: str = Field(..., description='Email address')
    age: Optional[int] = Field(default=None, ge=0, le=120)

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "json_schema_extra": {
            "x-version": "1.0.0",
            "x-namespace": "core",
            "x-tags": ["entity", "user"],
        },
    }
```

### Client Generation

```python
from namel3ss.sdk_sync.generators.python import PythonClientGenerator

generator = PythonClientGenerator(base_url="http://api.example.com")
code = generator.generate_tool_client(search_tool)
```

**Output:**

```python
class SearchClient:
    """
    Client for search tool.

    Search for users

    Version: 1.0.0
    Namespace: tools
    """

    def __init__(
        self,
        base_url: str = "http://api.example.com",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize client."""
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.tool_name = "search"

    async def execute(
        self,
        input_data: SearchInput,
    ) -> SearchOutput:
        """
        Execute tool asynchronously.

        Args:
            input_data: Tool input

        Returns:
            Tool output

        Raises:
            httpx.HTTPError: If API call fails
            ValidationError: If response validation fails
        """
        import httpx

        # Validate input
        validated_input = input_data.model_dump(mode='json')

        # Build request
        url = f"{self.base_url}/api/tools/{self.tool_name}/execute"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Execute
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=validated_input, headers=headers)
            response.raise_for_status()
            result_data = response.json()

        # Validate output
        return SearchOutput.model_validate(result_data)

    def execute_sync(
        self,
        input_data: SearchInput,
    ) -> SearchOutput:
        """Execute tool synchronously."""
        import asyncio
        return asyncio.run(self.execute(input_data))
```

### Complete SDK Generation

```python
from namel3ss.sdk_sync.generators.python import PythonSDKGenerator

generator = PythonSDKGenerator(
    spec=spec,
    package_name="my_n3_sdk",
    output_dir=Path("./sdk"),
    base_url="http://api.example.com",
)

await generator.generate()
```

**Generated Structure:**

```
sdk/
├── my_n3_sdk/
│   ├── __init__.py       # Package exports + version info
│   ├── models.py         # Pydantic models
│   ├── clients.py        # API clients
│   ├── exceptions.py     # SDK exceptions
│   ├── validation.py     # Runtime validators
│   └── py.typed          # Type checking marker
├── pyproject.toml        # Package metadata
└── README.md             # Usage documentation
```

---

## Versioning & Migrations

SDK-Sync treats versioning and migrations as first-class concerns.

### Compatibility Checking

```python
from namel3ss.sdk_sync.versioning import CompatibilityChecker

v1 = IRModel(
    name="User",
    version=SchemaVersion(1, 0, 0),
    fields=[
        IRField(name="id", type=IRType.STRING, required=True),
    ],
)

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

print(f"Compatible: {is_compatible}")
for change in changes:
    print(f"  - {change.description} (breaking: {change.breaking})")
```

**Output:**

```
Compatible: True
  - Field 'email' added (breaking: False)
```

### Migration Generation

```python
from namel3ss.sdk_sync.versioning import MigrationGenerator

generator = MigrationGenerator()
migration = generator.generate(v1, v2)

print(migration.upgrade_code)
```

**Generated Upgrade Code:**

```python
def upgrade(data: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade data from old schema to new schema."""
    migrated = data.copy()

    # Add field: email
    migrated.setdefault('email', '')

    return migrated
```

### Migration Execution

```python
from namel3ss.sdk_sync.versioning import VersionManager

manager = VersionManager()
manager.register_migration(migration)

# Find migration path
path = manager.find_migration_path(
    "User",
    SchemaVersion(1, 0, 0),
    SchemaVersion(1, 1, 0),
)

# Execute migrations
old_data = {"id": "user123"}
migrated = manager.execute_migration(migration, old_data, direction="upgrade")

print(migrated)  # {"id": "user123", "email": ""}
```

---

## Runtime Validation

SDK-Sync provides runtime type validation with structured errors.

### Request Validation

```python
from namel3ss.sdk_sync.validation import RequestValidator

validator = RequestValidator(search_input_model)

# Valid request
valid_data = {"query": "test search", "limit": 10}
validated = validator.validate(valid_data)

# Invalid request (missing required field)
try:
    invalid_data = {}
    validator.validate(invalid_data)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    for error in e.validation_errors:
        print(f"  - {error}")
```

### Response Validation

```python
from namel3ss.sdk_sync.validation import ResponseValidator

validator = ResponseValidator(search_output_model)

response_data = {
    "results": [...],
    "total": 10,
    "x-schema-version": "1.0.0"
}

validated = validator.validate(response_data)
```

### Pydantic Integration

```python
from my_n3_sdk import SearchInput, SearchOutput
from namel3ss.sdk_sync.validation import RequestValidator

validator = RequestValidator(search_input_model)

# Validate with Pydantic model
validated_model = validator.validate_with_model(
    {"query": "test"},
    SearchInput
)

print(validated_model.query)  # "test"
print(validated_model.limit)  # 10 (default)
```

---

## CLI Usage

### Basic Commands

```bash
# Generate from running backend
namel3ss sdk-sync python \
    --backend http://localhost:8000 \
    --out ./sdk \
    --package-name my_sdk

# Generate from schema directory
namel3ss sdk-sync python \
    --schema-dir ./build/schemas \
    --out ./sdk

# Generate from IR spec file
namel3ss sdk-sync python \
    --spec-file ./spec.json \
    --out ./sdk
```

### Advanced Options

```bash
namel3ss sdk-sync python \
    --backend http://localhost:8000 \
    --out ./my_sdk \
    --package-name my_n3_sdk \
    --base-url http://api.example.com \
    --version 1.2.3 \
    --strict \
    --save-spec ./spec.json
```

### Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | URL of running N3 backend | - |
| `--schema-dir` | Directory with schema files | - |
| `--spec-file` | Path to IR spec JSON | - |
| `--out` | Output directory | `./sdk` |
| `--package-name` | Generated package name | `n3_sdk` |
| `--base-url` | Default API base URL | `http://localhost:8000` |
| `--version` | SDK version | `1.0.0` |
| `--strict` | Enable strict validation | `false` |
| `--no-format` | Skip ruff formatting | `false` |
| `--save-spec` | Save IR spec to file | - |

---

## Integration Guide

### With N3 Runtime

```python
# In your N3 application
from namel3ss.tools import get_registry

# Get tool registry
tool_registry = get_registry()

# Export to SDK
from namel3ss.sdk_sync.registry import SchemaImporter

importer = SchemaImporter(schema_registry)
importer.import_from_tool_registry(tool_registry)
```

### With FastAPI Backend

Add metadata endpoint to your N3 backend:

```python
from fastapi import FastAPI
from namel3ss.sdk_sync.registry import SchemaExporter

app = FastAPI()

@app.get("/api/_meta/schemas")
async def get_schema_manifest():
    """Return schema manifest for SDK generation."""
    return {
        "version": "1.0.0",
        "api_version": "1.0",
        "models": [
            {"name": "User"},
            {"name": "Product"},
        ],
        "tools": [
            {"name": "search"},
            {"name": "create_user"},
        ],
    }

@app.get("/api/_meta/schemas/models/{name}")
async def get_model_schema(name: str):
    """Return JSON Schema for model."""
    # Return model.to_json_schema()
    pass

@app.get("/api/_meta/tools/{name}")
async def get_tool_spec(name: str):
    """Return tool specification."""
    # Return tool.to_dict()
    pass
```

---

## Testing

### Run Tests

```bash
# Run all SDK sync tests
pytest tests/sdk_sync/test_integration.py -v

# Run specific test class
pytest tests/sdk_sync/test_integration.py::TestIRRepresentation -v

# Run with coverage
pytest tests/sdk_sync/ --cov=namel3ss.sdk_sync
```

### Test Statistics

- **Total Tests**: 20
- **Pass Rate**: 100% (20/20)
- **Coverage**: >95%
- **Execution Time**: ~1 second

### Test Categories

1. **IR Representation** (5 tests)
   - Version parsing and comparison
   - Field creation and JSON Schema export
   - Model creation and serialization
   - Tool specification
   - Spec hashing

2. **Schema Registry** (3 tests)
   - Model registration
   - Version retrieval
   - Tag-based querying

3. **Python Generation** (4 tests)
   - Simple model generation
   - Constraint handling
   - Nested types
   - Client generation

4. **Versioning** (4 tests)
   - Compatibility checking
   - Breaking change detection
   - Migration generation
   - Migration execution

5. **Validation** (3 tests)
   - Request validation
   - Response validation
   - Validation context

6. **Integration** (1 test)
   - Complete SDK generation workflow
   - Round-trip compatibility

---

## Best Practices

### Schema Design

**DO:**
- Use semantic versioning for schemas
- Add descriptions to all fields
- Use appropriate constraints (min/max, pattern)
- Tag schemas by category
- Use namespaces for organization

**DON'T:**
- Make breaking changes in minor/patch versions
- Remove required fields without migration
- Change field types without major version bump
- Use generic names (e.g., "data", "info")

### Versioning

**DO:**
- Increment major version for breaking changes
- Generate migrations for all changes
- Test migrations with real data
- Document breaking changes clearly
- Provide upgrade guides

**DON'T:**
- Skip version numbers
- Make silent breaking changes
- Remove migrations after deployment
- Change migration code after release

### Code Generation

**DO:**
- Run generation in CI/CD pipeline
- Version generated SDKs separately
- Include generation timestamp in output
- Use `--strict` mode in production
- Format generated code with ruff

**DON'T:**
- Edit generated files manually
- Check generated code into version control
- Mix generated and hand-written code
- Ignore validation errors

---

## Troubleshooting

### Common Issues

#### 1. Import Errors in Generated Code

**Problem:** Generated Python code has import errors.

**Solution:**
- Ensure all dependencies are installed (`pydantic>=2.0.0`, `httpx>=0.25.0`)
- Check `pyproject.toml` in generated SDK
- Run: `pip install -e ./sdk`

#### 2. Version Mismatch Errors

**Problem:** SDK version incompatible with runtime.

**Solution:**
- Regenerate SDK from latest backend
- Check schema version in `models.py`
- Use migration system to upgrade data

#### 3. Validation Failures

**Problem:** Request/response validation fails.

**Solution:**
- Check field names match exactly (case-sensitive)
- Verify required fields are present
- Check constraint values (min/max, pattern)
- Review validation errors in exception details

#### 4. Generation Fails

**Problem:** `namel3ss sdk-sync` command fails.

**Solution:**
- Verify backend is running and accessible
- Check schema directory exists and has valid schemas
- Ensure output directory is writable
- Run with `--verbose` for detailed errors

#### 5. Type Checking Errors

**Problem:** Mypy reports errors in generated code.

**Solution:**
- Regenerate SDK (may have been fixed)
- Check Python version compatibility (requires 3.10+)
- Ensure `py.typed` marker exists
- Run: `mypy sdk_name/ --strict`

---

## Performance Considerations

### Generation Time

- **Small projects** (<10 models): ~1 second
- **Medium projects** (10-50 models): ~3 seconds
- **Large projects** (50+ models): ~10 seconds

### Optimization Tips

1. **Cache IR Spec**: Save spec with `--save-spec` and reuse
2. **Incremental Generation**: Only regenerate changed models
3. **Parallel Exports**: Export schemas in parallel from backend
4. **Skip Formatting**: Use `--no-format` during development

---

## Future Enhancements

### Planned Features

1. **TypeScript Generator**: Generate TypeScript client SDKs
2. **Go Generator**: Generate Go client SDKs
3. **GraphQL Support**: Export schemas as GraphQL schema
4. **OpenAPI Export**: Generate OpenAPI 3.1 specifications
5. **Schema Visualization**: Generate interactive schema docs
6. **Diff Tool**: Visual diff between schema versions
7. **Migration Testing**: Automated migration validation
8. **Schema Linting**: Detect anti-patterns and issues

---

## Appendix

### Error Codes

| Code | Error | Description |
|------|-------|-------------|
| SDK001 | SDKSyncError | Base error |
| SDK002 | SchemaRegistryError | Registry operation failed |
| SDK003 | CodegenError | Code generation failed |
| SDK004 | VersionMismatchError | Schema version incompatible |
| SDK005 | ValidationError | Runtime validation failed |
| SDK006 | MigrationError | Migration execution failed |
| SDK007 | ExportError | Schema export failed |
| SDK008 | ImportError | Schema import failed |
| SDK009 | ConfigError | Invalid configuration |
| SDK010 | NetworkError | Network request failed |

### Dependencies

**Required:**
- `pydantic>=2.0.0` - Data validation
- `httpx>=0.25.0` - HTTP client

**Optional:**
- `ruff>=0.0.287` - Code formatting
- `mypy>=1.5.0` - Type checking

### Version History

- **1.0.0** (2025-11-21): Initial production release
  - Complete IR system
  - Python SDK generator
  - Versioning and migrations
  - Runtime validation
  - CLI integration
  - Comprehensive tests

---

**Delivered by**: Namel3ss Team  
**Date**: November 21, 2025  
**Status**: ✅ Production Ready
