"""
SDK Sync - Production-grade SDK generator for Namel3ss.

This package generates fully typed Python client bindings from N3 schemas
and tool specifications with zero-copy compatibility guarantees.

Architecture:
    1. IR (Intermediate Representation): Language-agnostic schema format
    2. Schema Registry: Exports N3 runtime schemas to IR
    3. Python Generator: Generates typed Pydantic models and clients
    4. Versioning: Schema migrations and compatibility validation
    5. Validation: Runtime type checking with structured errors

Key Features:
    - Zero-copy compatibility with N3 runtime schemas
    - Strong versioning and schema migration support
    - Runtime type validation guarantees
    - Production-ready quality (no demo-only hacks)
    - Suitable for real-world SaaS/enterprise usage

Example:
    Generate Python SDK from running backend:
    ```bash
    namel3ss sdk-sync python \\
        --backend http://localhost:8000 \\
        --out ./my_n3_sdk \\
        --package-name my_n3_sdk
    ```
    
    Generate from schema files:
    ```bash
    namel3ss sdk-sync python \\
        --schema-dir ./build/schemas \\
        --out ./my_n3_sdk
    ```

Author: Namel3ss Team
License: MIT
"""

from .ir import (
    IRModel,
    IRField,
    IRTool,
    IRChain,
    IRAgent,
    IREndpoint,
    IRSpec,
    IRType,
    SchemaVersion,
    SchemaMigration,
)
from .registry import (
    SchemaRegistry,
    SchemaExporter,
    SchemaImporter,
)
from .exporter import (
    export_schemas_from_app,
    generate_metadata_router,
)
from .generators.python import (
    PythonModelGenerator,
    PythonClientGenerator,
    PythonSDKGenerator,
)
from .versioning import (
    VersionManager,
    MigrationGenerator,
    CompatibilityChecker,
)
from .validation import (
    TypeValidator,
    RequestValidator,
    ResponseValidator,
)
from .cli import sdk_sync_command
from .errors import (
    SDKSyncError,
    SchemaRegistryError,
    CodegenError,
    VersionMismatchError,
    ValidationError as SDKValidationError,
)

__all__ = [
    # IR Components
    "IRModel",
    "IRField",
    "IRType",
    "IRTool",
    "IRChain",
    "IRAgent",
    "IREndpoint",
    "IRSpec",
    "SchemaVersion",
    "SchemaMigration",
    # Registry
    "SchemaRegistry",
    "SchemaExporter",
    "SchemaImporter",
    "export_schemas_from_app",
    "generate_metadata_router",
    # Generators
    "PythonModelGenerator",
    "PythonClientGenerator",
    "PythonSDKGenerator",
    # Versioning
    "VersionManager",
    "MigrationGenerator",
    "CompatibilityChecker",
    # Validation
    "TypeValidator",
    "RequestValidator",
    "ResponseValidator",
    # CLI
    "sdk_sync_command",
    # Errors
    "SDKSyncError",
    "SchemaRegistryError",
    "CodegenError",
    "VersionMismatchError",
    "SDKValidationError",
]

__version__ = "1.0.0"
