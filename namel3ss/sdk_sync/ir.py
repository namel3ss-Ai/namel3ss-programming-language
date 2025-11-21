"""
Intermediate Representation (IR) for SDK generation.

Language-agnostic representation of N3 schemas, tools, and APIs.
Serves as the single source of truth for all SDK generators.

The IR design guarantees:
    1. Zero-copy compatibility with N3 runtime schemas
    2. Version tracking for migrations
    3. Complete type information for codegen
    4. Deterministic serialization

Example:
    Create IR model from N3 schema:
    ```python
    model = IRModel(
        name="User",
        version=SchemaVersion(major=1, minor=0, patch=0),
        fields=[
            IRField(
                name="id",
                type="string",
                required=True,
                description="User ID"
            ),
            IRField(
                name="email",
                type="string",
                required=True,
                constraints={"format": "email"}
            ),
        ]
    )
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
import json
import hashlib


class IRType(str, Enum):
    """Supported IR field types (maps to JSON Schema types)."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    ANY = "any"
    # Complex types
    UNION = "union"
    REF = "$ref"  # Reference to another model


@dataclass(frozen=True)
class SchemaVersion:
    """
    Semantic versioning for schemas.
    
    Follows semver principles:
    - major: Breaking changes (incompatible)
    - minor: New features (backward compatible)
    - patch: Bug fixes (backward compatible)
    """

    major: int
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) <= (
            other.major,
            other.minor,
            other.patch,
        )

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if this version is compatible with another."""
        # Same major version = compatible
        return self.major == other.major

    @classmethod
    def parse(cls, version_str: str) -> "SchemaVersion":
        """Parse version from string like '1.0.0'."""
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(
            major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2])
        )


@dataclass
class IRField:
    """
    Represents a field in a model/schema.
    
    Zero-copy guarantee: Field definition matches 1:1 with N3 runtime.
    """

    name: str
    type: Union[IRType, str]  # IRType or custom type name
    required: bool = False
    nullable: bool = False
    description: Optional[str] = None
    default: Optional[Any] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For array types
    items: Optional["IRField"] = None
    # For object types
    properties: Optional[Dict[str, "IRField"]] = None
    # For union types
    union_types: Optional[List["IRField"]] = None
    # For ref types
    ref_name: Optional[str] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Export to JSON Schema format.
        
        This ensures compatibility with N3 runtime schemas.
        """
        schema: Dict[str, Any] = {}

        # Type
        if self.type == IRType.REF and self.ref_name:
            schema["$ref"] = f"#/definitions/{self.ref_name}"
        elif self.type == IRType.UNION and self.union_types:
            schema["anyOf"] = [t.to_json_schema() for t in self.union_types]
        elif self.type == IRType.ARRAY and self.items:
            schema["type"] = "array"
            schema["items"] = self.items.to_json_schema()
        elif self.type == IRType.OBJECT and self.properties:
            schema["type"] = "object"
            schema["properties"] = {
                k: v.to_json_schema() for k, v in self.properties.items()
            }
        else:
            schema["type"] = str(self.type.value if isinstance(self.type, IRType) else self.type)

        # Nullable
        if self.nullable:
            if "type" in schema:
                schema["type"] = [schema["type"], "null"]

        # Description
        if self.description:
            schema["description"] = self.description

        # Default
        if self.default is not None:
            schema["default"] = self.default

        # Constraints
        schema.update(self.constraints)

        return schema

    def compute_hash(self) -> str:
        """Compute deterministic hash of field definition."""
        normalized = json.dumps(self.to_json_schema(), sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class IRModel:
    """
    Represents a data model/schema.
    
    Used for:
    - Entity models
    - Tool input/output schemas
    - Chain/agent configuration
    """

    name: str
    version: SchemaVersion
    fields: List[IRField]
    description: Optional[str] = None
    namespace: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_json_schema(self) -> Dict[str, Any]:
        """Export to JSON Schema format."""
        required_fields = [f.name for f in self.fields if f.required]

        schema = {
            "type": "object",
            "title": self.name,
            "properties": {
                f.name: f.to_json_schema() for f in self.fields
            },
        }

        if required_fields:
            schema["required"] = required_fields

        if self.description:
            schema["description"] = self.description

        # Add metadata
        schema["x-version"] = str(self.version)
        if self.namespace:
            schema["x-namespace"] = self.namespace
        if self.tags:
            schema["x-tags"] = self.tags

        return schema

    def compute_hash(self) -> str:
        """Compute deterministic hash of model definition."""
        normalized = json.dumps(self.to_json_schema(), sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get_field(self, name: str) -> Optional[IRField]:
        """Get field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None


@dataclass
class IRTool:
    """
    Represents a tool specification.
    
    Tools are executable operations with typed inputs/outputs.
    """

    name: str
    version: SchemaVersion
    description: str
    input_schema: IRModel
    output_schema: IRModel
    namespace: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    timeout: Optional[int] = None  # seconds
    retry_config: Dict[str, Any] = field(default_factory=dict)
    auth_required: bool = False
    streaming: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_full_name(self) -> str:
        """Get fully qualified tool name."""
        if self.namespace:
            return f"{self.namespace}.{self.name}"
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "name": self.name,
            "version": str(self.version),
            "description": self.description,
            "input_schema": self.input_schema.to_json_schema(),
            "output_schema": self.output_schema.to_json_schema(),
            "namespace": self.namespace,
            "tags": self.tags,
            "timeout": self.timeout,
            "retry_config": self.retry_config,
            "auth_required": self.auth_required,
            "streaming": self.streaming,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class IRChain:
    """Represents a chain specification."""

    name: str
    version: SchemaVersion
    description: str
    input_schema: IRModel
    output_schema: IRModel
    steps: List[Dict[str, Any]]
    namespace: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRAgent:
    """Represents an agent specification."""

    name: str
    version: SchemaVersion
    description: str
    input_schema: IRModel
    output_schema: IRModel
    tools: List[str]  # Tool names
    system_prompt: Optional[str] = None
    namespace: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IREndpoint:
    """Represents an API endpoint."""

    path: str
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    description: str
    request_schema: Optional[IRModel] = None
    response_schema: Optional[IRModel] = None
    auth_required: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaMigration:
    """
    Represents a schema migration between versions.
    
    Migrations must be explicit and versioned.
    """

    schema_name: str
    from_version: SchemaVersion
    to_version: SchemaVersion
    description: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    breaking: bool = False
    upgrade_code: Optional[str] = None  # Python code for migration
    downgrade_code: Optional[str] = None  # Python code for rollback
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "schema_name": self.schema_name,
            "from_version": str(self.from_version),
            "to_version": str(self.to_version),
            "description": self.description,
            "changes": self.changes,
            "breaking": self.breaking,
            "upgrade_code": self.upgrade_code,
            "downgrade_code": self.downgrade_code,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class IRSpec:
    """
    Complete SDK specification.
    
    Contains all models, tools, chains, agents, and endpoints.
    """

    version: SchemaVersion
    api_version: str
    models: Dict[str, IRModel] = field(default_factory=dict)
    tools: Dict[str, IRTool] = field(default_factory=dict)
    chains: Dict[str, IRChain] = field(default_factory=dict)
    agents: Dict[str, IRAgent] = field(default_factory=dict)
    endpoints: Dict[str, IREndpoint] = field(default_factory=dict)
    migrations: List[SchemaMigration] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "version": str(self.version),
            "api_version": self.api_version,
            "models": {k: v.to_json_schema() for k, v in self.models.items()},
            "tools": {k: v.to_dict() for k, v in self.tools.items()},
            "chains": {k: vars(v) for k, v in self.chains.items()},
            "agents": {k: vars(v) for k, v in self.agents.items()},
            "endpoints": {k: vars(v) for k, v in self.endpoints.items()},
            "migrations": [m.to_dict() for m in self.migrations],
            "metadata": self.metadata,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def compute_hash(self) -> str:
        """Compute deterministic hash of entire spec."""
        normalized = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()
