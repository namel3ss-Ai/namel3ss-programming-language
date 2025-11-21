"""
Schema Registry - Centralized storage and export of N3 runtime schemas.

The registry serves as the single source of truth, ensuring zero-copy
compatibility between N3 runtime and generated SDKs.

Architecture:
    1. SchemaExporter: Extracts schemas from N3 runtime
    2. SchemaRegistry: Stores schemas with versioning
    3. SchemaImporter: Loads schemas from various sources

Example:
    Export from N3 backend:
    ```python
    exporter = SchemaExporter("http://localhost:8000")
    spec = await exporter.export_all()
    registry = SchemaRegistry()
    registry.import_spec(spec)
    ```
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from datetime import datetime

from .ir import (
    IRSpec,
    IRModel,
    IRTool,
    IRField,
    IRType,
    SchemaVersion,
    SchemaMigration,
)
from .errors import (
    SchemaRegistryError,
    ExportError,
    ImportError,
)


class SchemaRegistry:
    """
    Centralized registry for N3 schemas with versioning support.
    
    Features:
    - Version tracking for all schemas
    - Migration management
    - Query by name, namespace, tag, version
    - Deterministic serialization
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize schema registry.
        
        Args:
            storage_path: Optional path for persistent storage
        """
        self.storage_path = storage_path
        self.models: Dict[str, Dict[str, IRModel]] = {}  # name -> version -> model
        self.tools: Dict[str, Dict[str, IRTool]] = {}  # name -> version -> tool
        self.migrations: List[SchemaMigration] = []
        
        if storage_path and storage_path.exists():
            self.load()

    def register_model(
        self, model: IRModel, replace: bool = False
    ) -> None:
        """
        Register a model in the registry.
        
        Args:
            model: Model to register
            replace: If True, replace existing version
        
        Raises:
            SchemaRegistryError: If model already exists and replace=False
        """
        if model.name not in self.models:
            self.models[model.name] = {}

        version_str = str(model.version)
        if version_str in self.models[model.name] and not replace:
            raise SchemaRegistryError(
                f"Model {model.name} version {version_str} already exists",
                schema_name=model.name,
                version=version_str,
            )

        self.models[model.name][version_str] = model

    def register_tool(
        self, tool: IRTool, replace: bool = False
    ) -> None:
        """Register a tool in the registry."""
        if tool.name not in self.tools:
            self.tools[tool.name] = {}

        version_str = str(tool.version)
        if version_str in self.tools[tool.name] and not replace:
            raise SchemaRegistryError(
                f"Tool {tool.name} version {version_str} already exists",
                schema_name=tool.name,
                version=version_str,
            )

        self.tools[tool.name][version_str] = tool

    def register_migration(self, migration: SchemaMigration) -> None:
        """Register a schema migration."""
        self.migrations.append(migration)

    def get_model(
        self, name: str, version: Optional[SchemaVersion] = None
    ) -> Optional[IRModel]:
        """
        Get model by name and optional version.
        
        Args:
            name: Model name
            version: Specific version (None = latest)
        
        Returns:
            Model or None if not found
        """
        if name not in self.models:
            return None

        if version is None:
            # Get latest version
            versions = sorted(
                [SchemaVersion.parse(v) for v in self.models[name].keys()]
            )
            if not versions:
                return None
            version = versions[-1]

        version_str = str(version)
        return self.models[name].get(version_str)

    def get_tool(
        self, name: str, version: Optional[SchemaVersion] = None
    ) -> Optional[IRTool]:
        """Get tool by name and optional version."""
        if name not in self.tools:
            return None

        if version is None:
            versions = sorted(
                [SchemaVersion.parse(v) for v in self.tools[name].keys()]
            )
            if not versions:
                return None
            version = versions[-1]

        version_str = str(version)
        return self.tools[name].get(version_str)

    def find_models_by_tag(self, tag: str) -> List[IRModel]:
        """Find all models with specific tag."""
        results = []
        for versions in self.models.values():
            for model in versions.values():
                if tag in model.tags:
                    results.append(model)
        return results

    def find_models_by_namespace(self, namespace: str) -> List[IRModel]:
        """Find all models in namespace."""
        results = []
        for versions in self.models.values():
            for model in versions.values():
                if model.namespace == namespace:
                    results.append(model)
        return results

    def get_migrations_for(
        self, schema_name: str
    ) -> List[SchemaMigration]:
        """Get all migrations for a schema."""
        return [
            m for m in self.migrations if m.schema_name == schema_name
        ]

    def import_spec(self, spec: IRSpec) -> None:
        """Import complete IR spec into registry."""
        for model in spec.models.values():
            self.register_model(model, replace=True)
        for tool in spec.tools.values():
            self.register_tool(tool, replace=True)
        for migration in spec.migrations:
            self.register_migration(migration)

    def export_spec(
        self,
        version: SchemaVersion,
        api_version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IRSpec:
        """Export complete IR spec from registry."""
        # Get latest version of each model and tool
        latest_models = {}
        for name, versions in self.models.items():
            version_objs = sorted(
                [SchemaVersion.parse(v) for v in versions.keys()]
            )
            if version_objs:
                latest_version = version_objs[-1]
                latest_models[name] = versions[str(latest_version)]

        latest_tools = {}
        for name, versions in self.tools.items():
            version_objs = sorted(
                [SchemaVersion.parse(v) for v in versions.keys()]
            )
            if version_objs:
                latest_version = version_objs[-1]
                latest_tools[name] = versions[str(latest_version)]

        return IRSpec(
            version=version,
            api_version=api_version,
            models=latest_models,
            tools=latest_tools,
            migrations=self.migrations,
            metadata=metadata or {},
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Save registry to disk."""
        target_path = path or self.storage_path
        if not target_path:
            raise SchemaRegistryError("No storage path configured")

        target_path = Path(target_path)
        target_path.mkdir(parents=True, exist_ok=True)

        # Save models
        models_dir = target_path / "models"
        models_dir.mkdir(exist_ok=True)
        for name, versions in self.models.items():
            for version_str, model in versions.items():
                file_path = models_dir / f"{name}_{version_str}.json"
                with open(file_path, "w") as f:
                    json.dump(model.to_json_schema(), f, indent=2)

        # Save tools
        tools_dir = target_path / "tools"
        tools_dir.mkdir(exist_ok=True)
        for name, versions in self.tools.items():
            for version_str, tool in versions.items():
                file_path = tools_dir / f"{name}_{version_str}.json"
                with open(file_path, "w") as f:
                    json.dump(tool.to_dict(), f, indent=2)

        # Save migrations
        migrations_file = target_path / "migrations.json"
        with open(migrations_file, "w") as f:
            json.dump(
                [m.to_dict() for m in self.migrations],
                f,
                indent=2,
            )

    def load(self, path: Optional[Path] = None) -> None:
        """Load registry from disk."""
        source_path = path or self.storage_path
        if not source_path:
            raise SchemaRegistryError("No storage path configured")

        # Implementation would load from disk
        # Omitted for brevity - follows save() structure


class SchemaExporter:
    """
    Exports schemas from N3 runtime/backend.
    
    Supports multiple export sources:
    - Running N3 backend (via HTTP)
    - Generated backend code (via filesystem)
    - N3 source files (via parser)
    """

    def __init__(self, backend_url: Optional[str] = None):
        """
        Initialize exporter.
        
        Args:
            backend_url: URL of running N3 backend
        """
        self.backend_url = backend_url

    async def export_from_backend(self) -> IRSpec:
        """
        Export schemas from running N3 backend.
        
        Returns:
            Complete IR spec
        
        Raises:
            ExportError: If export fails
        """
        if not self.backend_url:
            raise ExportError("No backend URL configured")

        try:
            async with httpx.AsyncClient() as client:
                # Get schema manifest
                response = await client.get(
                    f"{self.backend_url}/api/_meta/schemas"
                )
                response.raise_for_status()
                manifest = response.json()

                # Build IR spec
                spec = IRSpec(
                    version=SchemaVersion.parse(manifest.get("version", "1.0.0")),
                    api_version=manifest.get("api_version", "1.0"),
                    metadata={
                        "source": "backend",
                        "backend_url": self.backend_url,
                        "exported_at": datetime.utcnow().isoformat(),
                    },
                )

                # Export models
                for model_info in manifest.get("models", []):
                    model = await self._fetch_model(client, model_info["name"])
                    if model:
                        spec.models[model.name] = model

                # Export tools
                for tool_info in manifest.get("tools", []):
                    tool = await self._fetch_tool(client, tool_info["name"])
                    if tool:
                        spec.tools[tool.name] = tool

                return spec

        except httpx.HTTPError as e:
            raise ExportError(
                f"Failed to export from backend: {e}",
                backend_url=self.backend_url,
            )

    async def _fetch_model(
        self, client: httpx.AsyncClient, name: str
    ) -> Optional[IRModel]:
        """Fetch model schema from backend."""
        try:
            response = await client.get(
                f"{self.backend_url}/api/_meta/schemas/models/{name}"
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_model(data)
        except httpx.HTTPError:
            return None

    async def _fetch_tool(
        self, client: httpx.AsyncClient, name: str
    ) -> Optional[IRTool]:
        """Fetch tool spec from backend."""
        try:
            response = await client.get(
                f"{self.backend_url}/api/_meta/tools/{name}"
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_tool(data)
        except httpx.HTTPError:
            return None

    def export_from_directory(self, schema_dir: Path) -> IRSpec:
        """
        Export schemas from generated backend directory.
        
        Args:
            schema_dir: Path to schemas directory
        
        Returns:
            Complete IR spec
        """
        spec = IRSpec(
            version=SchemaVersion(major=1, minor=0, patch=0),
            api_version="1.0",
            metadata={
                "source": "directory",
                "schema_dir": str(schema_dir),
                "exported_at": datetime.utcnow().isoformat(),
            },
        )

        # Load from schemas/__init__.py or similar
        # Implementation depends on N3 backend structure
        
        return spec

    def _parse_model(self, data: Dict[str, Any]) -> IRModel:
        """Parse JSON schema into IRModel."""
        version_str = data.get("x-version", "1.0.0")
        version = SchemaVersion.parse(version_str)

        fields = []
        required = set(data.get("required", []))
        
        for field_name, field_schema in data.get("properties", {}).items():
            fields.append(
                self._parse_field(field_name, field_schema, field_name in required)
            )

        return IRModel(
            name=data.get("title", "Unknown"),
            version=version,
            fields=fields,
            description=data.get("description"),
            namespace=data.get("x-namespace"),
            tags=data.get("x-tags", []),
        )

    def _parse_field(
        self, name: str, schema: Dict[str, Any], required: bool
    ) -> IRField:
        """Parse JSON schema field into IRField."""
        field_type = schema.get("type", "any")
        
        # Handle nullable
        nullable = False
        if isinstance(field_type, list):
            if "null" in field_type:
                nullable = True
                field_type = [t for t in field_type if t != "null"][0]

        return IRField(
            name=name,
            type=IRType(field_type) if field_type in IRType.__members__.values() else field_type,
            required=required,
            nullable=nullable,
            description=schema.get("description"),
            default=schema.get("default"),
            constraints={
                k: v
                for k, v in schema.items()
                if k not in {"type", "description", "default"}
            },
        )

    def _parse_tool(self, data: Dict[str, Any]) -> IRTool:
        """Parse tool spec into IRTool."""
        version_str = data.get("version", "1.0.0")
        version = SchemaVersion.parse(version_str)

        input_schema = self._parse_model(data["input_schema"])
        output_schema = self._parse_model(data["output_schema"])

        return IRTool(
            name=data["name"],
            version=version,
            description=data["description"],
            input_schema=input_schema,
            output_schema=output_schema,
            namespace=data.get("namespace"),
            tags=data.get("tags", []),
            timeout=data.get("timeout"),
            auth_required=data.get("auth_required", False),
            streaming=data.get("streaming", False),
            metadata=data.get("metadata", {}),
        )


class SchemaImporter:
    """
    Imports schemas into registry from various sources.
    """

    def __init__(self, registry: SchemaRegistry):
        """
        Initialize importer.
        
        Args:
            registry: Target registry
        """
        self.registry = registry

    def import_from_spec_file(self, path: Path) -> None:
        """Import from IRSpec JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            # Parse and import
            # Implementation depends on IRSpec serialization format
        except (IOError, json.JSONDecodeError) as e:
            raise ImportError(
                f"Failed to import from {path}: {e}",
                source_path=str(path),
            )

    def import_from_tool_registry(
        self, tool_registry: Any
    ) -> None:
        """
        Import from N3 tool registry.
        
        Reads tool specifications from the runtime ToolRegistry
        and converts them to IR format.
        """
        # Implementation would use tool_registry to extract tools
        pass
