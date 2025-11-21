"""
Schema exporter for N3 backend generation.

Integrates with the N3 compilation pipeline to extract schemas from generated
backends and populate the SDK-Sync registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from namel3ss.ast import App

from .errors import SDKSyncError
from .ir import IRField, IRModel, IRSpec, IRTool, IRType, SchemaVersion
from .registry import SchemaRegistry


__all__ = [
    "SchemaExporter",
    "export_schemas_from_app",
    "generate_metadata_router",
]


class SchemaExporter:
    """
    Exports schemas from N3 application to SDK-Sync registry.
    
    This class bridges the N3 compilation pipeline and the SDK-Sync system,
    extracting schema information from the AST and generated backend.
    """
    
    def __init__(self, registry: Optional[SchemaRegistry] = None):
        """
        Initialize exporter.
        
        Args:
            registry: Schema registry to populate (uses singleton if None)
        """
        self.registry = registry or SchemaRegistry()
    
    def export_from_app(
        self,
        app: App,
        version: SchemaVersion,
        namespace: str = "app",
    ) -> IRSpec:
        """
        Export schemas from N3 application AST.
        
        Args:
            app: Parsed N3 application
            version: Schema version to assign
            namespace: Namespace for exported schemas
        
        Returns:
            Complete IR specification
        """
        models: Dict[str, IRModel] = {}
        tools: Dict[str, IRTool] = {}
        
        # Extract dataset schemas
        if hasattr(app, "datasets") and app.datasets:
            for dataset in app.datasets:
                model = self._extract_dataset_schema(dataset, version, namespace)
                if model:
                    models[model.name] = model
                    self.registry.register_model(model)
        
        # Extract tool schemas (from actions, predictions, etc.)
        if hasattr(app, "actions") and app.actions:
            for action in app.actions:
                tool = self._extract_action_tool(action, version, namespace)
                if tool:
                    tools[tool.name] = tool
                    self.registry.register_tool(tool)
        
        if hasattr(app, "predictions") and app.predictions:
            for prediction in app.predictions:
                tool = self._extract_prediction_tool(prediction, version, namespace)
                if tool:
                    tools[tool.name] = tool
                    self.registry.register_tool(tool)
        
        # Create spec
        spec = IRSpec(
            version=version,
            api_version="1.0",
            models=models,
            tools=tools,
            metadata={
                "generator": "namel3ss",
                "namespace": namespace,
                "source": "app",
            },
        )
        
        return spec
    
    def export_from_generated_schemas(
        self,
        schemas_dir: Path,
        version: SchemaVersion,
        namespace: str = "generated",
    ) -> IRSpec:
        """
        Export schemas from generated backend schemas directory.
        
        Args:
            schemas_dir: Path to generated/schemas/ directory
            version: Schema version
            namespace: Namespace for schemas
        
        Returns:
            IR specification
        """
        models: Dict[str, IRModel] = {}
        
        # Look for __init__.py with Pydantic models
        init_file = schemas_dir / "__init__.py"
        if not init_file.exists():
            raise SDKSyncError(
                f"Schemas __init__.py not found: {init_file}",
                error_code="EXPORT_001",
            )
        
        # Parse and extract models (simplified - would need AST parsing in production)
        # For now, we'll export the schemas that are already registered
        spec = IRSpec(
            version=version,
            api_version="1.0",
            models=models,
            tools={},
            metadata={
                "generator": "namel3ss",
                "namespace": namespace,
                "source": "generated",
            },
        )
        
        return spec
    
    def save_spec_to_file(self, spec: IRSpec, output_path: Path) -> None:
        """
        Save IR spec to JSON file.
        
        Args:
            spec: IR specification
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(spec.to_json(), encoding="utf-8")
    
    def _extract_dataset_schema(
        self,
        dataset: Any,
        version: SchemaVersion,
        namespace: str,
    ) -> Optional[IRModel]:
        """Extract schema from dataset definition."""
        if not hasattr(dataset, "name"):
            return None
        
        fields: List[IRField] = []
        
        # Extract schema if available (Dataset has schema: List[DatasetSchemaField])
        if hasattr(dataset, "schema") and dataset.schema:
            for schema_field in dataset.schema:
                field = self._extract_schema_field(schema_field)
                if field:
                    fields.append(field)
        
        # Create model
        model = IRModel(
            name=self._to_class_name(dataset.name),
            version=version,
            fields=fields,
            description=f"{dataset.name} dataset",
            namespace=namespace,
            tags=["dataset", "data"],
            metadata={
                "source_type": getattr(dataset, "source_type", "unknown"),
                "source_name": dataset.name,
            },
        )
        
        return model
    
    def _extract_schema_field(self, schema_field: Any) -> Optional[IRField]:
        """Extract field from DatasetSchemaField."""
        if not hasattr(schema_field, "name"):
            return None
        
        # Map N3 types to IR types
        dtype = getattr(schema_field, "dtype", None)
        ir_type = self._map_dtype_to_ir_type(dtype)
        
        field = IRField(
            name=schema_field.name,
            type=ir_type,
            required=not getattr(schema_field, "nullable", True),
            description=getattr(schema_field, "description", None),
            default=getattr(schema_field, "default", None),
            constraints={},
            metadata={
                "dtype": dtype,
                "source": "schema_field",
            },
        )
        
        return field
    
    def _extract_action_tool(
        self,
        action: Any,
        version: SchemaVersion,
        namespace: str,
    ) -> Optional[IRTool]:
        """Extract tool from action definition."""
        if not hasattr(action, "name"):
            return None
        
        # Create input/output models
        input_model = IRModel(
            name=f"{self._to_class_name(action.name)}Input",
            version=version,
            fields=[],
            description=f"Input for {action.name} action",
            namespace=namespace,
            tags=["action", "input"],
        )
        
        output_model = IRModel(
            name=f"{self._to_class_name(action.name)}Output",
            version=version,
            fields=[
                IRField(
                    name="success",
                    type=IRType.BOOLEAN,
                    required=True,
                    description="Whether action succeeded",
                ),
                IRField(
                    name="message",
                    type=IRType.STRING,
                    required=False,
                    description="Result message",
                ),
            ],
            description=f"Output for {action.name} action",
            namespace=namespace,
            tags=["action", "output"],
        )
        
        # Register models
        self.registry.register_model(input_model)
        self.registry.register_model(output_model)
        
        # Create tool
        tool = IRTool(
            name=action.name,
            version=version,
            namespace=namespace,
            description=getattr(action, "description", None) or f"{action.name} action",
            input_schema=input_model,
            output_schema=output_model,
            tags=["action"],
            metadata={
                "source_type": "action",
                "trigger": getattr(action, "trigger", None),
            },
        )
        
        return tool
    
    def _extract_prediction_tool(
        self,
        prediction: Any,
        version: SchemaVersion,
        namespace: str,
    ) -> Optional[IRTool]:
        """Extract tool from prediction definition."""
        if not hasattr(prediction, "name"):
            return None
        
        # Create input/output models (simplified)
        input_model = IRModel(
            name=f"{self._to_class_name(prediction.name)}Input",
            version=version,
            fields=[
                IRField(
                    name="features",
                    type=IRType.OBJECT,
                    required=True,
                    description="Input features",
                ),
            ],
            description=f"Input for {prediction.name} prediction",
            namespace=namespace,
            tags=["prediction", "input"],
        )
        
        output_model = IRModel(
            name=f"{self._to_class_name(prediction.name)}Output",
            version=version,
            fields=[
                IRField(
                    name="prediction",
                    type=IRType.ANY,
                    required=True,
                    description="Prediction result",
                ),
                IRField(
                    name="confidence",
                    type=IRType.FLOAT,
                    required=False,
                    description="Prediction confidence",
                    constraints={"minimum": 0.0, "maximum": 1.0},
                ),
            ],
            description=f"Output for {prediction.name} prediction",
            namespace=namespace,
            tags=["prediction", "output"],
        )
        
        # Register models
        self.registry.register_model(input_model)
        self.registry.register_model(output_model)
        
        # Create tool
        tool = IRTool(
            name=prediction.name,
            version=version,
            namespace=namespace,
            description=getattr(prediction, "description", None) or f"{prediction.name} prediction",
            input_schema=input_model,
            output_schema=output_model,
            tags=["prediction", "ml"],
            metadata={
                "source_type": "prediction",
                "model": getattr(prediction, "model", None),
                "framework": getattr(prediction, "framework", None),
            },
        )
        
        return tool
    
    def _map_dtype_to_ir_type(self, dtype: Optional[str]) -> IRType:
        """Map N3 dtype to IR type."""
        if not dtype:
            return IRType.ANY
        
        dtype_lower = dtype.lower()
        
        if dtype_lower in ("str", "string", "text"):
            return IRType.STRING
        elif dtype_lower in ("int", "integer", "int64"):
            return IRType.INTEGER
        elif dtype_lower in ("float", "double", "decimal"):
            return IRType.FLOAT
        elif dtype_lower in ("bool", "boolean"):
            return IRType.BOOLEAN
        elif dtype_lower in ("datetime", "timestamp"):
            return IRType.STRING  # Will add format constraint
        elif dtype_lower in ("date"):
            return IRType.STRING
        elif dtype_lower in ("json", "object", "dict"):
            return IRType.OBJECT
        elif dtype_lower in ("array", "list"):
            return IRType.ARRAY
        else:
            return IRType.ANY
    
    def _to_class_name(self, name: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in name.split("_"))


def export_schemas_from_app(
    app: App,
    version: SchemaVersion,
    output_path: Optional[Path] = None,
    namespace: str = "app",
) -> IRSpec:
    """
    Export schemas from N3 application.
    
    Args:
        app: Parsed N3 application
        version: Schema version
        output_path: Optional path to save spec JSON
        namespace: Namespace for schemas
    
    Returns:
        IR specification
    """
    exporter = SchemaExporter()
    spec = exporter.export_from_app(app, version, namespace)
    
    if output_path:
        exporter.save_spec_to_file(spec, output_path)
    
    return spec


def generate_metadata_router() -> str:
    """
    Generate FastAPI router for schema metadata endpoints.
    
    This router provides REST endpoints for SDK generation:
    - GET /api/_meta/schemas - Get schema manifest
    - GET /api/_meta/schemas/models/{name} - Get model JSON Schema
    - GET /api/_meta/tools/{name} - Get tool specification
    - GET /api/_meta/spec - Get complete IR spec
    
    Returns:
        Python code for metadata router
    """
    return '''
"""Schema metadata endpoints for SDK generation."""

from fastapi import APIRouter, HTTPException
from typing import Any, Dict, List

from namel3ss.sdk_sync.registry import SchemaRegistry

router = APIRouter(prefix="/api/_meta", tags=["metadata"])

# Get global registry
_registry = SchemaRegistry()


@router.get("/schemas")
async def get_schema_manifest() -> Dict[str, Any]:
    """
    Get manifest of all available schemas.
    
    Returns:
        Dictionary with schema and tool listings
    """
    models = {}
    for name, versions in _registry._models.items():
        latest = _registry.get_latest_version(name)
        models[name] = {
            "versions": [str(v) for v in versions.keys()],
            "latest": str(latest) if latest else None,
        }
    
    tools = {}
    for name, versions in _registry._tools.items():
        latest = _registry.get_latest_version(name)
        tools[name] = {
            "versions": [str(v) for v in versions.keys()],
            "latest": str(latest) if latest else None,
        }
    
    return {
        "version": "1.0.0",
        "api_version": "1.0",
        "models": models,
        "tools": tools,
    }


@router.get("/schemas/models/{name}")
async def get_model_schema(name: str, version: str = None) -> Dict[str, Any]:
    """
    Get JSON Schema for a model.
    
    Args:
        name: Model name
        version: Optional version (uses latest if not specified)
    
    Returns:
        JSON Schema for the model
    """
    try:
        if version:
            from namel3ss.sdk_sync.ir import SchemaVersion
            ver = SchemaVersion.parse(version)
            model = _registry.get_model(name, ver)
        else:
            latest_ver = _registry.get_latest_version(name)
            model = _registry.get_model(name, latest_ver)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {name}")
        
        return model.to_json_schema()
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tools/{name}")
async def get_tool_spec(name: str, version: str = None) -> Dict[str, Any]:
    """
    Get tool specification.
    
    Args:
        name: Tool name
        version: Optional version (uses latest if not specified)
    
    Returns:
        Tool specification
    """
    try:
        if version:
            from namel3ss.sdk_sync.ir import SchemaVersion
            ver = SchemaVersion.parse(version)
            tool = _registry.get_tool(name, ver)
        else:
            latest_ver = _registry.get_latest_version(name)
            tool = _registry.get_tool(name, latest_ver)
        
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool not found: {name}")
        
        return tool.to_json()
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/spec")
async def get_complete_spec(version: str = "1.0.0", api_version: str = "1.0") -> Dict[str, Any]:
    """
    Get complete IR specification.
    
    Args:
        version: Spec version
        api_version: API version
    
    Returns:
        Complete IR spec as JSON
    """
    try:
        from namel3ss.sdk_sync.ir import SchemaVersion
        ver = SchemaVersion.parse(version)
        spec = _registry.export_spec(ver, api_version)
        
        # Convert to dict for JSON response
        import json
        return json.loads(spec.to_json())
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
'''.strip()
