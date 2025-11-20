"""
OpenAPI adapter for importing tools from OpenAPI specifications.

Parses OpenAPI 3.0/3.1 specs and generates executable tool wrappers
that can be registered in the ToolRegistry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urljoin

import httpx
import yaml


@dataclass
class OpenAPIParameter:
    """Parameter definition from OpenAPI spec."""
    name: str
    location: str  # path, query, header, cookie
    schema: Dict[str, Any]
    required: bool = False
    description: str = ""


@dataclass
class OpenAPIOperation:
    """Operation (endpoint) definition from OpenAPI spec."""
    operation_id: str
    method: str  # GET, POST, PUT, DELETE, etc.
    path: str
    summary: str = ""
    description: str = ""
    parameters: List[OpenAPIParameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class OpenAPIToolConfig:
    """Configuration for OpenAPI tool generation."""
    spec_url: Optional[str] = None
    spec_dict: Optional[Dict[str, Any]] = None
    base_url: Optional[str] = None
    auth_header: Optional[str] = None
    auth_token: Optional[str] = None
    timeout: float = 30.0
    operation_filter: Optional[Callable[[OpenAPIOperation], bool]] = None
    name_prefix: str = ""


class OpenAPIAdapter:
    """
    Adapter for importing tools from OpenAPI specifications.
    
    Features:
    - Parses OpenAPI 3.0 and 3.1 specs (JSON or YAML)
    - Generates executable tool wrappers for each operation
    - Handles authentication (Bearer, API Key, etc.)
    - Converts OpenAPI schemas to tool input/output schemas
    - Supports path parameters, query parameters, request bodies
    
    Example:
        >>> adapter = OpenAPIAdapter()
        >>> tools = await adapter.import_from_url(
        ...     "https://api.example.com/openapi.json",
        ...     base_url="https://api.example.com",
        ...     auth_token="sk-..."
        ... )
        >>> for tool in tools:
        ...     registry.register_tool(tool)
    """
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client
    
    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    async def import_from_url(
        self,
        spec_url: str,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        operation_filter: Optional[Callable[[OpenAPIOperation], bool]] = None,
        name_prefix: str = "",
    ) -> List[Callable]:
        """
        Import tools from OpenAPI spec URL.
        
        Args:
            spec_url: URL to OpenAPI spec (JSON or YAML)
            base_url: Base URL for API calls (overrides spec servers)
            auth_token: Bearer token or API key for authentication
            operation_filter: Optional filter function for operations
            name_prefix: Prefix for tool names
        
        Returns:
            List of tool functions ready for registration
        """
        client = await self._get_client()
        
        # Fetch spec
        response = await client.get(spec_url)
        response.raise_for_status()
        
        # Parse spec (try JSON first, then YAML)
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            spec_dict = response.json()
        else:
            spec_dict = yaml.safe_load(response.text)
        
        return await self.import_from_dict(
            spec_dict=spec_dict,
            base_url=base_url,
            auth_token=auth_token,
            operation_filter=operation_filter,
            name_prefix=name_prefix,
        )
    
    async def import_from_dict(
        self,
        spec_dict: Dict[str, Any],
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        operation_filter: Optional[Callable[[OpenAPIOperation], bool]] = None,
        name_prefix: str = "",
    ) -> List[Callable]:
        """
        Import tools from OpenAPI spec dictionary.
        
        Args:
            spec_dict: OpenAPI specification dictionary
            base_url: Base URL for API calls
            auth_token: Authentication token
            operation_filter: Optional filter for operations
            name_prefix: Prefix for tool names
        
        Returns:
            List of tool functions
        """
        # Extract base URL from spec if not provided
        if base_url is None:
            servers = spec_dict.get("servers", [])
            if servers:
                base_url = servers[0].get("url", "")
        
        if not base_url:
            raise ValueError("Base URL must be provided or present in OpenAPI spec")
        
        # Parse operations
        operations = self._parse_operations(spec_dict)
        
        # Filter operations if filter provided
        if operation_filter:
            operations = [op for op in operations if operation_filter(op)]
        
        # Generate tool functions
        tools = []
        for operation in operations:
            tool_func = self._create_tool_function(
                operation=operation,
                base_url=base_url,
                auth_token=auth_token,
                name_prefix=name_prefix,
            )
            tools.append(tool_func)
        
        return tools
    
    def _parse_operations(self, spec_dict: Dict[str, Any]) -> List[OpenAPIOperation]:
        """Parse all operations from OpenAPI spec."""
        operations = []
        
        paths = spec_dict.get("paths", {})
        for path, path_item in paths.items():
            # Extract parameters at path level
            path_params = [
                self._parse_parameter(p) for p in path_item.get("parameters", [])
            ]
            
            # Parse each HTTP method
            for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
                if method not in path_item:
                    continue
                
                op_spec = path_item[method]
                operation_id = op_spec.get("operationId", f"{method}_{path.replace('/', '_')}")
                
                # Combine path-level and operation-level parameters
                op_params = path_params + [
                    self._parse_parameter(p) for p in op_spec.get("parameters", [])
                ]
                
                operation = OpenAPIOperation(
                    operation_id=operation_id,
                    method=method.upper(),
                    path=path,
                    summary=op_spec.get("summary", ""),
                    description=op_spec.get("description", ""),
                    parameters=op_params,
                    request_body=op_spec.get("requestBody"),
                    responses=op_spec.get("responses", {}),
                    tags=op_spec.get("tags", []),
                )
                
                operations.append(operation)
        
        return operations
    
    def _parse_parameter(self, param_spec: Dict[str, Any]) -> OpenAPIParameter:
        """Parse a parameter definition."""
        return OpenAPIParameter(
            name=param_spec["name"],
            location=param_spec["in"],
            schema=param_spec.get("schema", {}),
            required=param_spec.get("required", False),
            description=param_spec.get("description", ""),
        )
    
    def _create_tool_function(
        self,
        operation: OpenAPIOperation,
        base_url: str,
        auth_token: Optional[str],
        name_prefix: str,
    ) -> Callable:
        """Create an executable tool function for an operation."""
        
        async def tool_func(**kwargs) -> Dict[str, Any]:
            """Generated tool function for OpenAPI operation."""
            client = await self._get_client()
            
            # Build URL with path parameters
            url = base_url.rstrip("/") + operation.path
            for param in operation.parameters:
                if param.location == "path" and param.name in kwargs:
                    url = url.replace(f"{{{param.name}}}", str(kwargs[param.name]))
            
            # Build query parameters
            query_params = {}
            for param in operation.parameters:
                if param.location == "query" and param.name in kwargs:
                    query_params[param.name] = kwargs[param.name]
            
            # Build headers
            headers = {}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            for param in operation.parameters:
                if param.location == "header" and param.name in kwargs:
                    headers[param.name] = str(kwargs[param.name])
            
            # Build request body
            json_body = None
            if operation.request_body and "body" in kwargs:
                json_body = kwargs["body"]
            
            # Make request
            response = await client.request(
                method=operation.method,
                url=url,
                params=query_params,
                headers=headers,
                json=json_body,
            )
            
            response.raise_for_status()
            
            # Parse response
            content_type = response.headers.get("content-type", "")
            if "json" in content_type:
                return response.json()
            else:
                return {"text": response.text}
        
        # Set function metadata
        tool_name = f"{name_prefix}{operation.operation_id}"
        tool_func.__name__ = tool_name
        tool_func.__doc__ = operation.description or operation.summary
        
        # Attach metadata for ToolRegistry
        tool_func._tool_metadata = {
            "name": tool_name,
            "description": operation.description or operation.summary,
            "input_schema": self._build_input_schema(operation),
            "output_schema": self._build_output_schema(operation),
            "tags": operation.tags + ["openapi"],
            "source": "openapi",
        }
        
        return tool_func
    
    def _build_input_schema(self, operation: OpenAPIOperation) -> Dict[str, Any]:
        """Build JSON schema for tool inputs."""
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        
        # Add parameters
        for param in operation.parameters:
            schema["properties"][param.name] = {
                **param.schema,
                "description": param.description,
            }
            if param.required:
                schema["required"].append(param.name)
        
        # Add request body if present
        if operation.request_body:
            schema["properties"]["body"] = {
                "type": "object",
                "description": "Request body",
            }
            if operation.request_body.get("required", False):
                schema["required"].append("body")
        
        return schema
    
    def _build_output_schema(self, operation: OpenAPIOperation) -> Dict[str, Any]:
        """Build JSON schema for tool outputs."""
        # Try to extract schema from 200/201 response
        for status_code in ["200", "201"]:
            if status_code in operation.responses:
                response_spec = operation.responses[status_code]
                content = response_spec.get("content", {})
                if "application/json" in content:
                    return content["application/json"].get("schema", {})
        
        # Fallback to generic schema
        return {"type": "object"}
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._http_client:
            # Cannot await in __del__, so just set to None
            self._http_client = None
