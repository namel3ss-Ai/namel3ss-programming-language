"""HTTP tool implementation for making web API calls."""

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import BaseTool, ToolError, ToolResult


class HttpTool(BaseTool):
    """
    Tool for making HTTP/REST API calls.
    
    Supports GET, POST, PUT, DELETE methods with headers, query params, and body.
    """
    
    def __init__(
        self,
        *,
        name: str,
        tool_type: str = "http",
        endpoint: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        **config: Any,
    ):
        """
        Initialize HTTP tool.
        
        Args:
            name: Tool identifier
            tool_type: Always "http"
            endpoint: Base URL for the API
            method: HTTP method (GET, POST, PUT, DELETE)
            headers: Default headers to send
            input_schema: Schema for inputs
            output_schema: Schema for outputs  
            timeout: Request timeout in seconds
            **config: Additional configuration
        """
        super().__init__(
            name=name,
            tool_type=tool_type,
            input_schema=input_schema,
            output_schema=output_schema,
            timeout=timeout,
            **config,
        )
        
        if not endpoint:
            raise ToolError("HTTP tool requires 'endpoint' parameter", tool_name=name)
        
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        
        # Validate method
        if self.method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
            raise ToolError(
                f"Invalid HTTP method: {self.method}",
                tool_name=name,
            )
    
    def execute(self, **inputs: Any) -> ToolResult:
        """
        Execute the HTTP request.
        
        Args:
            **inputs: Request parameters (query params, body, etc.)
            
        Returns:
            ToolResult with response data
            
        Raises:
            ToolError: If request fails
        """
        try:
            self.validate_inputs(inputs)
            
            # Build URL with query params
            url = self.endpoint
            query_params = inputs.get("query", {})
            if query_params:
                query_string = "&".join(
                    f"{k}={v}" for k, v in query_params.items()
                )
                url = f"{url}?{query_string}"
            
            # Build request
            data = None
            headers = dict(self.headers)
            
            # Add body for POST/PUT/PATCH
            if self.method in {"POST", "PUT", "PATCH"}:
                body = inputs.get("body", inputs.get("data", {}))
                if body:
                    data = json.dumps(body).encode("utf-8")
                    headers.setdefault("Content-Type", "application/json")
            
            # Add any custom headers from inputs
            custom_headers = inputs.get("headers", {})
            headers.update(custom_headers)
            
            request = Request(
                url,
                data=data,
                headers=headers,
                method=self.method,
            )
            
            # Make request
            with urlopen(request, timeout=self.timeout) as response:
                response_data = response.read().decode("utf-8")
                
                # Try to parse as JSON
                try:
                    output = json.loads(response_data)
                except json.JSONDecodeError:
                    output = response_data
                
                return ToolResult(
                    output=output,
                    success=True,
                    metadata={
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "url": url,
                        "method": self.method,
                    },
                )
        
        except HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            return ToolResult(
                output=None,
                success=False,
                error=f"HTTP {e.code}: {error_body}",
                metadata={
                    "status_code": e.code,
                    "url": url,
                    "method": self.method,
                },
            )
        
        except URLError as e:
            return ToolResult(
                output=None,
                success=False,
                error=f"URL error: {e.reason}",
                metadata={"url": url, "method": self.method},
            )
        
        except Exception as e:
            raise ToolError(
                f"Failed to execute HTTP tool: {e}",
                tool_name=self.name,
                original_error=e,
            )
    
    def get_tool_type(self) -> str:
        """Return 'http'."""
        return "http"
