"""HTTP tool implementation for making web API calls."""

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import BaseTool, ToolError, ToolResult
from .errors import ToolExecutionError, ToolValidationError
from .validation import validate_http_endpoint, validate_http_method, validate_http_headers


class HttpTool(BaseTool):
    """
    Tool for making HTTP/REST API calls.
    
    Supports GET, POST, PUT, DELETE, PATCH methods with headers, query params, and body.
    Handles JSON serialization/deserialization automatically.
    
    Features:
        - Automatic JSON encoding/decoding
        - Custom headers per request
        - Query parameter support
        - Timeout handling
        - Error context in results
    
    Configuration:
        name: Tool identifier
        endpoint: Base URL (https://api.example.com/v1)
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        headers: Default headers (Authorization, Content-Type, etc.)
        timeout: Request timeout in seconds
    
    Input Schema:
        query: Dict of query parameters (optional)
        body/data: Request body for POST/PUT/PATCH (optional)
        headers: Additional headers for this request (optional)
    
    Output:
        Parsed JSON response or raw text if not JSON
    
    Example:
        >>> from namel3ss.tools import create_tool
        >>> tool = create_tool(
        ...     name=\"weather_api\",
        ...     tool_type=\"http\",
        ...     endpoint=\"https://api.weather.com/v1/current\",
        ...     method=\"GET\",
        ...     headers={\"API-Key\": \"your-key\"},
        ...     timeout=10.0
        ... )
        >>> result = tool.execute(query={\"location\": \"NYC\", \"units\": \"metric\"})
        >>> if result.success:
        ...     print(f\"Temperature: {result.output['temp']}°C\")
    
    Error Handling:
        - HTTP errors (4xx, 5xx): Returns ToolResult(success=False)
        - Network errors: Returns ToolResult(success=False)
        - Timeout: Returns ToolResult(success=False)
        - Configuration errors: Raises ToolError
    
    Best Practices:
        - Set API keys in environment variables, not hardcoded
        - Use appropriate timeouts (default 30s may be too long)
        - Handle both success and error cases
        - Check metadata.status_code for HTTP details
        - Include retry logic for transient failures (5xx)
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
        """Initialize HTTP tool.
        
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
            raise ToolValidationError(
                "HTTP tool requires 'endpoint' parameter",
                code="TOOL013",
                tool_name=name,
                field="endpoint",
                tool_type="http",
            )
        
        # Validate configuration
        validate_http_endpoint(endpoint, tool_name=name)
        validate_http_method(method, tool_name=name)
        if headers:
            validate_http_headers(headers, tool_name=name)
        
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        
        # Validate method
        if self.method not in {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}:
            raise ToolValidationError(
                f"Invalid HTTP method: {self.method}",
                code="TOOL012",
                tool_name=name,
                field="method",
                value=self.method,
                expected="GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS",
                tool_type="http",
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
                    "retryable": e.code >= 500,  # 5xx errors are retryable
                },
            )
        
        except URLError as e:
            return ToolResult(
                output=None,
                success=False,
                error=f"URL error: {e.reason}",
                metadata={
                    "url": url,
                    "method": self.method,
                    "retryable": True,  # Network errors are retryable
                },
            )
        
        except Exception as e:
            raise ToolExecutionError(
                f"Failed to execute HTTP tool: {e}",
                code="TOOL031",
                tool_name=self.name,
                operation=f"{self.method} {url}",
                original_error=e,
            )
    
    def get_tool_type(self) -> str:
        """Return 'http'."""
        return "http"
