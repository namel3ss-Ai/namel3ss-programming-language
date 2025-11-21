"""HTTP adapter for calling REST and GraphQL APIs from N3.

Provides typed HTTP client with retry, timeout, and auth support.
"""

from enum import Enum
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field

from .base import (
    AdapterConfig,
    AdapterType,
    BaseAdapter,
    AdapterExecutionError,
    AdapterTimeoutError,
    AdapterValidationError,
)


class HttpMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class AuthType(str, Enum):
    """Authentication types."""
    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "api_key"


class HttpAdapterConfig(AdapterConfig):
    """Configuration for HTTP adapter."""
    
    adapter_type: AdapterType = Field(default=AdapterType.HTTP)
    
    # HTTP settings
    base_url: str = Field(..., description="Base URL for API")
    endpoint: str = Field(..., description="API endpoint path")
    method: HttpMethod = Field(default=HttpMethod.POST, description="HTTP method")
    
    # Authentication
    auth_type: AuthType = Field(default=AuthType.NONE, description="Auth type")
    auth_token: Optional[str] = Field(None, description="Bearer/API token")
    auth_username: Optional[str] = Field(None, description="Basic auth username")
    auth_password: Optional[str] = Field(None, description="Basic auth password")
    auth_header_name: Optional[str] = Field(None, description="API key header name")
    
    # Headers
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    
    # Request/response format
    request_format: str = Field(default="json", description="Request body format")
    response_format: str = Field(default="json", description="Response format")
    
    # TLS
    verify_ssl: bool = Field(default=True, description="Verify TLS certificates")


class HttpAdapter(BaseAdapter):
    """Adapter for calling HTTP APIs from N3.
    
    Supports REST and GraphQL with authentication, custom headers, and
    automatic retry/timeout.
    
    Features:
        - Multiple auth types (Bearer, Basic, API Key)
        - Custom headers
        - JSON, form, and raw body formats
        - Automatic retries on network errors
        - Request/response validation
    
    Example:
        REST API call:
        ```n3
        tool "fetch_weather" {
          adapter: "http"
          base_url: "https://api.weather.com"
          endpoint: "/v1/forecast"
          method: "GET"
          auth_type: "api_key"
          auth_header_name: "X-API-Key"
          auth_token: env("WEATHER_API_KEY")
        }
        ```
        
        Programmatic usage:
        >>> from namel3ss.adapters.http import HttpAdapter, HttpAdapterConfig
        >>> 
        >>> config = HttpAdapterConfig(
        ...     name="github_api",
        ...     base_url="https://api.github.com",
        ...     endpoint="/repos/{owner}/{repo}",
        ...     method="GET",
        ...     auth_type="bearer",
        ...     auth_token="ghp_..."
        ... )
        >>> adapter = HttpAdapter(config)
        >>> result = adapter.execute(owner="python", repo="cpython")
    """
    
    def __init__(self, config: HttpAdapterConfig):
        super().__init__(config)
        self.config: HttpAdapterConfig = config
        self._client: Optional[httpx.Client] = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup HTTP client with auth and headers."""
        headers = dict(self.config.headers)
        
        # Add authentication
        if self.config.auth_type == AuthType.BEARER:
            if not self.config.auth_token:
                raise AdapterValidationError(
                    "Bearer auth requires auth_token",
                    adapter_name=self.config.name,
                    adapter_type=self.config.adapter_type,
                )
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        
        elif self.config.auth_type == AuthType.API_KEY:
            if not self.config.auth_token or not self.config.auth_header_name:
                raise AdapterValidationError(
                    "API key auth requires auth_token and auth_header_name",
                    adapter_name=self.config.name,
                    adapter_type=self.config.adapter_type,
                )
            headers[self.config.auth_header_name] = self.config.auth_token
        
        # Create client
        auth = None
        if self.config.auth_type == AuthType.BASIC:
            if not self.config.auth_username or not self.config.auth_password:
                raise AdapterValidationError(
                    "Basic auth requires auth_username and auth_password",
                    adapter_name=self.config.name,
                    adapter_type=self.config.adapter_type,
                )
            auth = (self.config.auth_username, self.config.auth_password)
        
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            auth=auth,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )
    
    def _execute_impl(self, **inputs: Any) -> Any:
        """Execute HTTP request."""
        if not self._client:
            raise AdapterExecutionError(
                "HTTP client not initialized",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        try:
            # Build endpoint with path parameters
            endpoint = self.config.endpoint
            path_params = {}
            for key, value in inputs.items():
                if f"{{{key}}}" in endpoint:
                    path_params[key] = value
                    endpoint = endpoint.replace(f"{{{key}}}", str(value))
            
            # Remaining inputs are body/query params
            body_params = {k: v for k, v in inputs.items() if k not in path_params}
            
            # Make request based on method
            if self.config.method in [HttpMethod.GET, HttpMethod.DELETE]:
                response = self._client.request(
                    self.config.method.value,
                    endpoint,
                    params=body_params,
                )
            else:
                if self.config.request_format == "json":
                    response = self._client.request(
                        self.config.method.value,
                        endpoint,
                        json=body_params,
                    )
                elif self.config.request_format == "form":
                    response = self._client.request(
                        self.config.method.value,
                        endpoint,
                        data=body_params,
                    )
                else:
                    response = self._client.request(
                        self.config.method.value,
                        endpoint,
                        content=body_params.get("body", b""),
                    )
            
            # Check status code
            response.raise_for_status()
            
            # Parse response
            if self.config.response_format == "json":
                return response.json()
            else:
                return response.text
        
        except httpx.TimeoutException as e:
            raise AdapterTimeoutError(
                f"HTTP request timed out after {self.config.timeout}s",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        except httpx.HTTPStatusError as e:
            raise AdapterExecutionError(
                f"HTTP {e.response.status_code}: {e.response.text}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
                context={"status_code": e.response.status_code},
            )
        
        except Exception as e:
            raise AdapterExecutionError(
                f"HTTP request failed: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def __del__(self):
        """Cleanup HTTP client."""
        if self._client:
            self._client.close()
