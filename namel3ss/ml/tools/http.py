"""HTTP tool implementation."""

from typing import Any, Dict, Optional

from namel3ss.ml.connectors.base import make_resilient_request, RetryConfig
from namel3ss.observability.logging import get_logger
from namel3ss.observability.metrics import record_metric

from .base import Tool, ToolResult, ToolError


logger = get_logger(__name__)


class HttpTool(Tool):
    """
    HTTP tool for making external API calls.
    
    Supports GET, POST, PUT, DELETE, PATCH with retry logic and validation.
    """
    
    def __init__(self, *, name: str, endpoint: str, method: str = "POST",
                 headers: Optional[Dict[str, str]] = None, timeout: float = 30.0,
                 input_schema: Optional[Dict] = None, output_schema: Optional[Dict] = None,
                 **config):
        """
        Initialize HTTP tool.
        
        Args:
            name: Tool identifier
            endpoint: HTTP endpoint URL
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            headers: HTTP headers to include
            timeout: Request timeout in seconds
            input_schema: JSON schema for inputs
            output_schema: JSON schema for outputs
            **config: Additional configuration
        """
        super().__init__(name=name, input_schema=input_schema, output_schema=output_schema, **config)
        
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        self.timeout = timeout
        
        # Configure retry logic
        self.retry_config = RetryConfig(
            max_retries=config.get("max_retries", 3),
            initial_delay=config.get("initial_delay", 1.0),
            max_delay=config.get("max_delay", 60.0),
            exponential_base=config.get("exponential_base", 2.0),
            retryable_status_codes=config.get(
                "retryable_status_codes",
                {429, 500, 502, 503, 504}
            )
        )
    
    def execute(self, **inputs) -> ToolResult:
        """
        Execute HTTP request with given inputs.
        
        Args:
            **inputs: Input data (sent as JSON body for POST/PUT/PATCH, query params for GET)
            
        Returns:
            ToolResult with parsed response
        """
        # Validate inputs
        try:
            self.validate_inputs(inputs)
        except ToolError as e:
            logger.error(f"Tool '{self.name}' input validation failed: {e}")
            return ToolResult(output=None, success=False, error=str(e))
        
        # Prepare request
        url = self.endpoint
        headers = self.headers.copy()
        json_data = None
        params = None
        
        if self.method in ("POST", "PUT", "PATCH"):
            json_data = inputs
            if "Content-Type" not in headers:
                headers["Content-Type"] = "application/json"
        elif self.method == "GET":
            params = inputs
        
        logger.info(f"Tool '{self.name}' executing: {self.method} {url}")
        
        try:
            response = make_resilient_request(
                url=url,
                method=self.method,
                headers=headers,
                json_data=json_data,
                params=params,
                retry_config=self.retry_config,
                timeout=self.timeout
            )
            
            # Check response status
            if response.status_code not in (200, 201, 202, 204):
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Tool '{self.name}' failed: {error_msg}")
                record_metric("tool.execution.error", 1, tags={"tool": self.name, "method": self.method})
                
                return ToolResult(
                    output=None,
                    success=False,
                    error=error_msg,
                    metadata={"status_code": response.status_code}
                )
            
            # Parse response
            try:
                output = response.json() if response.text else {}
            except ValueError:
                # Not JSON, return raw text
                output = response.text
            
            # Record success metrics
            record_metric("tool.execution.success", 1, tags={"tool": self.name, "method": self.method})
            record_metric("tool.execution.duration_ms", response.elapsed.total_seconds() * 1000,
                         tags={"tool": self.name, "method": self.method})
            
            logger.info(f"Tool '{self.name}' completed successfully")
            
            return ToolResult(
                output=output,
                success=True,
                metadata={
                    "status_code": response.status_code,
                    "duration_ms": response.elapsed.total_seconds() * 1000,
                }
            )
            
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(f"Tool '{self.name}' error: {error_msg}")
            record_metric("tool.execution.error", 1, tags={"tool": self.name, "method": self.method})
            
            return ToolResult(
                output=None,
                success=False,
                error=error_msg,
                metadata={"exception": type(e).__name__}
            )
    
    def __repr__(self) -> str:
        return f"HttpTool(name='{self.name}', method={self.method}, endpoint='{self.endpoint}')"
