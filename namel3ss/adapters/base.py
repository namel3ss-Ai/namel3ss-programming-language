"""Tool Adapter Framework - Base interfaces and types.

Provides first-class adapters for extending N3 to interact with external systems:
- Python functions (FFI)
- HTTP APIs (REST/GraphQL)
- Databases (Postgres/MySQL via SQLAlchemy)
- Message queues (Celery/RQ/Kafka)
- ML models (OpenAI/Anthropic/HuggingFace/Custom)

All adapters support:
- Typed schemas (Pydantic validation)
- Retry + backoff policies
- OpenTelemetry tracing
- Version contracts
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AdapterType(str, Enum):
    """Supported adapter types."""
    PYTHON = "python"
    HTTP = "http"
    DATABASE = "db"
    QUEUE = "queue"
    MODEL = "model"


class RetryPolicy(BaseModel):
    """Retry configuration for adapter calls."""
    
    enabled: bool = Field(default=True, description="Enable retries")
    max_attempts: int = Field(default=3, ge=1, le=10, description="Max retry attempts")
    backoff_factor: float = Field(default=1.0, ge=0.0, description="Backoff multiplier")
    backoff_max: float = Field(default=60.0, ge=1.0, description="Max backoff seconds")
    retryable_exceptions: List[str] = Field(
        default_factory=lambda: [
            "TimeoutError",
            "ConnectionError",
            "httpx.TimeoutException",
            "httpx.ConnectError",
        ],
        description="Exception types that trigger retries"
    )


class AdapterConfig(BaseModel):
    """Base configuration for all adapters."""
    
    name: str = Field(..., description="Adapter identifier")
    adapter_type: AdapterType = Field(..., description="Adapter type")
    description: Optional[str] = Field(None, description="Human-readable description")
    
    # Retry configuration
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    
    # Timeout configuration
    timeout: float = Field(default=30.0, ge=0.1, description="Execution timeout")
    
    # Tracing configuration
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    
    # Schema validation
    validate_input: bool = Field(default=True, description="Validate inputs against schema")
    validate_output: bool = Field(default=True, description="Validate outputs against schema")
    
    # Version contract
    version: Optional[str] = Field(None, description="Adapter version for compatibility")
    
    class Config:
        use_enum_values = True


@dataclass
class AdapterResult:
    """Result from adapter execution."""
    
    output: Any
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tracing information
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    duration_ms: Optional[float] = None


class AdapterError(Exception):
    """Base exception for adapter errors."""
    
    def __init__(
        self,
        message: str,
        adapter_name: Optional[str] = None,
        adapter_type: Optional[AdapterType] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.adapter_name = adapter_name
        self.adapter_type = adapter_type
        self.context = context or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        parts = [self.message]
        if self.adapter_name:
            parts.append(f"adapter={self.adapter_name}")
        if self.adapter_type:
            parts.append(f"type={self.adapter_type.value}")
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"context=({ctx_str})")
        return " | ".join(parts)


class AdapterValidationError(AdapterError):
    """Input/output validation failed."""
    pass


class AdapterExecutionError(AdapterError):
    """Adapter execution failed."""
    pass


class AdapterTimeoutError(AdapterError):
    """Adapter execution timed out."""
    pass


class BaseAdapter(ABC):
    """Base class for all adapters.
    
    Adapters are typed, versioned, instrumented interfaces to external systems.
    They provide:
    - Schema validation for inputs/outputs
    - Automatic retries with backoff
    - OpenTelemetry tracing
    - Error context for debugging
    
    Example:
        class CustomAdapter(BaseAdapter):
            def _execute_impl(self, **inputs):
                result = external_service.call(**inputs)
                return result
        
        adapter = CustomAdapter(config)
        result = adapter.execute(param="value")
    """
    
    def __init__(self, config: AdapterConfig):
        """Initialize adapter with configuration."""
        self.config = config
        self._input_schema: Optional[type[BaseModel]] = None
        self._output_schema: Optional[type[BaseModel]] = None
    
    def set_input_schema(self, schema: type[BaseModel]) -> None:
        """Set Pydantic schema for input validation."""
        self._input_schema = schema
    
    def set_output_schema(self, schema: type[BaseModel]) -> None:
        """Set Pydantic schema for output validation."""
        self._output_schema = schema
    
    def execute(self, **inputs: Any) -> AdapterResult:
        """Execute adapter with inputs.
        
        Handles validation, retries, tracing, and error handling.
        
        Args:
            **inputs: Adapter inputs
        
        Returns:
            AdapterResult with output and metadata
        
        Raises:
            AdapterValidationError: Invalid inputs/outputs
            AdapterExecutionError: Execution failed
            AdapterTimeoutError: Execution timed out
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs
            if self.config.validate_input and self._input_schema:
                inputs = self._validate_inputs(inputs)
            
            # Execute with retry logic
            output = self._execute_with_retry(**inputs)
            
            # Validate outputs
            if self.config.validate_output and self._output_schema:
                output = self._validate_output(output)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return AdapterResult(
                output=output,
                success=True,
                metadata={
                    "adapter_name": self.config.name,
                    "adapter_type": self.config.adapter_type.value,
                    "duration_ms": duration_ms,
                },
                duration_ms=duration_ms,
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(e, (AdapterValidationError, AdapterExecutionError, AdapterTimeoutError)):
                raise
            
            raise AdapterExecutionError(
                f"Adapter execution failed: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
                context={"duration_ms": duration_ms},
            )
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inputs against schema."""
        if not self._input_schema:
            return inputs
        
        try:
            validated = self._input_schema(**inputs)
            return validated.model_dump()
        except Exception as e:
            raise AdapterValidationError(
                f"Input validation failed: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _validate_output(self, output: Any) -> Any:
        """Validate output against schema."""
        if not self._output_schema:
            return output
        
        try:
            if isinstance(output, dict):
                validated = self._output_schema(**output)
                return validated.model_dump()
            else:
                validated = self._output_schema(value=output)
                return validated.value
        except Exception as e:
            raise AdapterValidationError(
                f"Output validation failed: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _execute_with_retry(self, **inputs: Any) -> Any:
        """Execute with retry logic."""
        import time
        
        if not self.config.retry_policy.enabled:
            return self._execute_impl(**inputs)
        
        last_exception = None
        
        for attempt in range(self.config.retry_policy.max_attempts):
            try:
                return self._execute_impl(**inputs)
            
            except Exception as e:
                last_exception = e
                exception_type = type(e).__name__
                
                # Check if exception is retryable
                if exception_type not in self.config.retry_policy.retryable_exceptions:
                    raise
                
                # Don't retry on last attempt
                if attempt >= self.config.retry_policy.max_attempts - 1:
                    raise
                
                # Calculate backoff delay
                delay = self.config.retry_policy.backoff_factor * (2 ** attempt)
                delay = min(delay, self.config.retry_policy.backoff_max)
                
                time.sleep(delay)
        
        # Should not reach here, but handle it
        if last_exception:
            raise last_exception
        
        raise AdapterExecutionError(
            "Retry logic failed",
            adapter_name=self.config.name,
            adapter_type=self.config.adapter_type,
        )
    
    @abstractmethod
    def _execute_impl(self, **inputs: Any) -> Any:
        """Implementation-specific execution logic.
        
        Subclasses must implement this method to perform actual work.
        
        Args:
            **inputs: Validated inputs
        
        Returns:
            Execution result
        
        Raises:
            Any exception indicating failure
        """
        pass


class AdapterRegistry:
    """Global registry for adapter instances."""
    
    def __init__(self):
        self._adapters: Dict[str, BaseAdapter] = {}
    
    def register(self, name: str, adapter: BaseAdapter) -> None:
        """Register an adapter."""
        self._adapters[name] = adapter
    
    def get(self, name: str) -> Optional[BaseAdapter]:
        """Get adapter by name."""
        return self._adapters.get(name)
    
    def list_adapters(self) -> List[str]:
        """List registered adapter names."""
        return list(self._adapters.keys())


# Global registry instance
_adapter_registry = AdapterRegistry()


def register_adapter(name: str, adapter: BaseAdapter) -> None:
    """Register an adapter in the global registry."""
    _adapter_registry.register(name, adapter)


def get_adapter(name: str) -> Optional[BaseAdapter]:
    """Get adapter from global registry."""
    return _adapter_registry.get(name)


def list_adapters() -> List[str]:
    """List all registered adapters."""
    return _adapter_registry.list_adapters()
