"""
Domain-specific errors for AI subsystem validation and operations.

This module defines AI-specific exceptions that provide detailed context
about what went wrong during AI construct validation, configuration, or execution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from namel3ss.errors import N3Error


class AIValidationError(N3Error):
    """
    Raised when AI construct validation fails.
    
    This error indicates that an AI construct (Prompt, Chain, Model, etc.)
    has invalid configuration, missing required fields, or violates constraints.
    
    Args:
        message: Human-readable error message
        construct_type: Type of AI construct (e.g., "Prompt", "Chain")
        construct_name: Name of the specific construct that failed
        field: Specific field that caused the validation error
        value: The invalid value (will be sanitized in error message)
        **kwargs: Additional error context passed to N3Error
        
    Example:
        raise AIValidationError(
            "Model name cannot be empty",
            construct_type="AIModel",
            construct_name="gpt_model",
            field="model_name",
            code="AI001"
        )
    """
    
    def __init__(
        self,
        message: str,
        *,
        construct_type: Optional[str] = None,
        construct_name: Optional[str] = None,
        field: Optional[str] = None,
        value: Any = None,
        **kwargs
    ) -> None:
        super().__init__(message, **kwargs)
        self.construct_type = construct_type
        self.construct_name = construct_name
        self.field = field
        self.value = value
        
    def format(self) -> str:
        """Format error with AI-specific context."""
        parts = []
        
        if self.construct_type and self.construct_name:
            parts.append(f"{self.construct_type} '{self.construct_name}': {self.message}")
        elif self.construct_type:
            parts.append(f"{self.construct_type}: {self.message}")
        else:
            parts.append(self.message)
            
        if self.field:
            parts.append(f"(field: {self.field})")
            
        meta_parts = []
        location_desc = self.location.describe()
        if location_desc != "unknown location":
            meta_parts.append(location_desc)
        if self.code:
            meta_parts.append(self.code)
        if meta_parts:
            parts.append(f"({'; '.join(meta_parts)})")
            
        if self.hint:
            parts.append(f"Hint: {self.hint}")
            
        return " ".join(parts)


class AIConfigurationError(N3Error):
    """
    Raised when AI runtime configuration is invalid.
    
    This error indicates problems with provider configuration, credential issues,
    or runtime setup problems that prevent AI operations from executing.
    
    Args:
        message: Human-readable error message
        provider: Provider name (e.g., "openai", "anthropic")
        config_key: Specific configuration key that is invalid
        **kwargs: Additional error context passed to N3Error
        
    Example:
        raise AIConfigurationError(
            "API key not configured for provider",
            provider="openai",
            config_key="api_key",
            hint="Set OPENAI_API_KEY environment variable"
        )
    """
    
    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        config_key: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(message, **kwargs)
        self.provider = provider
        self.config_key = config_key


class AIExecutionError(N3Error):
    """
    Raised when AI operation execution fails at runtime.
    
    This error wraps runtime failures during:
    - Prompt execution
    - Chain/workflow execution
    - Tool invocation
    - Model inference
    
    Args:
        message: Human-readable error message
        operation: Type of operation that failed (e.g., "prompt_execution")
        operation_name: Name of the specific operation
        cause: Original exception that caused the failure
        **kwargs: Additional error context passed to N3Error
        
    Example:
        try:
            result = await provider.complete(prompt)
        except Exception as e:
            raise AIExecutionError(
                "Prompt execution failed",
                operation="prompt_execution",
                operation_name=prompt.name,
                cause=e,
                hint="Check provider API status"
            ) from e
    """
    
    def __init__(
        self,
        message: str,
        *,
        operation: Optional[str] = None,
        operation_name: Optional[str] = None,
        cause: Optional[Exception] = None,
        **kwargs
    ) -> None:
        super().__init__(message, **kwargs)
        self.operation = operation
        self.operation_name = operation_name
        self.cause = cause


__all__ = [
    "AIValidationError",
    "AIConfigurationError",
    "AIExecutionError",
]
