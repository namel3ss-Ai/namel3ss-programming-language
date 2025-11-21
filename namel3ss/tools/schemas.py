"""
Pydantic schema models for tool input/output validation.

This module defines base classes and utilities for creating strongly-typed,
validated tool schemas using Pydantic v2.

Key Classes:
    - ToolInputModel: Base class for tool inputs
    - ToolOutputModel: Base class for tool outputs
    - ToolChunkModel: Base class for streaming outputs

Design Principles:
    - Use Pydantic v2 for schema definition and validation
    - All models are JSON-serializable
    - Support for JSON Schema export (for codegen)
    - Rich validation with custom validators
    - Nested model support
    - Default values and optional fields

Example:
    from namel3ss.tools.schemas import ToolInputModel, ToolOutputModel
    from pydantic import Field
    
    class WeatherInput(ToolInputModel):
        '''Input for weather API tool.'''
        location: str = Field(..., description="City name or coordinates")
        units: str = Field("metric", description="Temperature units")
        include_forecast: bool = Field(False, description="Include 7-day forecast")
    
    class WeatherOutput(ToolOutputModel):
        '''Output from weather API tool.'''
        temperature: float = Field(..., description="Current temperature")
        condition: str = Field(..., description="Weather condition")
        humidity: int = Field(..., ge=0, le=100, description="Humidity percentage")
        forecast: Optional[List[dict]] = None

Thread Safety:
    Model classes are thread-safe.
    Model instances should be treated as immutable after validation.

Performance:
    - Validation is optimized by Pydantic's Rust core (v2)
    - Schema export is cached automatically
    - Use model_validate() for dict input (faster than __init__)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolInputModel(BaseModel):
    """
    Base class for tool input schemas.
    
    All tool inputs should extend this class to gain:
    - Automatic validation
    - JSON Schema generation
    - Serialization/deserialization
    - Type checking
    - Documentation generation
    
    Features:
        - Pydantic v2 validation
        - Extra fields forbidden by default
        - Alias support for backward compatibility
        - Custom validators via field_validator
    
    Configuration:
        - extra="forbid": Reject unknown fields (strict validation)
        - validate_assignment=True: Validate on field assignment
        - use_enum_values=True: Use enum values instead of enum objects
        - populate_by_name=True: Allow aliased field names
    
    Example:
        >>> from pydantic import Field
        >>> 
        >>> class SearchInput(ToolInputModel):
        ...     query: str = Field(..., description="Search query")
        ...     limit: int = Field(10, ge=1, le=100, description="Max results")
        ...     filters: Optional[Dict[str, Any]] = None
        >>> 
        >>> # Validation
        >>> input = SearchInput(query="hello", limit=50)
        >>> print(input.query)  # "hello"
        >>> 
        >>> # Invalid input raises ValidationError
        >>> SearchInput(query="hello", limit=200)  # Raises: limit <= 100
    
    JSON Schema Export:
        >>> schema = SearchInput.model_json_schema()
        >>> print(schema["properties"]["query"])
        {'type': 'string', 'description': 'Search query'}
    
    Best Practices:
        - Use Field(...) for required fields
        - Add descriptions to all fields (used in prompts/docs)
        - Use validators (ge, le, min_length, etc.) for constraints
        - Provide sensible defaults for optional fields
        - Use nested models for complex structures
        - Add examples via Field(examples=[...])
    """
    
    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True,  # Validate on assignment
        use_enum_values=True,  # Use enum values
        populate_by_name=True,  # Allow aliases
        str_strip_whitespace=True,  # Strip whitespace from strings
        json_schema_extra={
            "type": "tool_input",
        },
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary (alias for model_dump).
        
        Returns:
            Dictionary representation
        """
        return self.model_dump()
    
    def to_json(self) -> str:
        """
        Convert to JSON string.
        
        Returns:
            JSON string representation
        """
        return self.model_dump_json()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolInputModel:
        """
        Create instance from dictionary.
        
        Args:
            data: Dictionary with field values
        
        Returns:
            Validated model instance
        
        Raises:
            ValidationError: If validation fails
        """
        return cls.model_validate(data)
    
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """
        Get JSON Schema for this model.
        
        Returns:
            JSON Schema dictionary
        """
        return cls.model_json_schema()


class ToolOutputModel(BaseModel):
    """
    Base class for tool output schemas.
    
    All tool outputs should extend this class to gain:
    - Automatic validation
    - JSON Schema generation
    - Serialization/deserialization
    - Type checking
    
    Features:
        - Pydantic v2 validation
        - Extra fields allowed (for extensibility)
        - Timestamp tracking
        - Metadata support
    
    Configuration:
        - extra="allow": Allow additional fields (extensibility)
        - validate_assignment=True: Validate on assignment
    
    Example:
        >>> from pydantic import Field
        >>> 
        >>> class AnalysisOutput(ToolOutputModel):
        ...     sentiment: str = Field(..., description="Sentiment label")
        ...     confidence: float = Field(..., ge=0.0, le=1.0)
        ...     entities: List[str] = Field(default_factory=list)
        >>> 
        >>> output = AnalysisOutput(
        ...     sentiment="positive",
        ...     confidence=0.95,
        ...     entities=["Apple", "iPhone"]
        ... )
    
    Metadata:
        Outputs can include execution metadata:
        >>> class MyOutput(ToolOutputModel):
        ...     result: str
        ...     metadata: Optional[Dict[str, Any]] = Field(
        ...         None,
        ...         description="Execution metadata"
        ...     )
    
    Best Practices:
        - Include metadata field for execution info
        - Use Optional for fields that may not always be present
        - Provide clear field descriptions
        - Use nested models for complex structures
        - Include timestamps for debugging
    """
    
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "type": "tool_output",
        },
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolOutputModel:
        """Create instance from dictionary."""
        return cls.model_validate(data)
    
    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Get JSON Schema for this model."""
        return cls.model_json_schema()


class ToolChunkModel(BaseModel):
    """
    Base class for streaming output chunks.
    
    Used by streaming tools to return incremental results.
    Each chunk represents a partial result that can be:
    - Aggregated with other chunks
    - Displayed incrementally
    - Processed independently
    
    Features:
        - Lightweight (minimal overhead)
        - Sequence number tracking
        - Completion indication
        - Metadata support
    
    Example:
        >>> from pydantic import Field
        >>> 
        >>> class LLMChunk(ToolChunkModel):
        ...     token: str = Field(..., description="Generated token")
        ...     finish_reason: Optional[str] = None
        >>> 
        >>> # Stream chunks
        >>> async for chunk in llm_tool.invoke_stream(input, context):
        ...     print(chunk.token, end="", flush=True)
        ...     if chunk.is_final:
        ...         print(f"\nFinish reason: {chunk.finish_reason}")
    
    Sequence Numbers:
        >>> chunks = []
        >>> async for i, chunk in enumerate(stream):
        ...     chunk.sequence = i
        ...     chunks.append(chunk)
    
    Best Practices:
        - Keep chunks small (minimize latency)
        - Set is_final=True on last chunk
        - Include finish_reason for completion status
        - Add sequence numbers for ordering
        - Include metadata for debugging
    """
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )
    
    sequence: Optional[int] = Field(
        None,
        description="Sequence number in stream",
    )
    is_final: bool = Field(
        False,
        description="Whether this is the final chunk",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Chunk generation timestamp",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional chunk metadata",
    )


class ToolErrorModel(BaseModel):
    """
    Schema for tool error responses.
    
    Provides structured error information that can be:
    - Serialized to JSON
    - Logged for debugging
    - Returned to clients
    - Displayed in UIs
    
    Attributes:
        code: Machine-readable error code
        message: Human-readable error message
        tool_name: Name of the tool that raised the error
        field: Field that caused validation error (if applicable)
        details: Additional error details
        timestamp: Error occurrence timestamp
    
    Example:
        >>> error = ToolErrorModel(
        ...     code="TOOL003",
        ...     message="Invalid input: query is required",
        ...     tool_name="search",
        ...     field="query",
        ...     details={"expected": "string", "got": "null"}
        ... )
        >>> print(error.to_json())
    
    Usage in Error Handling:
        >>> try:
        ...     result = await tool.invoke(input, context)
        ... except ToolValidationError as e:
        ...     error_model = ToolErrorModel(
        ...         code=e.code,
        ...         message=e.message,
        ...         tool_name=e.tool_name,
        ...         field=e.field
        ...     )
        ...     return error_model
    """
    
    code: str = Field(..., description="Error code (e.g., TOOL003)")
    message: str = Field(..., description="Human-readable error message")
    tool_name: Optional[str] = Field(None, description="Tool that raised error")
    field: Optional[str] = Field(None, description="Field that caused error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp",
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()


# Common reusable input/output models

class SimpleTextInput(ToolInputModel):
    """Simple text input (single string)."""
    text: str = Field(..., description="Input text")


class SimpleTextOutput(ToolOutputModel):
    """Simple text output (single string)."""
    result: str = Field(..., description="Output text")


class KeyValueInput(ToolInputModel):
    """Key-value pair input."""
    key: str = Field(..., description="Key")
    value: Any = Field(..., description="Value")


class KeyValueOutput(ToolOutputModel):
    """Key-value pair output."""
    key: str = Field(..., description="Key")
    value: Any = Field(..., description="Value")


class ListInput(ToolInputModel):
    """List of items input."""
    items: List[Any] = Field(..., description="List of items")


class ListOutput(ToolOutputModel):
    """List of items output."""
    items: List[Any] = Field(..., description="List of items")


class JSONInput(ToolInputModel):
    """Generic JSON input."""
    data: Dict[str, Any] = Field(..., description="JSON data")


class JSONOutput(ToolOutputModel):
    """Generic JSON output."""
    data: Dict[str, Any] = Field(..., description="JSON data")


# Validation helpers

def merge_schemas(
    *schemas: type[BaseModel],
    name: str = "MergedSchema",
) -> type[BaseModel]:
    """
    Merge multiple Pydantic models into a single schema.
    
    Useful for combining inputs from multiple tools or
    creating composite schemas.
    
    Args:
        *schemas: Model classes to merge
        name: Name for the merged model
    
    Returns:
        New model class with fields from all input models
    
    Example:
        >>> class Schema1(ToolInputModel):
        ...     field1: str
        >>> 
        >>> class Schema2(ToolInputModel):
        ...     field2: int
        >>> 
        >>> MergedSchema = merge_schemas(Schema1, Schema2, name="Combined")
        >>> merged = MergedSchema(field1="hello", field2=42)
    
    Note:
        Field name conflicts are resolved by last-wins.
    """
    from pydantic import create_model
    
    # Collect all fields
    fields = {}
    for schema in schemas:
        for field_name, field_info in schema.model_fields.items():
            fields[field_name] = (field_info.annotation, field_info)
    
    # Create new model
    return create_model(name, **fields, __base__=ToolInputModel)


def extend_schema(
    base_schema: type[BaseModel],
    **additional_fields: Any,
) -> type[BaseModel]:
    """
    Extend a schema with additional fields.
    
    Args:
        base_schema: Base model class
        **additional_fields: Additional fields to add
    
    Returns:
        New model class extending base schema
    
    Example:
        >>> class BaseInput(ToolInputModel):
        ...     query: str
        >>> 
        >>> ExtendedInput = extend_schema(
        ...     BaseInput,
        ...     limit=(int, Field(10, ge=1)),
        ...     offset=(int, Field(0, ge=0))
        ... )
        >>> extended = ExtendedInput(query="test", limit=20)
    """
    from pydantic import create_model
    
    return create_model(
        f"Extended{base_schema.__name__}",
        __base__=base_schema,
        **additional_fields,
    )


def schema_to_json_schema(model: type[BaseModel]) -> Dict[str, Any]:
    """
    Convert Pydantic model to JSON Schema.
    
    Args:
        model: Pydantic model class
    
    Returns:
        JSON Schema dictionary
    
    Example:
        >>> class MyInput(ToolInputModel):
        ...     field: str
        >>> 
        >>> schema = schema_to_json_schema(MyInput)
        >>> print(schema["properties"]["field"])
    """
    return model.model_json_schema()


def validate_against_schema(
    data: Dict[str, Any],
    schema: type[BaseModel],
) -> BaseModel:
    """
    Validate data against a schema.
    
    Args:
        data: Data to validate
        schema: Schema to validate against
    
    Returns:
        Validated model instance
    
    Raises:
        ValidationError: If validation fails
    
    Example:
        >>> data = {"field": "value"}
        >>> validated = validate_against_schema(data, MyInput)
    """
    return schema.model_validate(data)
