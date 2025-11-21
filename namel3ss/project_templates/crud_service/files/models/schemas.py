"""
Pydantic schemas for API request/response validation.

These schemas provide strong typing and validation for the HTTP API layer.
They are separate from the domain model to allow API evolution without
affecting core business logic.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


class {{ entity_name }}Base(BaseModel):
    """Base schema with common {{ entity_name }} fields."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="{{ entity_name }} name",
        examples=["Widget A", "Premium Service"]
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional description",
        examples=["A high-quality widget for enterprise use"]
    )
    quantity: int = Field(
        default=0,
        ge=0,
        description="Available quantity",
        examples=[100, 0, 50]
    )
    price: float = Field(
        default=0.0,
        ge=0.0,
        description="Unit price",
        examples=[19.99, 0.0, 1499.99]
    )
    is_active: bool = Field(
        default=True,
        description="Whether item is active",
        examples=[True, False]
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Tags for categorization",
        examples=[["electronics", "hardware"], ["service", "premium"]]
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional key-value metadata",
        examples=[{"color": "blue", "weight": "1.5kg"}, {}]
    )
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate and normalize tags."""
        if not v:
            return []
        
        # Remove duplicates and empty strings
        tags = [tag.strip().lower() for tag in v if tag and tag.strip()]
        unique_tags = list(dict.fromkeys(tags))  # Preserve order while removing duplicates
        
        # Validate tag format
        for tag in unique_tags:
            if not tag.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"Tag '{tag}' contains invalid characters. Use alphanumeric, hyphens, and underscores only.")
            if len(tag) > 50:
                raise ValueError(f"Tag '{tag}' exceeds maximum length of 50 characters")
        
        return unique_tags
    
    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Validate price precision (2 decimal places)."""
        if round(v, 2) != v:
            raise ValueError("Price must have at most 2 decimal places")
        return v


class {{ entity_name }}Create({{ entity_name }}Base):
    """Schema for creating a new {{ entity_name }}."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Premium Widget",
                "description": "High-quality widget for enterprise customers",
                "quantity": 100,
                "price": 49.99,
                "is_active": True,
                "tags": ["electronics", "premium"],
                "metadata": {"color": "silver", "warranty": "2 years"}
            }
        }
    )


class {{ entity_name }}Update(BaseModel):
    """
    Schema for updating an existing {{ entity_name }}.
    
    All fields are optional. Only provided fields will be updated.
    """
    
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="{{ entity_name }} name"
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Description"
    )
    quantity: Optional[int] = Field(
        None,
        ge=0,
        description="Quantity"
    )
    price: Optional[float] = Field(
        None,
        ge=0.0,
        description="Price"
    )
    is_active: Optional[bool] = Field(
        None,
        description="Active status"
    )
    tags: Optional[list[str]] = Field(
        None,
        max_length=20,
        description="Tags"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Metadata"
    )
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate and normalize tags."""
        if v is None:
            return None
        return {{ entity_name }}Base.validate_tags(v)
    
    @field_validator("price")
    @classmethod
    def validate_price(cls, v: Optional[float]) -> Optional[float]:
        """Validate price precision."""
        if v is None:
            return None
        return {{ entity_name }}Base.validate_price(v)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Updated Widget Name",
                "price": 59.99,
                "is_active": False
            }
        }
    )


class {{ entity_name }}Response({{ entity_name }}Base):
    """Schema for {{ entity_name }} in API responses."""
    
    id: UUID = Field(..., description="Unique identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    deleted_at: Optional[datetime] = Field(None, description="Deletion timestamp (if soft-deleted)")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier (multi-tenancy)")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Premium Widget",
                "description": "High-quality widget",
                "quantity": 100,
                "price": 49.99,
                "is_active": True,
                "tags": ["electronics", "premium"],
                "metadata": {"color": "silver"},
                "created_at": "2025-11-21T10:00:00Z",
                "updated_at": "2025-11-21T10:00:00Z",
                "deleted_at": None,
                "tenant_id": None
            }
        }
    )


class {{ entity_name }}List(BaseModel):
    """Schema for paginated {{ entity_name }} list response."""
    
    items: list[{{ entity_name }}Response] = Field(..., description="List of items")
    total: int = Field(..., ge=0, description="Total number of items (before pagination)")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "Widget A",
                        "quantity": 50,
                        "price": 29.99,
                        "is_active": True,
                        "tags": ["electronics"],
                        "metadata": {},
                        "created_at": "2025-11-21T10:00:00Z",
                        "updated_at": "2025-11-21T10:00:00Z",
                        "deleted_at": None,
                        "tenant_id": None
                    }
                ],
                "total": 150,
                "page": 1,
                "page_size": 20,
                "has_next": True,
                "has_prev": False
            }
        }
    )


class ErrorDetail(BaseModel):
    """Schema for detailed error information."""
    
    field: Optional[str] = Field(None, description="Field name (for validation errors)")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[list[ErrorDetail]] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {
                        "field": "price",
                        "message": "Price must be greater than or equal to 0",
                        "code": "value_error.number.not_ge"
                    }
                ],
                "request_id": "req_123456789"
            }
        }
    )
