"""
Unit tests for Pydantic schemas.

Tests request/response validation and serialization.
"""

import pytest
from pydantic import ValidationError

from models.schemas import (
    {{ entity_name }}Create,
    {{ entity_name }}Update,
    {{ entity_name }}Response,
    {{ entity_name }}List,
    ErrorResponse,
    ErrorDetail,
)


@pytest.mark.unit
class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_create_schema_valid(self):
        """Test valid create schema."""
        data = {
            "name": "Test Item",
            "description": "A test item",
            "quantity": 100,
            "price": 29.99,
            "is_active": True,
            "tags": ["electronics", "premium"],
            "metadata": {"color": "blue"},
        }
        
        schema = {{ entity_name }}Create(**data)
        
        assert schema.name == "Test Item"
        assert schema.quantity == 100
        assert schema.price == 29.99
        assert schema.tags == ["electronics", "premium"]
    
    def test_create_schema_with_defaults(self):
        """Test create schema with default values."""
        schema = {{ entity_name }}Create(name="Minimal")
        
        assert schema.name == "Minimal"
        assert schema.quantity == 0
        assert schema.price == 0.0
        assert schema.is_active is True
        assert schema.tags == []
        assert schema.metadata == {}
    
    def test_create_schema_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValidationError) as exc:
            {{ entity_name }}Create(quantity=10)
        
        errors = exc.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)
    
    def test_create_schema_name_min_length(self):
        """Test name minimum length validation."""
        with pytest.raises(ValidationError) as exc:
            {{ entity_name }}Create(name="")
        
        errors = exc.value.errors()
        assert any("name" in str(e["loc"]) for e in errors)
    
    def test_create_schema_negative_quantity(self):
        """Test negative quantity validation."""
        with pytest.raises(ValidationError) as exc:
            {{ entity_name }}Create(name="Test", quantity=-10)
        
        errors = exc.value.errors()
        assert any("quantity" in str(e["loc"]) for e in errors)
    
    def test_create_schema_negative_price(self):
        """Test negative price validation."""
        with pytest.raises(ValidationError) as exc:
            {{ entity_name }}Create(name="Test", price=-5.00)
        
        errors = exc.value.errors()
        assert any("price" in str(e["loc"]) for e in errors)
    
    def test_create_schema_price_precision(self):
        """Test price decimal precision validation."""
        with pytest.raises(ValidationError) as exc:
            {{ entity_name }}Create(name="Test", price=19.999)
        
        errors = exc.value.errors()
        assert any("price" in str(e["loc"]) for e in errors)
    
    def test_create_schema_tags_normalization(self):
        """Test tags normalization (lowercase, deduplicated)."""
        schema = {{ entity_name }}Create(
            name="Test",
            tags=["Electronics", "PREMIUM", "electronics", "  premium  "]
        )
        
        assert schema.tags == ["electronics", "premium"]
    
    def test_create_schema_tags_invalid_format(self):
        """Test tags with invalid characters."""
        with pytest.raises(ValidationError) as exc:
            {{ entity_name }}Create(name="Test", tags=["valid-tag", "invalid tag!"])
        
        errors = exc.value.errors()
        assert any("tag" in str(e["msg"]).lower() for e in errors)
    
    def test_update_schema_all_optional(self):
        """Test that all update fields are optional."""
        schema = {{ entity_name }}Update()
        
        # Should create successfully with no fields
        assert schema.model_dump(exclude_unset=True) == {}
    
    def test_update_schema_partial_update(self):
        """Test partial update with some fields."""
        schema = {{ entity_name }}Update(name="Updated", price=39.99)
        
        data = schema.model_dump(exclude_unset=True)
        assert data == {"name": "Updated", "price": 39.99}
    
    def test_response_schema_from_dict(self):
        """Test response schema from dictionary."""
        from datetime import datetime
        from uuid import uuid4
        
        data = {
            "id": uuid4(),
            "name": "Test",
            "description": None,
            "quantity": 50,
            "price": 19.99,
            "is_active": True,
            "tags": ["test"],
            "metadata": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "deleted_at": None,
            "tenant_id": None,
        }
        
        schema = {{ entity_name }}Response(**data)
        
        assert str(schema.id) == str(data["id"])
        assert schema.name == "Test"
    
    def test_list_schema(self):
        """Test list schema with pagination."""
        from datetime import datetime
        from uuid import uuid4
        
        items = [
            {{ entity_name }}Response(
                id=uuid4(),
                name=f"Item {i}",
                quantity=i * 10,
                price=float(i * 10),
                is_active=True,
                tags=[],
                metadata={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                deleted_at=None,
                tenant_id=None,
            )
            for i in range(3)
        ]
        
        list_schema = {{ entity_name }}List(
            items=items,
            total=100,
            page=2,
            page_size=20,
            has_next=True,
            has_prev=True,
        )
        
        assert len(list_schema.items) == 3
        assert list_schema.total == 100
        assert list_schema.page == 2
        assert list_schema.has_next is True
    
    def test_error_detail_schema(self):
        """Test error detail schema."""
        detail = ErrorDetail(
            field="name",
            message="Name is required",
            code="value_error.missing",
        )
        
        assert detail.field == "name"
        assert detail.message == "Name is required"
    
    def test_error_response_schema(self):
        """Test error response schema."""
        error = ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            details=[
                ErrorDetail(field="name", message="Required", code="missing")
            ],
            request_id="req_123",
        )
        
        assert error.error == "validation_error"
        assert len(error.details) == 1
        assert error.request_id == "req_123"
