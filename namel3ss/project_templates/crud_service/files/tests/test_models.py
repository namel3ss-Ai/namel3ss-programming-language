"""
Unit tests for domain models.

Tests the core business entity without external dependencies.
"""

from datetime import datetime
from uuid import UUID

import pytest

from models.domain import {{ entity_name }}


@pytest.mark.unit
class TestDomainModel:
    """Test {{ entity_name }} domain model."""
    
    def test_create_item_with_defaults(self):
        """Test creating item with default values."""
        item = {{ entity_name }}(name="Test Item")
        
        assert isinstance(item.id, UUID)
        assert item.name == "Test Item"
        assert item.description is None
        assert item.quantity == 0
        assert item.price == 0.0
        assert item.is_active is True
        assert item.tags == []
        assert item.metadata == {}
        assert isinstance(item.created_at, datetime)
        assert isinstance(item.updated_at, datetime)
        assert item.deleted_at is None
        assert item.tenant_id is None
    
    def test_create_item_with_all_fields(self):
        """Test creating item with all fields specified."""
        item = {{ entity_name }}(
            name="Premium Widget",
            description="High-quality widget",
            quantity=100,
            price=49.99,
            is_active=True,
            tags=["premium", "electronics"],
            metadata={"color": "silver", "weight": "1.5kg"},
            tenant_id="tenant123",
        )
        
        assert item.name == "Premium Widget"
        assert item.description == "High-quality widget"
        assert item.quantity == 100
        assert item.price == 49.99
        assert item.is_active is True
        assert item.tags == ["premium", "electronics"]
        assert item.metadata == {"color": "silver", "weight": "1.5kg"}
        assert item.tenant_id == "tenant123"
    
    def test_is_deleted_property(self):
        """Test is_deleted property."""
        item = {{ entity_name }}(name="Test")
        
        assert item.is_deleted is False
        
        item.deleted_at = datetime.utcnow()
        assert item.is_deleted is True
    
    def test_mark_deleted(self):
        """Test soft delete functionality."""
        item = {{ entity_name }}(name="Test")
        original_updated = item.updated_at
        
        item.mark_deleted()
        
        assert item.deleted_at is not None
        assert item.is_deleted is True
        assert item.updated_at > original_updated
    
    def test_restore(self):
        """Test restoring soft-deleted item."""
        item = {{ entity_name }}(name="Test")
        item.mark_deleted()
        
        assert item.is_deleted is True
        
        item.restore()
        
        assert item.deleted_at is None
        assert item.is_deleted is False
    
    def test_update_fields(self):
        """Test updating item fields."""
        item = {{ entity_name }}(
            name="Original",
            quantity=10,
            price=19.99,
        )
        original_updated = item.updated_at
        
        item.update_fields(
            name="Updated",
            quantity=20,
        )
        
        assert item.name == "Updated"
        assert item.quantity == 20
        assert item.price == 19.99  # Unchanged
        assert item.updated_at > original_updated
    
    def test_update_fields_ignores_none(self):
        """Test that None values don't update fields."""
        item = {{ entity_name }}(name="Original", quantity=10)
        
        item.update_fields(name=None, quantity=20)
        
        assert item.name == "Original"  # Not updated
        assert item.quantity == 20  # Updated
    
    def test_to_dict(self):
        """Test converting item to dictionary."""
        item = {{ entity_name }}(
            name="Test",
            quantity=50,
            tags=["tag1"],
        )
        
        data = item.to_dict()
        
        assert isinstance(data, dict)
        assert data["name"] == "Test"
        assert data["quantity"] == 50
        assert data["tags"] == ["tag1"]
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_equality(self):
        """Test item equality based on ID."""
        item1 = {{ entity_name }}(name="Test")
        item2 = {{ entity_name }}(name="Test")
        item3 = item1
        
        assert item1 == item1
        assert item1 == item3
        assert item1 != item2  # Different IDs
        assert item1 != "not an item"
    
    def test_hash(self):
        """Test item hashing."""
        item = {{ entity_name }}(name="Test")
        
        # Should be hashable
        hash_value = hash(item)
        assert isinstance(hash_value, int)
        
        # Can be used in sets
        item_set = {item}
        assert item in item_set
    
    def test_repr(self):
        """Test string representation."""
        item = {{ entity_name }}(name="Test Widget", quantity=50, price=29.99)
        
        repr_str = repr(item)
        
        assert "{{ entity_name }}" in repr_str
        assert "Test Widget" in repr_str
        assert "50" in repr_str
        assert "29.99" in repr_str
