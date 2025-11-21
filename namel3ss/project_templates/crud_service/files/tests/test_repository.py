"""
Integration tests for repository layer.

These tests require a PostgreSQL database and test actual database operations.
"""

from uuid import uuid4

import pytest

from models.domain import {{ entity_name }}
from repository import Postgres{{ entity_name }}Repository


@pytest.mark.integration
class TestRepository:
    """Test repository database operations."""
    
    async def test_create_item(
        self,
        repository: Postgres{{ entity_name }}Repository,
        sample_item: {{ entity_name }},
        clean_database,
    ):
        """Test creating an item in database."""
        created = await repository.create(sample_item)
        
        assert created.id == sample_item.id
        assert created.name == sample_item.name
        assert created.quantity == sample_item.quantity
        assert created.created_at is not None
    
    async def test_create_duplicate_id_fails(
        self,
        repository: Postgres{{ entity_name }}Repository,
        created_item: {{ entity_name }},
    ):
        """Test that creating item with duplicate ID fails."""
        duplicate = {{ entity_name }}(
            id=created_item.id,
            name="Different Name",
        )
        
        with pytest.raises(ValueError, match="already exists"):
            await repository.create(duplicate)
    
    async def test_get_by_id(
        self,
        repository: Postgres{{ entity_name }}Repository,
        created_item: {{ entity_name }},
    ):
        """Test retrieving item by ID."""
        retrieved = await repository.get_by_id(created_item.id)
        
        assert retrieved is not None
        assert retrieved.id == created_item.id
        assert retrieved.name == created_item.name
    
    async def test_get_by_id_not_found(
        self,
        repository: Postgres{{ entity_name }}Repository,
        clean_database,
    ):
        """Test retrieving non-existent item returns None."""
        result = await repository.get_by_id(uuid4())
        
        assert result is None
    
    async def test_get_by_id_excludes_deleted(
        self,
        repository: Postgres{{ entity_name }}Repository,
        created_item: {{ entity_name }},
    ):
        """Test that deleted items are excluded by default."""
        # Soft delete the item
        await repository.delete(created_item.id, soft=True)
        
        # Should not find it
        result = await repository.get_by_id(created_item.id)
        assert result is None
        
        # Should find it with include_deleted
        result = await repository.get_by_id(created_item.id, include_deleted=True)
        assert result is not None
    
    async def test_list_items(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test listing items with pagination."""
        items, total = await repository.list_items(page=1, page_size=3)
        
        assert len(items) == 3
        assert total == 5
    
    async def test_list_items_pagination(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test pagination works correctly."""
        # First page
        page1, total = await repository.list_items(page=1, page_size=2)
        assert len(page1) == 2
        assert total == 5
        
        # Second page
        page2, total = await repository.list_items(page=2, page_size=2)
        assert len(page2) == 2
        
        # No overlap
        page1_ids = {item.id for item in page1}
        page2_ids = {item.id for item in page2}
        assert page1_ids.isdisjoint(page2_ids)
    
    async def test_list_items_filter_by_active(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test filtering by active status."""
        active_items, total = await repository.list_items(is_active=True)
        
        assert all(item.is_active for item in active_items)
        assert total == 3  # Items 0, 2, 4 are active
    
    async def test_list_items_filter_by_tags(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test filtering by tags."""
        items, total = await repository.list_items(tags=["common"])
        
        assert total == 5  # All have "common" tag
        
        items, total = await repository.list_items(tags=["tag0"])
        assert total == 1  # Only first item
    
    async def test_update_item(
        self,
        repository: Postgres{{ entity_name }}Repository,
        created_item: {{ entity_name }},
    ):
        """Test updating an item."""
        created_item.name = "Updated Name"
        created_item.quantity = 999
        
        updated = await repository.update(created_item)
        
        assert updated.name == "Updated Name"
        assert updated.quantity == 999
        assert updated.updated_at > created_item.created_at
    
    async def test_update_nonexistent_fails(
        self,
        repository: Postgres{{ entity_name }}Repository,
        sample_item: {{ entity_name }},
        clean_database,
    ):
        """Test updating non-existent item fails."""
        with pytest.raises(ValueError, match="not found"):
            await repository.update(sample_item)
    
    async def test_soft_delete(
        self,
        repository: Postgres{{ entity_name }}Repository,
        created_item: {{ entity_name }},
    ):
        """Test soft delete sets deleted_at."""
        result = await repository.delete(created_item.id, soft=True)
        
        assert result is True
        
        # Item should not be found normally
        item = await repository.get_by_id(created_item.id)
        assert item is None
        
        # But should be found with include_deleted
        item = await repository.get_by_id(created_item.id, include_deleted=True)
        assert item is not None
        assert item.deleted_at is not None
    
    async def test_hard_delete(
        self,
        repository: Postgres{{ entity_name }}Repository,
        created_item: {{ entity_name }},
    ):
        """Test hard delete removes item permanently."""
        result = await repository.delete(created_item.id, soft=False)
        
        assert result is True
        
        # Item should not exist even with include_deleted
        item = await repository.get_by_id(created_item.id, include_deleted=True)
        assert item is None
    
    async def test_delete_nonexistent_returns_false(
        self,
        repository: Postgres{{ entity_name }}Repository,
        clean_database,
    ):
        """Test deleting non-existent item returns False."""
        result = await repository.delete(uuid4(), soft=True)
        
        assert result is False
    
    async def test_restore(
        self,
        repository: Postgres{{ entity_name }}Repository,
        created_item: {{ entity_name }},
    ):
        """Test restoring soft-deleted item."""
        # Soft delete
        await repository.delete(created_item.id, soft=True)
        
        # Restore
        result = await repository.restore(created_item.id)
        assert result is True
        
        # Should now be found normally
        item = await repository.get_by_id(created_item.id)
        assert item is not None
        assert item.deleted_at is None
    
    async def test_restore_not_deleted_returns_false(
        self,
        repository: Postgres{{ entity_name }}Repository,
        created_item: {{ entity_name }},
    ):
        """Test restoring non-deleted item returns False."""
        result = await repository.restore(created_item.id)
        
        assert result is False
    
    async def test_count(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test counting items."""
        count = await repository.count()
        
        assert count == 5
    
    async def test_count_with_filters(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test counting with filters."""
        # Count active only
        count = await repository.count(is_active=True)
        assert count == 3
        
        # Count inactive only
        count = await repository.count(is_active=False)
        assert count == 2
    
    async def test_search_by_name(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test searching items by name."""
        items, total = await repository.search_by_name("Item 2")
        
        assert total == 1
        assert items[0].name == "Item 2"
    
    async def test_search_by_name_case_insensitive(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test search is case-insensitive."""
        items, total = await repository.search_by_name("ITEM")
        
        assert total == 5  # All items match
    
    async def test_search_by_name_partial_match(
        self,
        repository: Postgres{{ entity_name }}Repository,
        multiple_items: list[{{ entity_name }}],
    ):
        """Test search supports partial matching."""
        items, total = await repository.search_by_name("Item")
        
        assert total == 5  # All items contain "Item"
    
    async def test_tenant_isolation(
        self,
        repository: Postgres{{ entity_name }}Repository,
        clean_database,
        tenant_id: str,
    ):
        """Test multi-tenancy isolation."""
        # Create items for different tenants
        item1 = {{ entity_name }}(name="Tenant 1 Item", tenant_id="tenant1")
        item2 = {{ entity_name }}(name="Tenant 2 Item", tenant_id="tenant2")
        
        await repository.create(item1)
        await repository.create(item2)
        
        # Query with tenant filter
        items, total = await repository.list_items(tenant_id="tenant1")
        assert total == 1
        assert items[0].tenant_id == "tenant1"
        
        items, total = await repository.list_items(tenant_id="tenant2")
        assert total == 1
        assert items[0].tenant_id == "tenant2"
