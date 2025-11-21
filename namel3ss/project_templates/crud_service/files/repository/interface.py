"""
Repository interface for {{ entity_name }} persistence.

Defines the contract for data access operations.
Implementations can use different storage backends.
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from models.domain import {{ entity_name }}


class {{ entity_name }}Repository(ABC):
    """
    Abstract repository for {{ entity_name }} persistence.
    
    This interface defines all data access operations.
    Concrete implementations handle the actual storage mechanism.
    """
    
    @abstractmethod
    async def create(self, item: {{ entity_name }}) -> {{ entity_name }}:
        """
        Create a new item.
        
        Args:
            item: Item to create
            
        Returns:
            Created item with generated ID and timestamps
            
        Raises:
            ValueError: If item with same ID already exists
        """
        pass
    
    @abstractmethod
    async def get_by_id(
        self,
        item_id: UUID,
        include_deleted: bool = False,
        tenant_id: Optional[str] = None
    ) -> Optional[{{ entity_name }}]:
        """
        Get item by ID.
        
        Args:
            item_id: Item identifier
            include_deleted: Whether to include soft-deleted items
            tenant_id: Tenant filter (for multi-tenancy)
            
        Returns:
            Item if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def list_items(
        self,
        page: int = 1,
        page_size: int = 20,
        include_deleted: bool = False,
        is_active: Optional[bool] = None,
        tags: Optional[list[str]] = None,
        tenant_id: Optional[str] = None
    ) -> tuple[list[{{ entity_name }}], int]:
        """
        List items with pagination and filtering.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            include_deleted: Whether to include soft-deleted items
            is_active: Filter by active status (None = all)
            tags: Filter by tags (items must have ALL tags)
            tenant_id: Tenant filter (for multi-tenancy)
            
        Returns:
            Tuple of (items, total_count)
        """
        pass
    
    @abstractmethod
    async def update(self, item: {{ entity_name }}) -> {{ entity_name }}:
        """
        Update existing item.
        
        Args:
            item: Item with updated fields
            
        Returns:
            Updated item
            
        Raises:
            ValueError: If item not found
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        item_id: UUID,
        soft: bool = True,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Delete item.
        
        Args:
            item_id: Item identifier
            soft: If True, soft delete (set deleted_at). If False, hard delete.
            tenant_id: Tenant filter (for multi-tenancy)
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def restore(
        self,
        item_id: UUID,
        tenant_id: Optional[str] = None
    ) -> bool:
        """
        Restore soft-deleted item.
        
        Args:
            item_id: Item identifier
            tenant_id: Tenant filter (for multi-tenancy)
            
        Returns:
            True if restored, False if not found or not deleted
        """
        pass
    
    @abstractmethod
    async def count(
        self,
        include_deleted: bool = False,
        is_active: Optional[bool] = None,
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Count items matching criteria.
        
        Args:
            include_deleted: Whether to include soft-deleted items
            is_active: Filter by active status (None = all)
            tenant_id: Tenant filter (for multi-tenancy)
            
        Returns:
            Number of items
        """
        pass
    
    @abstractmethod
    async def search_by_name(
        self,
        query: str,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None
    ) -> tuple[list[{{ entity_name }}], int]:
        """
        Search items by name (case-insensitive partial match).
        
        Args:
            query: Search query
            page: Page number (1-indexed)
            page_size: Items per page
            tenant_id: Tenant filter (for multi-tenancy)
            
        Returns:
            Tuple of (items, total_count)
        """
        pass
