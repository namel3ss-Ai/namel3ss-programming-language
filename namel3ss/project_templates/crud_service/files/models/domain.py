"""
Domain model for {{ entity_name }}.

This represents the core business entity. All database operations
and business logic should work with this model.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


class {{ entity_name }}:
    """
    {{ entity_name }} domain entity.
    
    This is the core domain model. It's separate from database models
    and API schemas to maintain clean architecture boundaries.
    
    To extend this entity:
    1. Add new fields here
    2. Update the database schema (migrations.sql)
    3. Update Pydantic schemas (schemas.py)
    4. Update repository queries if needed
    """
    
    def __init__(
        self,
        id: Optional[UUID] = None,
        name: str = "",
        description: Optional[str] = None,
        quantity: int = 0,
        price: float = 0.0,
        is_active: bool = True,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        deleted_at: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Initialize {{ entity_name }}.
        
        Args:
            id: Unique identifier (auto-generated if None)
            name: {{ entity_name }} name (required)
            description: Optional description
            quantity: Available quantity (default: 0)
            price: Unit price (default: 0.0)
            is_active: Whether item is active (default: True)
            tags: Optional list of tags for categorization
            metadata: Optional key-value metadata
            created_at: Creation timestamp (auto-set if None)
            updated_at: Last update timestamp
            deleted_at: Soft delete timestamp (None if not deleted)
            tenant_id: Tenant identifier for multi-tenancy (optional)
        """
        self.id = id or uuid4()
        self.name = name
        self.description = description
        self.quantity = quantity
        self.price = price
        self.is_active = is_active
        self.tags = tags or []
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.deleted_at = deleted_at
        self.tenant_id = tenant_id
    
    @property
    def is_deleted(self) -> bool:
        """Check if item is soft-deleted."""
        return self.deleted_at is not None
    
    def mark_deleted(self) -> None:
        """Mark item as deleted (soft delete)."""
        self.deleted_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore soft-deleted item."""
        self.deleted_at = None
        self.updated_at = datetime.utcnow()
    
    def update_fields(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        is_active: Optional[bool] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Update item fields.
        
        Only provided fields are updated. None values are ignored.
        Always updates updated_at timestamp.
        """
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if quantity is not None:
            self.quantity = quantity
        if price is not None:
            self.price = price
        if is_active is not None:
            self.is_active = is_active
        if tags is not None:
            self.tags = tags
        if metadata is not None:
            self.metadata = metadata
        
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "quantity": self.quantity,
            "price": self.price,
            "is_active": self.is_active,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "tenant_id": self.tenant_id,
        }
    
    def __repr__(self) -> str:
        return f"{{ entity_name }}(id={self.id}, name='{self.name}', quantity={self.quantity}, price={self.price})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, {{ entity_name }}):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)
