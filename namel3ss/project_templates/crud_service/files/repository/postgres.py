"""
PostgreSQL repository implementation for {{ entity_name }}.

Uses asyncpg for async PostgreSQL operations with connection pooling.
"""

from typing import Optional
from uuid import UUID
import asyncpg

from models.domain import {{ entity_name }}
from repository.interface import {{ entity_name }}Repository


class Postgres{{ entity_name }}Repository({{ entity_name }}Repository):
    """
    PostgreSQL implementation of {{ entity_name }} repository.
    
    Uses asyncpg connection pool for efficient database access.
    All queries use parameterized statements to prevent SQL injection.
    """
    
    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize repository with connection pool.
        
        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool
    
    async def create(self, item: {{ entity_name }}) -> {{ entity_name }}:
        """Create a new item in database."""
        query = """
            INSERT INTO {{ table_name }} (
                id, name, description, quantity, price, is_active,
                tags, metadata, created_at, updated_at, tenant_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
        """
        
        async with self.pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    query,
                    item.id,
                    item.name,
                    item.description,
                    item.quantity,
                    item.price,
                    item.is_active,
                    item.tags,
                    item.metadata,
                    item.created_at,
                    item.updated_at,
                    item.tenant_id,
                )
                return self._row_to_domain(row)
            except asyncpg.UniqueViolationError as e:
                raise ValueError(f"{{ entity_name }} with ID {item.id} already exists") from e
    
    async def get_by_id(
        self,
        item_id: UUID,
        include_deleted: bool = False,
        tenant_id: Optional[str] = None
    ) -> Optional[{{ entity_name }}]:
        """Get item by ID."""
        query = """
            SELECT * FROM {{ table_name }}
            WHERE id = $1
        """
        params = [item_id]
        
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        
        if tenant_id is not None:
            query += f" AND tenant_id = ${len(params) + 1}"
            params.append(tenant_id)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return self._row_to_domain(row) if row else None
    
    async def list_items(
        self,
        page: int = 1,
        page_size: int = 20,
        include_deleted: bool = False,
        is_active: Optional[bool] = None,
        tags: Optional[list[str]] = None,
        tenant_id: Optional[str] = None
    ) -> tuple[list[{{ entity_name }}], int]:
        """List items with pagination and filtering."""
        # Build WHERE clause
        where_clauses = []
        params = []
        param_idx = 1
        
        if not include_deleted:
            where_clauses.append("deleted_at IS NULL")
        
        if is_active is not None:
            where_clauses.append(f"is_active = ${param_idx}")
            params.append(is_active)
            param_idx += 1
        
        if tags:
            # PostgreSQL array contains operator
            where_clauses.append(f"tags @> ${param_idx}")
            params.append(tags)
            param_idx += 1
        
        if tenant_id is not None:
            where_clauses.append(f"tenant_id = ${param_idx}")
            params.append(tenant_id)
            param_idx += 1
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Count query
        count_query = f"SELECT COUNT(*) FROM {{ table_name }} {where_clause}"
        
        # Data query with pagination
        offset = (page - 1) * page_size
        data_query = f"""
            SELECT * FROM {{ table_name }}
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([page_size, offset])
        
        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])  # Exclude LIMIT/OFFSET params
            rows = await conn.fetch(data_query, *params)
            
            items = [self._row_to_domain(row) for row in rows]
            return items, total
    
    async def update(self, item: {{ entity_name }}) -> {{ entity_name }}:
        """Update existing item."""
        query = """
            UPDATE {{ table_name }}
            SET
                name = $2,
                description = $3,
                quantity = $4,
                price = $5,
                is_active = $6,
                tags = $7,
                metadata = $8,
                updated_at = $9
            WHERE id = $1 AND deleted_at IS NULL
            RETURNING *
        """
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                item.id,
                item.name,
                item.description,
                item.quantity,
                item.price,
                item.is_active,
                item.tags,
                item.metadata,
                item.updated_at,
            )
            
            if not row:
                raise ValueError(f"{{ entity_name }} with ID {item.id} not found or is deleted")
            
            return self._row_to_domain(row)
    
    async def delete(
        self,
        item_id: UUID,
        soft: bool = True,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Delete item (soft or hard)."""
        if soft:
            query = """
                UPDATE {{ table_name }}
                SET deleted_at = NOW(), updated_at = NOW()
                WHERE id = $1 AND deleted_at IS NULL
            """
        else:
            query = "DELETE FROM {{ table_name }} WHERE id = $1"
        
        params = [item_id]
        
        if tenant_id is not None:
            query += f" AND tenant_id = ${len(params) + 1}"
            params.append(tenant_id)
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            # Check if any rows were affected
            return result.split()[-1] != "0"
    
    async def restore(
        self,
        item_id: UUID,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Restore soft-deleted item."""
        query = """
            UPDATE {{ table_name }}
            SET deleted_at = NULL, updated_at = NOW()
            WHERE id = $1 AND deleted_at IS NOT NULL
        """
        params = [item_id]
        
        if tenant_id is not None:
            query += f" AND tenant_id = ${len(params) + 1}"
            params.append(tenant_id)
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            return result.split()[-1] != "0"
    
    async def count(
        self,
        include_deleted: bool = False,
        is_active: Optional[bool] = None,
        tenant_id: Optional[str] = None
    ) -> int:
        """Count items matching criteria."""
        where_clauses = []
        params = []
        param_idx = 1
        
        if not include_deleted:
            where_clauses.append("deleted_at IS NULL")
        
        if is_active is not None:
            where_clauses.append(f"is_active = ${param_idx}")
            params.append(is_active)
            param_idx += 1
        
        if tenant_id is not None:
            where_clauses.append(f"tenant_id = ${param_idx}")
            params.append(tenant_id)
            param_idx += 1
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        query = f"SELECT COUNT(*) FROM {{ table_name }} {where_clause}"
        
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *params)
    
    async def search_by_name(
        self,
        query: str,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None
    ) -> tuple[list[{{ entity_name }}], int]:
        """Search items by name (case-insensitive)."""
        where_clauses = ["deleted_at IS NULL", "name ILIKE $1"]
        params = [f"%{query}%"]
        param_idx = 2
        
        if tenant_id is not None:
            where_clauses.append(f"tenant_id = ${param_idx}")
            params.append(tenant_id)
            param_idx += 1
        
        where_clause = "WHERE " + " AND ".join(where_clauses)
        
        # Count query
        count_query = f"SELECT COUNT(*) FROM {{ table_name }} {where_clause}"
        
        # Data query
        offset = (page - 1) * page_size
        data_query = f"""
            SELECT * FROM {{ table_name }}
            {where_clause}
            ORDER BY name ASC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([page_size, offset])
        
        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(data_query, *params)
            
            items = [self._row_to_domain(row) for row in rows]
            return items, total
    
    def _row_to_domain(self, row: asyncpg.Record) -> {{ entity_name }}:
        """Convert database row to domain model."""
        return {{ entity_name }}(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            quantity=row["quantity"],
            price=row["price"],
            is_active=row["is_active"],
            tags=row["tags"] or [],
            metadata=row["metadata"] or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            deleted_at=row["deleted_at"],
            tenant_id=row["tenant_id"],
        )
