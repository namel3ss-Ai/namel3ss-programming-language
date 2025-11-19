"""PostgreSQL + pgvector backend implementation."""

from __future__ import annotations

import os
import logging
import json
from typing import List, Optional, Dict, Any

from .base import VectorIndexBackend, ScoredDocument, register_vector_backend

logger = logging.getLogger(__name__)


class PgVectorBackend(VectorIndexBackend):
    """
    PostgreSQL with pgvector extension backend.
    
    Configuration:
        dsn: Database connection string (or use env var NAMEL3SS_PG_DSN)
        table_name: Table name for vectors (default: "embeddings")
        dimension: Vector dimension
        create_table: Whether to auto-create table (default: True)
        pool_size: Connection pool size (default: 5)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dsn = config.get("dsn") or os.getenv("NAMEL3SS_PG_DSN")
        if not self.dsn:
            raise ValueError(
                "PostgreSQL DSN not configured. "
                "Set 'dsn' in config or NAMEL3SS_PG_DSN environment variable."
            )
        
        self.table_name = config.get("table_name", "embeddings")
        self.dimension = config.get("dimension", 1536)
        self.create_table = config.get("create_table", True)
        self.pool_size = config.get("pool_size", 5)
        
        self._pool = None
    
    async def _get_pool(self):
        """Lazy initialize connection pool."""
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self.dsn,
                    min_size=1,
                    max_size=self.pool_size,
                )
            except ImportError:
                raise ImportError(
                    "asyncpg not installed. Run: pip install asyncpg"
                )
        return self._pool
    
    async def initialize(self):
        """Initialize pgvector extension and create table."""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            if self.create_table:
                # Create table with vector column
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        embedding vector({self.dimension}),
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for vector similarity search
                # Using ivfflat index for better performance on large datasets
                try:
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                        ON {self.table_name} 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                except Exception as e:
                    # Fallback to basic index if ivfflat fails
                    logger.warning(f"Failed to create ivfflat index: {e}")
                    await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                        ON {self.table_name} 
                        USING ivfflat (embedding vector_l2_ops);
                    """)
                
                # Create GIN index for metadata filtering
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_idx 
                    ON {self.table_name} 
                    USING GIN (metadata);
                """)
                
                logger.info(f"Initialized pgvector table: {self.table_name}")
    
    async def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Insert or update vectors."""
        if not (len(ids) == len(embeddings) == len(contents) == len(metadatas)):
            raise ValueError("All input lists must have the same length")
        
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            # Use INSERT ... ON CONFLICT for upsert
            for doc_id, embedding, content, metadata in zip(ids, embeddings, contents, metadatas):
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, embedding, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) 
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        created_at = CURRENT_TIMESTAMP;
                    """,
                    doc_id,
                    embedding,
                    content,
                    json.dumps(metadata),
                )
    
    async def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",
    ) -> List[ScoredDocument]:
        """Query for similar vectors."""
        pool = await self._get_pool()
        
        # Choose distance operator based on metric
        if distance_metric == "cosine":
            distance_op = "<=>"
            order = "ASC"
        elif distance_metric == "euclidean" or distance_metric == "l2":
            distance_op = "<->"
            order = "ASC"
        elif distance_metric == "dot" or distance_metric == "inner":
            distance_op = "<#>"
            order = "DESC"
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        # Build query
        query_sql = f"""
            SELECT id, content, metadata, embedding {distance_op} $1 as distance
            FROM {self.table_name}
        """
        
        params = [query_vector]
        
        # Add metadata filters if provided
        if filters:
            filter_conditions = []
            param_idx = 2
            for key, value in filters.items():
                filter_conditions.append(f"metadata @> ${param_idx}::jsonb")
                params.append(json.dumps({key: value}))
                param_idx += 1
            
            if filter_conditions:
                query_sql += " WHERE " + " AND ".join(filter_conditions)
        
        query_sql += f" ORDER BY distance {order} LIMIT {top_k};"
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)
        
        results = []
        for row in rows:
            results.append(ScoredDocument(
                id=row["id"],
                content=row["content"],
                score=float(row["distance"]),
                metadata=dict(row["metadata"]) if row["metadata"] else {},
                embedding=None,  # Don't return embeddings by default
            ))
        
        return results
    
    async def delete(self, ids: List[str]):
        """Delete documents by ID."""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.table_name} WHERE id = ANY($1);",
                ids
            )
    
    async def count(self) -> int:
        """Count documents in the index."""
        pool = await self._get_pool()
        
        async with pool.acquire() as conn:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name};")
            return result
    
    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


# Register pgvector backend
register_vector_backend("pgvector", PgVectorBackend)
register_vector_backend("postgres", PgVectorBackend)  # Alias


__all__ = ["PgVectorBackend"]
