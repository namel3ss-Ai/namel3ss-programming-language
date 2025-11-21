"""Database adapter for safe SQL queries via SQLAlchemy.

Provides typed database access with connection pooling, parameterization,
and transaction support.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .base import (
    AdapterConfig,
    AdapterType,
    BaseAdapter,
    AdapterExecutionError,
    AdapterTimeoutError,
    AdapterValidationError,
)

try:
    from sqlalchemy import create_engine, text, MetaData, Table
    from sqlalchemy.engine import Engine, Connection
    from sqlalchemy.pool import QueuePool
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


class DatabaseEngine(str, Enum):
    """Supported database engines."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class QueryType(str, Enum):
    """SQL query types."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    RAW = "raw"


class DatabaseAdapterConfig(AdapterConfig):
    """Configuration for database adapter."""
    
    adapter_type: AdapterType = Field(default=AdapterType.DATABASE)
    
    # Connection settings
    connection_url: str = Field(..., description="SQLAlchemy connection URL")
    engine_type: DatabaseEngine = Field(..., description="Database engine type")
    
    # Pool settings
    pool_size: int = Field(default=5, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, le=50, description="Max overflow connections")
    pool_timeout: float = Field(default=30.0, ge=1.0, description="Pool checkout timeout")
    pool_recycle: int = Field(default=3600, ge=0, description="Connection recycle time (seconds)")
    
    # Query settings
    allow_raw_sql: bool = Field(default=False, description="Allow raw SQL queries")
    max_results: int = Field(default=1000, ge=1, description="Max rows to return")
    
    # Transaction settings
    autocommit: bool = Field(default=True, description="Auto-commit queries")


class DatabaseAdapter(BaseAdapter):
    """Adapter for safe database queries via SQLAlchemy.
    
    Provides connection pooling, SQL injection prevention through
    parameterization, and transaction management.
    
    Features:
        - SQLAlchemy ORM integration
        - Connection pooling (configurable)
        - Parameterized queries (SQL injection prevention)
        - Transaction support
        - Multiple database support (Postgres, MySQL, SQLite)
        - Query result validation
        - Automatic type conversion
    
    Example:
        PostgreSQL query:
        ```n3
        tool "fetch_users" {
          adapter: "db"
          connection_url: env("DATABASE_URL")
          engine_type: "postgresql"
          pool_size: 10
          max_results: 100
        }
        
        chain "get_active_users" {
          call: "fetch_users"
          inputs: {
            query: "SELECT * FROM users WHERE active = :active"
            params: {active: true}
          }
        }
        ```
        
        Programmatic usage:
        >>> from namel3ss.adapters import DatabaseAdapter, DatabaseAdapterConfig
        >>> 
        >>> config = DatabaseAdapterConfig(
        ...     name="analytics_db",
        ...     connection_url="postgresql://user:pass@localhost/db",
        ...     engine_type="postgresql",
        ...     pool_size=10
        ... )
        >>> adapter = DatabaseAdapter(config)
        >>> 
        >>> # Safe parameterized query
        >>> result = adapter.execute(
        ...     query="SELECT * FROM orders WHERE status = :status",
        ...     params={"status": "completed"}
        ... )
        >>> print(result.output)  # List[Dict]
    
    Security:
        - Always use parameterized queries (:param syntax)
        - Never concatenate user input into SQL
        - Set allow_raw_sql=False in production
        - Use read-only database users when possible
        - Limit max_results to prevent memory exhaustion
    """
    
    def __init__(self, config: DatabaseAdapterConfig):
        if not HAS_SQLALCHEMY:
            raise AdapterExecutionError(
                "SQLAlchemy not installed. Install with: pip install sqlalchemy",
                adapter_name=config.name,
                adapter_type=config.adapter_type,
            )
        
        super().__init__(config)
        self.config: DatabaseAdapterConfig = config
        self._engine: Optional[Engine] = None
        self._metadata: Optional[MetaData] = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Setup SQLAlchemy engine with connection pooling."""
        try:
            self._engine = create_engine(
                self.config.connection_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False,  # Set to True for SQL debugging
            )
            
            self._metadata = MetaData()
            
            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        
        except Exception as e:
            raise AdapterExecutionError(
                f"Failed to connect to database: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def _execute_impl(self, **inputs: Any) -> Any:
        """Execute database query."""
        if not self._engine:
            raise AdapterExecutionError(
                "Database engine not initialized",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        # Extract query and parameters
        query = inputs.get("query")
        if not query:
            raise AdapterValidationError(
                "Missing required parameter: query",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        params = inputs.get("params", {})
        query_type = inputs.get("query_type", QueryType.SELECT)
        
        # Validate raw SQL
        if not self.config.allow_raw_sql:
            self._validate_query_safety(query)
        
        try:
            with self._engine.connect() as conn:
                # Execute query
                stmt = text(query)
                result = conn.execute(stmt, params)
                
                # Handle different query types
                if query_type == QueryType.SELECT or query.strip().upper().startswith("SELECT"):
                    # Fetch results
                    rows = result.fetchmany(self.config.max_results)
                    
                    # Convert to list of dicts
                    columns = result.keys()
                    output = [dict(zip(columns, row)) for row in rows]
                    
                    if self.config.autocommit:
                        conn.commit()
                    
                    return output
                
                else:
                    # INSERT, UPDATE, DELETE
                    rowcount = result.rowcount
                    
                    if self.config.autocommit:
                        conn.commit()
                    
                    return {
                        "rowcount": rowcount,
                        "status": "success",
                    }
        
        except Exception as e:
            error_msg = str(e)
            
            # Sanitize error message (don't leak connection details)
            if "password" in error_msg.lower():
                error_msg = "Database authentication failed"
            
            raise AdapterExecutionError(
                f"Query execution failed: {error_msg}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
                context={"query_type": query_type},
            )
    
    def _validate_query_safety(self, query: str):
        """Validate query for common SQL injection patterns.
        
        This is NOT a complete security solution - always use parameterized queries!
        """
        dangerous_patterns = [
            # String concatenation
            "' +",
            "' ||",
            "\" +",
            "\" ||",
            # Comments
            "--",
            "/*",
            "*/",
            # Stacked queries
            ";",
            # Union attacks
            "UNION SELECT",
            "UNION ALL SELECT",
        ]
        
        query_upper = query.upper()
        
        for pattern in dangerous_patterns:
            if pattern.upper() in query_upper:
                raise AdapterValidationError(
                    f"Query contains potentially dangerous pattern: {pattern}. "
                    f"Use parameterized queries with :param syntax instead.",
                    adapter_name=self.config.name,
                    adapter_type=self.config.adapter_type,
                )
    
    def execute_transaction(self, queries: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple queries in a transaction.
        
        Args:
            queries: List of query dicts with 'query' and 'params' keys
        
        Returns:
            List of results for each query
        
        Raises:
            AdapterExecutionError: Transaction failed
        
        Example:
            >>> results = adapter.execute_transaction([
            ...     {"query": "INSERT INTO orders (...) VALUES (:val)", "params": {"val": 1}},
            ...     {"query": "UPDATE inventory SET qty = qty - 1 WHERE id = :id", "params": {"id": 123}}
            ... ])
        """
        if not self._engine:
            raise AdapterExecutionError(
                "Database engine not initialized",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
        
        results = []
        
        try:
            with self._engine.begin() as conn:  # Auto-commit on success, rollback on error
                for query_dict in queries:
                    query = query_dict.get("query")
                    params = query_dict.get("params", {})
                    
                    if not query:
                        raise AdapterValidationError(
                            "Missing query in transaction",
                            adapter_name=self.config.name,
                            adapter_type=self.config.adapter_type,
                        )
                    
                    stmt = text(query)
                    result = conn.execute(stmt, params)
                    
                    if query.strip().upper().startswith("SELECT"):
                        rows = result.fetchmany(self.config.max_results)
                        columns = result.keys()
                        output = [dict(zip(columns, row)) for row in rows]
                        results.append(output)
                    else:
                        results.append({"rowcount": result.rowcount})
            
            return results
        
        except Exception as e:
            raise AdapterExecutionError(
                f"Transaction failed: {e}",
                adapter_name=self.config.name,
                adapter_type=self.config.adapter_type,
            )
    
    def __del__(self):
        """Cleanup database connections."""
        if self._engine:
            self._engine.dispose()
