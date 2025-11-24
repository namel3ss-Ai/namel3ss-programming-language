"""
SQL query compiler for dataset operations.

This module provides safe SQL query generation for dataset CRUD operations
with support for pagination, filtering, sorting, and search.
"""

import textwrap
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compile_dataset_to_sql(
    dataset_name: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
    search: Optional[str] = None,
    searchable_fields: List[str] = None,
    filters: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Compile a dataset read query to SQL with pagination and filtering.
    
    Args:
        dataset_name: Name of the dataset
        page: Page number (1-based)
        page_size: Items per page
        sort_by: Column to sort by
        sort_order: Sort order ('asc' or 'desc')
        search: Search term
        searchable_fields: Fields to search in
        filters: Additional filters
        
    Returns:
        Dictionary with 'query', 'count_query', and 'params'
    """
    table_name = _sanitize_table_name(dataset_name)
    searchable_fields = searchable_fields or []
    filters = filters or {}
    
    # Build WHERE clause
    where_conditions = []
    params = {}
    param_counter = 0
    
    # Add search conditions
    if search and searchable_fields:
        search_conditions = []
        for field in searchable_fields:
            field_name = _sanitize_column_name(field)
            param_name = f"search_{param_counter}"
            search_conditions.append(f"{field_name} ILIKE %({param_name})s")
            params[param_name] = f"%{search}%"
            param_counter += 1
        
        if search_conditions:
            where_conditions.append(f"({' OR '.join(search_conditions)})")
    
    # Add filter conditions
    for field, value in filters.items():
        if value is not None:
            field_name = _sanitize_column_name(field)
            param_name = f"filter_{param_counter}"
            where_conditions.append(f"{field_name} = %({param_name})s")
            params[param_name] = value
            param_counter += 1
    
    # Build WHERE clause
    where_clause = ""
    if where_conditions:
        where_clause = f"WHERE {' AND '.join(where_conditions)}"
    
    # Build ORDER BY clause
    order_clause = ""
    if sort_by:
        sort_column = _sanitize_column_name(sort_by)
        sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        order_clause = f"ORDER BY {sort_column} {sort_direction}"
    
    # Build main query with pagination
    offset = (page - 1) * page_size
    query = f"""
        SELECT * FROM {table_name}
        {where_clause}
        {order_clause}
        LIMIT {page_size} OFFSET {offset}
    """.strip()
    
    # Build count query
    count_query = f"""
        SELECT COUNT(*) FROM {table_name}
        {where_clause}
    """.strip()
    
    return {
        "query": query,
        "count_query": count_query,
        "params": params,
    }


def compile_dataset_insert(
    dataset_name: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compile dataset insert operation to SQL.
    
    Args:
        dataset_name: Name of the dataset
        data: Data to insert
        
    Returns:
        Dictionary with 'query', 'params', and 'select_query'
    """
    table_name = _sanitize_table_name(dataset_name)
    
    # Filter out None values and prepare columns/values
    filtered_data = {k: v for k, v in data.items() if v is not None}
    
    if not filtered_data:
        raise ValueError("No valid data provided for insert")
    
    columns = [_sanitize_column_name(col) for col in filtered_data.keys()]
    placeholders = [f"%({col})s" for col in filtered_data.keys()]
    
    query = f"""
        INSERT INTO {table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        RETURNING id
    """.strip()
    
    # Query to select the created record
    select_query = f"""
        SELECT * FROM {table_name} WHERE id = %(id)s
    """.strip()
    
    return {
        "query": query,
        "params": filtered_data,
        "select_query": select_query,
    }


def compile_dataset_update(
    dataset_name: str,
    record_id: str,
    data: Dict[str, Any],
    primary_key: str = "id",
) -> Dict[str, Any]:
    """
    Compile dataset update operation to SQL.
    
    Args:
        dataset_name: Name of the dataset
        record_id: ID of record to update
        data: Data to update
        primary_key: Primary key column name
        
    Returns:
        Dictionary with 'query', 'params', and 'select_query'
    """
    table_name = _sanitize_table_name(dataset_name)
    pk_column = _sanitize_column_name(primary_key)
    
    # Filter out None values and prepare SET clause
    filtered_data = {k: v for k, v in data.items() if v is not None}
    
    if not filtered_data:
        raise ValueError("No valid data provided for update")
    
    set_clauses = []
    for column in filtered_data.keys():
        safe_column = _sanitize_column_name(column)
        set_clauses.append(f"{safe_column} = %({column})s")
    
    # Add primary key to params
    params = dict(filtered_data)
    params[primary_key] = record_id
    
    query = f"""
        UPDATE {table_name}
        SET {', '.join(set_clauses)}
        WHERE {pk_column} = %({primary_key})s
    """.strip()
    
    # Query to select the updated record
    select_query = f"""
        SELECT * FROM {table_name} WHERE {pk_column} = %({primary_key})s
    """.strip()
    
    return {
        "query": query,
        "params": params,
        "select_query": select_query,
    }


def compile_dataset_delete(
    dataset_name: str,
    record_id: str,
    primary_key: str = "id",
) -> Dict[str, Any]:
    """
    Compile dataset delete operation to SQL.
    
    Args:
        dataset_name: Name of the dataset
        record_id: ID of record to delete
        primary_key: Primary key column name
        
    Returns:
        Dictionary with 'query' and 'params'
    """
    table_name = _sanitize_table_name(dataset_name)
    pk_column = _sanitize_column_name(primary_key)
    
    query = f"""
        DELETE FROM {table_name}
        WHERE {pk_column} = %({primary_key})s
    """.strip()
    
    return {
        "query": query,
        "params": {primary_key: record_id},
    }


def _sanitize_table_name(name: str) -> str:
    """Sanitize table name to prevent SQL injection."""
    # Remove any non-alphanumeric characters except underscores
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    # Ensure it starts with a letter
    if not sanitized or not sanitized[0].isalpha():
        raise ValueError(f"Invalid table name: {name}")
    
    return sanitized


def _sanitize_column_name(name: str) -> str:
    """Sanitize column name to prevent SQL injection."""
    # Remove any non-alphanumeric characters except underscores
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    # Ensure it starts with a letter or underscore
    if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == '_'):
        raise ValueError(f"Invalid column name: {name}")
    
    return sanitized