"""
Factory for creating dataset adapters from AST dataset definitions.

Provides unified interface for converting various dataset sources
(CSV, SQL, in-memory) into logic adapters for query execution.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from namel3ss.ast import App
from namel3ss.ast.datasets import Dataset, DatasetSchemaField
from namel3ss.codegen.backend.core.runtime.logic_adapters import (
    AdapterRegistry,
    DatasetAdapter,
)


# ============================================================================
# Adapter Creation
# ============================================================================

def create_adapter_registry(app: App) -> AdapterRegistry:
    """
    Create an adapter registry populated from app datasets.
    
    Args:
        app: Application AST containing dataset definitions
        
    Returns:
        AdapterRegistry with all datasets registered as adapters
    """
    registry = AdapterRegistry()
    
    for dataset in app.datasets:
        try:
            adapter = create_dataset_adapter(dataset)
            if adapter:
                registry.register(dataset.name, adapter)
        except Exception as e:
            # Log but don't fail - allow queries to run with partial data
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to create adapter for dataset {dataset.name!r}: {e}"
            )
    
    return registry


def create_dataset_adapter(dataset: Dataset) -> Optional[DatasetAdapter]:
    """
    Create a dataset adapter from a dataset AST node.
    
    Args:
        dataset: Dataset AST node
        
    Returns:
        DatasetAdapter or None if dataset cannot be loaded
    """
    source_type = dataset.source_type.lower()
    
    if source_type == "csv":
        return _create_csv_adapter(dataset)
    elif source_type == "json":
        return _create_json_adapter(dataset)
    elif source_type == "sql":
        return _create_sql_adapter(dataset)
    elif source_type == "inline":
        return _create_inline_adapter(dataset)
    else:
        # Unknown source type - log and skip
        import logging
        logging.getLogger(__name__).warning(
            f"Unknown dataset source type: {source_type!r} for dataset {dataset.name!r}"
        )
        return None


# ============================================================================
# CSV Adapter
# ============================================================================

def _create_csv_adapter(dataset: Dataset) -> Optional[DatasetAdapter]:
    """Create adapter for CSV dataset."""
    source_path = dataset.source
    
    if not source_path:
        return None
    
    # Resolve relative paths
    path = Path(source_path)
    if not path.is_absolute():
        # Try to resolve relative to runtime directory
        # In production, this would be configured via environment
        path = Path.cwd() / source_path
    
    if not path.exists():
        import logging
        logging.getLogger(__name__).warning(
            f"CSV file not found: {source_path} for dataset {dataset.name!r}"
        )
        return None
    
    try:
        # Read CSV file
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            records = list(reader)
        
        # Infer schema from records
        schema = _infer_schema_from_records(records, dataset.schema)
        
        # Convert schema list to dict for DatasetAdapter
        schema_dict = {f.name: f.dtype for f in schema}
        
        return DatasetAdapter(
            dataset_name=dataset.name,
            schema=schema_dict,
            records=records,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to load CSV dataset {dataset.name!r}: {e}"
        )
        return None


# ============================================================================
# JSON Adapter
# ============================================================================

def _create_json_adapter(dataset: Dataset) -> Optional[DatasetAdapter]:
    """Create adapter for JSON dataset."""
    source_path = dataset.source
    
    if not source_path:
        return None
    
    # Resolve relative paths
    path = Path(source_path)
    if not path.is_absolute():
        path = Path.cwd() / source_path
    
    if not path.exists():
        import logging
        logging.getLogger(__name__).warning(
            f"JSON file not found: {source_path} for dataset {dataset.name!r}"
        )
        return None
    
    try:
        # Read JSON file
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Expect array of objects
        if not isinstance(data, list):
            data = [data]
        
        records = data
        
        # Infer schema from records
        schema = _infer_schema_from_records(records, dataset.schema)
        
        # Convert schema list to dict for DatasetAdapter
        schema_dict = {f.name: f.dtype for f in schema}
        
        return DatasetAdapter(
            dataset_name=dataset.name,
            schema=schema_dict,
            records=records,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to load JSON dataset {dataset.name!r}: {e}"
        )
        return None


# ============================================================================
# SQL Adapter
# ============================================================================

def _create_sql_adapter(dataset: Dataset) -> Optional[DatasetAdapter]:
    """
    Create adapter for SQL dataset.
    
    Note: This is a simplified implementation. In production, you would:
    - Use SQLAlchemy or similar ORM
    - Load connection details from dataset.connector
    - Execute queries with proper parameterization
    - Handle pagination at the database level
    """
    import logging
    logging.getLogger(__name__).info(
        f"SQL adapter not fully implemented for dataset {dataset.name!r}. "
        "Using empty dataset."
    )
    
    # For now, return empty dataset with schema from AST if provided
    schema = dataset.schema if dataset.schema else []
    schema_dict = {f.name: f.dtype for f in schema} if schema else {}
    
    return DatasetAdapter(
        dataset_name=dataset.name,
        schema=schema_dict,
        records=[],
    )


# ============================================================================
# Inline Adapter
# ============================================================================

def _create_inline_adapter(dataset: Dataset) -> Optional[DatasetAdapter]:
    """
    Create adapter for inline dataset.
    
    Inline datasets have records embedded in the AST (used for testing).
    """
    # Check if dataset has inline records in metadata
    records = dataset.metadata.get("records", [])
    
    if not records:
        return None
    
    # Infer schema from records
    schema = _infer_schema_from_records(records, dataset.schema)
    
    # Convert schema list to dict for DatasetAdapter
    schema_dict = {f.name: f.dtype for f in schema}
    
    return DatasetAdapter(
        dataset_name=dataset.name,
        schema=schema_dict,
        records=records,
    )


# ============================================================================
# Schema Inference
# ============================================================================

def _infer_schema_from_records(
    records: List[Dict[str, Any]],
    schema_fields: Optional[List[Any]] = None,
) -> List[DatasetSchemaField]:
    """
    Infer schema from records.
    
    If schema_fields is provided (from AST), use that as the source of truth.
    Otherwise, infer types from the data.
    
    Args:
        records: List of record dictionaries
        schema_fields: Optional list of DatasetSchemaField from AST
        
    Returns:
        List of DatasetSchemaField objects
    """
    # If explicit schema provided, use it
    if schema_fields:
        return schema_fields
    
    # Otherwise, infer from data
    if not records:
        return []
    
    # Get field names from first record (or union of all records)
    all_fields = set()
    for record in records:
        if isinstance(record, dict):
            all_fields.update(record.keys())
        else:
            # Skip non-dict records
            import logging
            logging.getLogger(__name__).warning(
                f"Skipping non-dict record in schema inference: {type(record)}"
            )
            continue
    
    schema = []
    for field_name in sorted(all_fields):
        # Infer type from first non-None value
        field_type = "string"  # default
        
        for record in records:
            value = record.get(field_name)
            if value is not None:
                if isinstance(value, bool):
                    field_type = "bool"
                elif isinstance(value, int):
                    field_type = "int"
                elif isinstance(value, float):
                    field_type = "float"
                elif isinstance(value, str):
                    field_type = "string"
                break
        
        schema.append(DatasetSchemaField(name=field_name, dtype=field_type))
    
    return schema


__all__ = [
    "create_adapter_registry",
    "create_dataset_adapter",
]
