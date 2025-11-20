"""Factory for creating dataset loaders from Dataset AST nodes."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..ast.datasets import Dataset, DatasetConnectorConfig
from .loaders import (
    DatasetLoader,
    CSVDatasetLoader,
    JSONDatasetLoader,
    InlineDatasetLoader,
    DatabaseDatasetLoader,
)

logger = logging.getLogger(__name__)


class DatasetLoaderError(Exception):
    """Raised when dataset loader creation fails."""
    pass


def get_dataset_loader(
    dataset: Dataset,
    app_context: Optional[Any] = None,
) -> DatasetLoader:
    """
    Create a dataset loader from a Dataset AST node.
    
    Args:
        dataset: Dataset definition from the AST
        app_context: Optional application context (for resolving connectors, etc.)
        
    Returns:
        DatasetLoader instance
        
    Raises:
        DatasetLoaderError: If loader creation fails
    """
    dataset_name = dataset.name
    source_type = dataset.source_type.lower()
    source = dataset.source
    
    # Extract field mappings from metadata or config
    content_field = dataset.metadata.get("content_field", "content")
    id_field = dataset.metadata.get("id_field", None)
    metadata_fields = dataset.metadata.get("metadata_fields", None)
    
    # Handle custom connectors
    if dataset.connector:
        return _create_custom_loader(
            dataset_name=dataset_name,
            connector_config=dataset.connector,
            content_field=content_field,
            id_field=id_field,
            metadata_fields=metadata_fields,
            app_context=app_context,
        )
    
    # Handle built-in source types
    try:
        # CSV files
        if source_type in ("csv", "file") and (
            source.lower().endswith(".csv") or source.lower().endswith(".tsv")
        ):
            return _create_csv_loader(
                dataset_name=dataset_name,
                source=source,
                content_field=content_field,
                id_field=id_field,
                metadata_fields=metadata_fields,
                dataset=dataset,
            )
        
        # JSON files
        elif source_type in ("json", "jsonl", "file") and (
            source.lower().endswith(".json") or source.lower().endswith(".jsonl")
        ):
            return _create_json_loader(
                dataset_name=dataset_name,
                source=source,
                content_field=content_field,
                id_field=id_field,
                metadata_fields=metadata_fields,
                dataset=dataset,
            )
        
        # Inline datasets
        elif source_type == "inline":
            return _create_inline_loader(
                dataset_name=dataset_name,
                content_field=content_field,
                id_field=id_field,
                metadata_fields=metadata_fields,
                dataset=dataset,
            )
        
        # Database datasets
        elif source_type in ("sql", "database", "db"):
            return _create_database_loader(
                dataset_name=dataset_name,
                source=source,
                content_field=content_field,
                id_field=id_field,
                metadata_fields=metadata_fields,
                dataset=dataset,
                app_context=app_context,
            )
        
        else:
            raise DatasetLoaderError(
                f"Unsupported dataset source_type: '{source_type}' for dataset '{dataset_name}'. "
                f"Supported types: csv, json, jsonl, inline, sql, database. "
                f"For custom loaders, specify a connector configuration."
            )
    
    except DatasetLoaderError:
        raise
    except Exception as e:
        raise DatasetLoaderError(
            f"Failed to create loader for dataset '{dataset_name}': {e}"
        ) from e


def _create_csv_loader(
    dataset_name: str,
    source: str,
    content_field: str,
    id_field: Optional[str],
    metadata_fields: Optional[list],
    dataset: Dataset,
) -> CSVDatasetLoader:
    """Create a CSV dataset loader."""
    # Resolve file path
    file_path = Path(source).expanduser()
    if not file_path.is_absolute():
        # Resolve relative to current working directory
        file_path = Path.cwd() / file_path
    
    # Extract CSV-specific config
    config = {}
    if "delimiter" in dataset.metadata:
        config["delimiter"] = dataset.metadata["delimiter"]
    if "quotechar" in dataset.metadata:
        config["quotechar"] = dataset.metadata["quotechar"]
    
    logger.info(f"Creating CSV loader for dataset '{dataset_name}' from '{file_path}'")
    
    return CSVDatasetLoader(
        dataset_name=dataset_name,
        file_path=file_path,
        content_field=content_field,
        id_field=id_field,
        metadata_fields=metadata_fields,
        config=config,
    )


def _create_json_loader(
    dataset_name: str,
    source: str,
    content_field: str,
    id_field: Optional[str],
    metadata_fields: Optional[list],
    dataset: Dataset,
) -> JSONDatasetLoader:
    """Create a JSON/JSONL dataset loader."""
    # Resolve file path
    file_path = Path(source).expanduser()
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    
    # Detect JSONL format
    is_jsonl = source.lower().endswith(".jsonl") or dataset.metadata.get("format") == "jsonl"
    
    logger.info(
        f"Creating JSON{'L' if is_jsonl else ''} loader for dataset '{dataset_name}' from '{file_path}'"
    )
    
    return JSONDatasetLoader(
        dataset_name=dataset_name,
        file_path=file_path,
        content_field=content_field,
        id_field=id_field,
        metadata_fields=metadata_fields,
        is_jsonl=is_jsonl,
        config={},
    )


def _create_inline_loader(
    dataset_name: str,
    content_field: str,
    id_field: Optional[str],
    metadata_fields: Optional[list],
    dataset: Dataset,
) -> InlineDatasetLoader:
    """Create an inline dataset loader."""
    # Get records from metadata
    records = dataset.metadata.get("records", [])
    
    if not records:
        logger.warning(f"No records found for inline dataset '{dataset_name}'")
    
    logger.info(f"Creating inline loader for dataset '{dataset_name}' with {len(records)} records")
    
    return InlineDatasetLoader(
        dataset_name=dataset_name,
        records=records,
        content_field=content_field,
        id_field=id_field,
        metadata_fields=metadata_fields,
        config={},
    )


def _create_database_loader(
    dataset_name: str,
    source: str,
    content_field: str,
    id_field: Optional[str],
    metadata_fields: Optional[list],
    dataset: Dataset,
    app_context: Optional[Any],
) -> DatabaseDatasetLoader:
    """Create a database dataset loader."""
    # Get query from source or metadata
    query = source if source else dataset.metadata.get("query", "")
    
    if not query:
        raise DatasetLoaderError(
            f"No SQL query specified for database dataset '{dataset_name}'. "
            f"Set 'source' to a SQL query or specify 'query' in metadata."
        )
    
    # Get database connector
    # This assumes the app_context has a method to retrieve connectors by name
    # For now, we'll use a placeholder that needs to be wired up
    if not app_context:
        raise DatasetLoaderError(
            f"Database loader requires app_context to resolve connectors for dataset '{dataset_name}'"
        )
    
    # Get connector configuration
    connector_name = dataset.metadata.get("connector", "default")
    
    # Try to get connector from app_context
    # This will need to be implemented based on the actual connector infrastructure
    if hasattr(app_context, "get_connector"):
        connector = app_context.get_connector(connector_name)
    else:
        raise DatasetLoaderError(
            f"Cannot resolve database connector '{connector_name}' for dataset '{dataset_name}'. "
            f"App context does not support connector resolution."
        )
    
    logger.info(f"Creating database loader for dataset '{dataset_name}' using connector '{connector_name}'")
    
    # Extract query parameters
    query_params = dataset.metadata.get("query_params", {})
    
    return DatabaseDatasetLoader(
        dataset_name=dataset_name,
        connector=connector,
        query=query,
        content_field=content_field,
        id_field=id_field,
        metadata_fields=metadata_fields,
        config={"query_params": query_params},
    )


def _create_custom_loader(
    dataset_name: str,
    connector_config: DatasetConnectorConfig,
    content_field: str,
    id_field: Optional[str],
    metadata_fields: Optional[list],
    app_context: Optional[Any],
) -> DatasetLoader:
    """
    Create a custom dataset loader from connector configuration.
    
    The connector_config should specify either:
    - connector_type: A module path like "my_project.loaders.CustomLoader"
    - connector_name: A registered connector name
    
    The custom loader class must implement the DatasetLoader protocol.
    """
    connector_type = connector_config.connector_type
    connector_name = connector_config.connector_name
    options = connector_config.options or {}
    
    # Try to dynamically import and instantiate the loader
    try:
        if "." in connector_type:
            # Treat as module path: "package.module.ClassName"
            module_path, class_name = connector_type.rsplit(".", 1)
            module = importlib.import_module(module_path)
            loader_class = getattr(module, class_name)
        else:
            # Try to resolve from a registry or known loaders
            raise DatasetLoaderError(
                f"Connector type '{connector_type}' is not a valid module path. "
                f"Use format 'package.module.ClassName' for custom loaders."
            )
        
        # Instantiate the loader
        # Pass standard arguments plus options
        loader = loader_class(
            dataset_name=dataset_name,
            content_field=content_field,
            id_field=id_field,
            metadata_fields=metadata_fields,
            config=options,
            connector_name=connector_name,
            app_context=app_context,
        )
        
        # Verify it implements the protocol
        if not hasattr(loader, "iter_documents"):
            raise DatasetLoaderError(
                f"Custom loader '{connector_type}' does not implement iter_documents method"
            )
        
        logger.info(
            f"Created custom loader '{connector_type}' for dataset '{dataset_name}'"
        )
        
        return loader
    
    except ImportError as e:
        raise DatasetLoaderError(
            f"Failed to import custom loader '{connector_type}' for dataset '{dataset_name}': {e}"
        ) from e
    except AttributeError as e:
        raise DatasetLoaderError(
            f"Failed to find loader class in '{connector_type}' for dataset '{dataset_name}': {e}"
        ) from e
    except Exception as e:
        raise DatasetLoaderError(
            f"Failed to instantiate custom loader '{connector_type}' for dataset '{dataset_name}': {e}"
        ) from e


__all__ = [
    "get_dataset_loader",
    "DatasetLoaderError",
]
