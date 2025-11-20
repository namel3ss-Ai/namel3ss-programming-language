"""Dataset loaders for RAG indexing pipeline."""

from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


class LoadedDocument(TypedDict):
    """
    A document loaded from a dataset.
    
    Attributes:
        id: Unique document identifier
        content: Full text content for embedding
        metadata: Additional metadata (tags, source, timestamps, etc.)
    """
    id: str
    content: str
    metadata: Dict[str, Any]


@runtime_checkable
class DatasetLoader(Protocol):
    """
    Protocol for dataset loaders that provide async document iteration.
    
    All loaders must implement async iteration to support streaming
    large datasets without loading everything into memory.
    """
    
    async def iter_documents(
        self,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
    ) -> AsyncIterator[LoadedDocument]:
        """
        Iterate over documents in the dataset.
        
        Args:
            limit: Maximum number of documents to yield (None for all)
            filters: Metadata filters to apply (e.g., {"tag": "support", "lang": "en"})
            offset: Number of documents to skip before yielding
            
        Yields:
            LoadedDocument instances
        """
        ...


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    Provides common utilities for document ID generation, field mapping,
    and metadata extraction.
    """
    
    def __init__(
        self,
        dataset_name: str,
        content_field: str = "content",
        id_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base loader.
        
        Args:
            dataset_name: Name of the dataset
            content_field: Field name to use as document content
            id_field: Field name to use as document ID (auto-generated if None)
            metadata_fields: List of fields to include in metadata (None = all except content)
            config: Additional loader-specific configuration
        """
        self.dataset_name = dataset_name
        self.content_field = content_field
        self.id_field = id_field
        self.metadata_fields = metadata_fields
        self.config = config or {}
        self._doc_counter = 0
    
    def _generate_doc_id(self, record: Dict[str, Any]) -> str:
        """Generate a document ID from a record."""
        if self.id_field and self.id_field in record:
            return str(record[self.id_field])
        else:
            self._doc_counter += 1
            return f"{self.dataset_name}_{self._doc_counter}"
    
    def _extract_content(self, record: Dict[str, Any]) -> str:
        """Extract content field from a record."""
        content = record.get(self.content_field, "")
        if not isinstance(content, str):
            content = str(content)
        return content.strip()
    
    def _extract_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata fields from a record."""
        metadata = {"source": self.dataset_name}
        
        if self.metadata_fields is None:
            # Include all fields except content
            for key, value in record.items():
                if key != self.content_field:
                    metadata[key] = value
        else:
            # Include only specified fields
            for field in self.metadata_fields:
                if field in record:
                    metadata[field] = record[field]
        
        return metadata
    
    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if metadata matches the given filters."""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        
        return True
    
    def _create_document(self, record: Dict[str, Any]) -> Optional[LoadedDocument]:
        """
        Create a LoadedDocument from a record.
        
        Returns None if the record is invalid or missing required fields.
        """
        try:
            content = self._extract_content(record)
            if not content:
                logger.warning(
                    f"Skipping record with empty content in dataset '{self.dataset_name}'"
                )
                return None
            
            doc_id = self._generate_doc_id(record)
            metadata = self._extract_metadata(record)
            
            return LoadedDocument(
                id=doc_id,
                content=content,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(
                f"Error creating document from record in dataset '{self.dataset_name}': {e}"
            )
            return None
    
    @abstractmethod
    async def iter_documents(
        self,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
    ) -> AsyncIterator[LoadedDocument]:
        """Iterate over documents in the dataset."""
        ...


class CSVDatasetLoader(BaseDatasetLoader):
    """
    Load documents from CSV files.
    
    Supports:
    - Field mapping for content and metadata
    - Safe parsing with error recovery
    - Large file streaming
    """
    
    def __init__(
        self,
        dataset_name: str,
        file_path: Path,
        content_field: str = "content",
        id_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize CSV loader.
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to CSV file
            content_field: Column name for document content
            id_field: Column name for document ID
            metadata_fields: List of column names to include in metadata
            config: Additional config (e.g., csv.DictReader kwargs)
        """
        super().__init__(dataset_name, content_field, id_field, metadata_fields, config)
        self.file_path = file_path
    
    async def iter_documents(
        self,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
    ) -> AsyncIterator[LoadedDocument]:
        """Iterate over documents from CSV file."""
        if not self.file_path.exists():
            logger.error(f"CSV file not found: {self.file_path}")
            return
        
        yielded = 0
        skipped = 0
        
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                # Extract csv.DictReader kwargs from config
                reader_kwargs = {
                    k: v for k, v in self.config.items()
                    if k in ("delimiter", "quotechar", "escapechar", "doublequote", "skipinitialspace")
                }
                reader = csv.DictReader(f, **reader_kwargs)
                
                for row_num, row in enumerate(reader, start=1):
                    try:
                        # Skip offset rows
                        if skipped < offset:
                            skipped += 1
                            continue
                        
                        # Create document
                        doc = self._create_document(row)
                        if doc is None:
                            continue
                        
                        # Apply filters
                        if not self._matches_filters(doc["metadata"], filters):
                            continue
                        
                        yield doc
                        yielded += 1
                        
                        # Check limit
                        if limit is not None and yielded >= limit:
                            break
                    
                    except Exception as e:
                        logger.error(
                            f"Error processing CSV row {row_num} in '{self.dataset_name}': {e}"
                        )
                        continue
        
        except Exception as e:
            logger.error(f"Error reading CSV file '{self.file_path}': {e}")


class JSONDatasetLoader(BaseDatasetLoader):
    """
    Load documents from JSON files.
    
    Supports:
    - JSON arrays: [{"content": "...", ...}, ...]
    - Line-delimited JSON (JSONL): one JSON object per line
    - Field mapping and metadata extraction
    """
    
    def __init__(
        self,
        dataset_name: str,
        file_path: Path,
        content_field: str = "content",
        id_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        is_jsonl: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize JSON loader.
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to JSON/JSONL file
            content_field: Field name for document content
            id_field: Field name for document ID
            metadata_fields: List of fields to include in metadata
            is_jsonl: True if file is line-delimited JSON
            config: Additional configuration
        """
        super().__init__(dataset_name, content_field, id_field, metadata_fields, config)
        self.file_path = file_path
        self.is_jsonl = is_jsonl
    
    async def iter_documents(
        self,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
    ) -> AsyncIterator[LoadedDocument]:
        """Iterate over documents from JSON file."""
        if not self.file_path.exists():
            logger.error(f"JSON file not found: {self.file_path}")
            return
        
        yielded = 0
        skipped = 0
        
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                if self.is_jsonl:
                    # Line-delimited JSON
                    for line_num, line in enumerate(f, start=1):
                        try:
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Skip offset rows
                            if skipped < offset:
                                skipped += 1
                                continue
                            
                            record = json.loads(line)
                            doc = self._create_document(record)
                            if doc is None:
                                continue
                            
                            if not self._matches_filters(doc["metadata"], filters):
                                continue
                            
                            yield doc
                            yielded += 1
                            
                            if limit is not None and yielded >= limit:
                                break
                        
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Invalid JSON on line {line_num} in '{self.dataset_name}': {e}"
                            )
                            continue
                        except Exception as e:
                            logger.error(
                                f"Error processing line {line_num} in '{self.dataset_name}': {e}"
                            )
                            continue
                
                else:
                    # Standard JSON array
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            logger.error(
                                f"Expected JSON array in '{self.file_path}', got {type(data).__name__}"
                            )
                            return
                        
                        for idx, record in enumerate(data):
                            try:
                                # Skip offset rows
                                if skipped < offset:
                                    skipped += 1
                                    continue
                                
                                doc = self._create_document(record)
                                if doc is None:
                                    continue
                                
                                if not self._matches_filters(doc["metadata"], filters):
                                    continue
                                
                                yield doc
                                yielded += 1
                                
                                if limit is not None and yielded >= limit:
                                    break
                            
                            except Exception as e:
                                logger.error(
                                    f"Error processing record {idx} in '{self.dataset_name}': {e}"
                                )
                                continue
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in file '{self.file_path}': {e}")
        
        except Exception as e:
            logger.error(f"Error reading JSON file '{self.file_path}': {e}")


class InlineDatasetLoader(BaseDatasetLoader):
    """
    Load documents from inline records (defined in metadata).
    
    Useful for testing and small static datasets.
    """
    
    def __init__(
        self,
        dataset_name: str,
        records: List[Dict[str, Any]],
        content_field: str = "content",
        id_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize inline loader.
        
        Args:
            dataset_name: Name of the dataset
            records: List of record dictionaries
            content_field: Field name for document content
            id_field: Field name for document ID
            metadata_fields: List of fields to include in metadata
            config: Additional configuration
        """
        super().__init__(dataset_name, content_field, id_field, metadata_fields, config)
        self.records = records
    
    async def iter_documents(
        self,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
    ) -> AsyncIterator[LoadedDocument]:
        """Iterate over inline documents."""
        yielded = 0
        skipped = 0
        
        for idx, record in enumerate(self.records):
            try:
                # Skip offset rows
                if skipped < offset:
                    skipped += 1
                    continue
                
                doc = self._create_document(record)
                if doc is None:
                    continue
                
                if not self._matches_filters(doc["metadata"], filters):
                    continue
                
                yield doc
                yielded += 1
                
                if limit is not None and yielded >= limit:
                    break
            
            except Exception as e:
                logger.error(
                    f"Error processing inline record {idx} in '{self.dataset_name}': {e}"
                )
                continue


class DatabaseDatasetLoader(BaseDatasetLoader):
    """
    Load documents from database queries.
    
    Supports:
    - Parameterized SQL queries
    - Streaming results
    - Multiple database backends via connectors
    """
    
    def __init__(
        self,
        dataset_name: str,
        connector: Any,  # Database connector instance
        query: str,
        content_field: str = "content",
        id_field: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize database loader.
        
        Args:
            dataset_name: Name of the dataset
            connector: Database connector instance with execute_query method
            query: SQL query to fetch records
            content_field: Column name for document content
            id_field: Column name for document ID
            metadata_fields: List of columns to include in metadata
            config: Additional configuration (e.g., query parameters)
        """
        super().__init__(dataset_name, content_field, id_field, metadata_fields, config)
        self.connector = connector
        self.query = query
    
    async def iter_documents(
        self,
        *,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        offset: int = 0,
    ) -> AsyncIterator[LoadedDocument]:
        """Iterate over documents from database query."""
        # Build query with limit and offset
        query = self.query
        params = self.config.get("query_params", {})
        
        # Apply limit and offset at SQL level if possible
        if limit is not None or offset > 0:
            if "LIMIT" not in query.upper() and "OFFSET" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit or 'ALL'} OFFSET {offset}"
        
        yielded = 0
        
        try:
            # Execute query
            async for row in self.connector.execute_query(query, params):
                try:
                    # Convert row to dict (handle different result formats)
                    if isinstance(row, dict):
                        record = row
                    elif hasattr(row, '_asdict'):
                        record = row._asdict()
                    elif hasattr(row, 'keys'):
                        record = dict(zip(row.keys(), row))
                    else:
                        logger.warning(f"Unexpected row format: {type(row)}")
                        continue
                    
                    doc = self._create_document(record)
                    if doc is None:
                        continue
                    
                    if not self._matches_filters(doc["metadata"], filters):
                        continue
                    
                    yield doc
                    yielded += 1
                    
                    if limit is not None and yielded >= limit:
                        break
                
                except Exception as e:
                    logger.error(
                        f"Error processing database row in '{self.dataset_name}': {e}"
                    )
                    continue
        
        except Exception as e:
            logger.error(f"Error executing database query in '{self.dataset_name}': {e}")


__all__ = [
    "LoadedDocument",
    "DatasetLoader",
    "BaseDatasetLoader",
    "CSVDatasetLoader",
    "JSONDatasetLoader",
    "InlineDatasetLoader",
    "DatabaseDatasetLoader",
]
