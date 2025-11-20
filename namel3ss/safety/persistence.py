"""Safety event persistence sink using frames/datasets system.

This module provides production-grade persistence of safety events
to the namel3ss frames/datasets abstraction, supporting multiple backends
(DB, object store, message queue) via dataset adapters.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from namel3ss.codegen.backend.core.runtime.logic_adapters import (
    AdapterRegistry,
    DatasetAdapter,
)

logger = logging.getLogger(__name__)


class SafetyEventSink:
    """Persistence sink for safety events using dataset adapters.
    
    This class provides the bridge between the safety logging system
    and the frames/datasets persistence layer. It handles:
    - Converting SafetyEvent objects to dataset rows
    - Writing to configured dataset adapters (DB, object store, queue)
    - Error handling and retry logic
    - Fallback persistence for critical events
    """
    
    def __init__(
        self,
        dataset_name: str = "safety_events",
        adapter_registry: Optional[AdapterRegistry] = None,
        fallback_path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        """Initialize safety event sink.
        
        Args:
            dataset_name: Name of dataset to write events to
            adapter_registry: Registry of dataset adapters  
            fallback_path: Path for fallback file persistence on adapter failure
            max_retries: Maximum number of retry attempts
            retry_delay_seconds: Delay between retries
        """
        self.dataset_name = dataset_name
        self.adapter_registry = adapter_registry or AdapterRegistry()
        self.fallback_path = fallback_path
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        
        self._adapter: Optional[DatasetAdapter] = None
        self._fallback_count = 0
        
        # Create fallback directory if path specified
        if self.fallback_path:
            Path(self.fallback_path).mkdir(parents=True, exist_ok=True)
        
    def _get_adapter(self) -> Optional[DatasetAdapter]:
        """Get or create dataset adapter for safety events.
        
        Returns:
            DatasetAdapter instance or None if not available
        """
        if self._adapter is None:
            try:
                self._adapter = self.adapter_registry.get(self.dataset_name)
            except KeyError:
                logger.warning(
                    f"Dataset adapter '{self.dataset_name}' not found in registry. "
                    "Safety events will use fallback persistence."
                )
                return None
        
        return self._adapter
    
    async def write_events(self, events: List[Any]) -> None:
        """Write safety events to persistent storage.
        
        This method attempts to write events using the configured dataset
        adapter. If that fails, it falls back to file-based persistence
        and logs critical errors.
        
        Args:
            events: List of SafetyEvent objects to persist
            
        Raises:
            Does not raise exceptions - handles errors internally to prevent
            disruption of main application flow
        """
        if not events:
            return
        
        # Convert events to dicts
        try:
            rows = [self._event_to_row(event) for event in events]
        except Exception as e:
            logger.error(
                f"Failed to convert {len(events)} safety events to rows: {e}",
                exc_info=True
            )
            await self._fallback_write(events)
            return
        
        # Attempt to write via adapter
        adapter = self._get_adapter()
        
        if adapter is None:
            # No adapter available - use fallback
            logger.warning(
                f"No dataset adapter available for '{self.dataset_name}'. "
                f"Writing {len(events)} events to fallback storage."
            )
            await self._fallback_write(events)
            return
        
        # Try to write with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                await self._write_to_adapter(adapter, rows)
                logger.debug(
                    f"Successfully wrote {len(events)} safety events to "
                    f"dataset '{self.dataset_name}'"
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed to write safety events (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                
                if attempt < self.max_retries - 1:
                    # Wait before retry
                    import asyncio
                    await asyncio.sleep(self.retry_delay_seconds * (attempt + 1))
        
        # All retries failed - use fallback
        logger.error(
            f"Failed to write {len(events)} safety events after {self.max_retries} attempts. "
            f"Last error: {last_error}",
            exc_info=True
        )
        await self._fallback_write(events)
    
    async def _write_to_adapter(self, adapter: DatasetAdapter, rows: List[Dict[str, Any]]) -> None:
        """Write rows to dataset adapter.
        
        Args:
            adapter: Dataset adapter instance
            rows: List of row dictionaries
        """
        # Use adapter's insert method if available (synchronous)
        if hasattr(adapter, 'insert'):
            adapter.insert(rows)
        elif hasattr(adapter, 'append'):
            # Append method might expect strings/bytes for some adapters
            for row in rows:
                adapter.append(json.dumps(row))
        elif hasattr(adapter, 'write'):
            adapter.write(rows)
        else:
            # Fall back to modifying records directly
            # This is for in-memory adapters
            if hasattr(adapter, 'records'):
                adapter.records.extend(rows)
            else:
                raise NotImplementedError(
                    f"Adapter {type(adapter).__name__} does not support write operations"
                )
    
    def _event_to_row(self, event: Any) -> Dict[str, Any]:
        """Convert SafetyEvent to row dictionary.
        
        Args:
            event: SafetyEvent instance
            
        Returns:
            Dictionary with event data suitable for dataset storage
        """
        # Use the event's to_dict method if available
        if hasattr(event, 'to_dict'):
            return event.to_dict()
        
        # Fall back to dataclass conversion
        if hasattr(event, '__dataclass_fields__'):
            row = asdict(event)
            # Convert datetime to ISO format
            if 'timestamp' in row and hasattr(row['timestamp'], 'isoformat'):
                row['timestamp'] = row['timestamp'].isoformat()
            return row
        
        # Last resort - try to convert to dict
        return dict(event)
    
    async def _fallback_write(self, events: List[Any]) -> None:
        """Write events to fallback storage (file-based).
        
        This method is used when the primary dataset adapter fails.
        Events are written to a local file for manual recovery.
        
        Args:
            events: List of SafetyEvent objects
        """
        if not self.fallback_path:
            logger.critical(
                f"No fallback path configured. {len(events)} safety events LOST. "
                "Configure fallback_path to prevent data loss."
            )
            return
        
        try:
            import json
            from pathlib import Path
            from datetime import datetime
            
            fallback_dir = Path(self.fallback_path)
            fallback_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped fallback file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fallback_file = fallback_dir / f"safety_events_{timestamp}_{self._fallback_count}.json"
            self._fallback_count += 1
            
            # Convert events to dicts
            rows = [self._event_to_row(event) for event in events]
            
            # Write to file
            with open(fallback_file, 'w', encoding='utf-8') as f:
                json.dump(rows, f, indent=2, default=str)
            
            logger.warning(
                f"Wrote {len(events)} safety events to fallback file: {fallback_file}"
            )
            
        except Exception as e:
            logger.critical(
                f"Fallback write failed! {len(events)} safety events LOST: {e}",
                exc_info=True
            )


class SafetyEventDatabaseWriter:
    """Specialized writer for SQL database backends.
    
    This class provides optimized bulk insert operations for SQL databases,
    with support for transactions and connection pooling.
    """
    
    def __init__(self, connection_string: str, table_name: str = "safety_events"):
        """Initialize database writer.
        
        Args:
            connection_string: SQLAlchemy connection string
            table_name: Name of table to write to
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self._engine = None
        
    async def write_events(self, rows: List[Dict[str, Any]]) -> None:
        """Write events to database using bulk insert.
        
        Args:
            rows: List of event row dictionaries
        """
        if not rows:
            return
        
        try:
            from sqlalchemy import create_engine, MetaData, Table, insert
            from sqlalchemy.ext.asyncio import create_async_engine
            
            # Initialize engine if needed
            if self._engine is None:
                # Check if async is supported
                if self.connection_string.startswith('postgresql+asyncpg://') or \
                   self.connection_string.startswith('sqlite+aiosqlite://'):
                    self._engine = create_async_engine(self.connection_string)
                else:
                    self._engine = create_engine(self.connection_string)
            
            # Perform bulk insert
            if hasattr(self._engine, 'begin'):
                # Async engine
                async with self._engine.begin() as conn:
                    metadata = MetaData()
                    await conn.run_sync(metadata.reflect, bind=conn)
                    
                    if self.table_name in metadata.tables:
                        table = metadata.tables[self.table_name]
                        await conn.execute(insert(table), rows)
                    else:
                        logger.error(f"Table '{self.table_name}' not found in database")
            else:
                # Sync engine
                with self._engine.begin() as conn:
                    metadata = MetaData()
                    metadata.reflect(bind=conn)
                    
                    if self.table_name in metadata.tables:
                        table = metadata.tables[self.table_name]
                        conn.execute(insert(table), rows)
                    else:
                        logger.error(f"Table '{self.table_name}' not found in database")
                        
        except Exception as e:
            logger.error(f"Database write failed: {e}", exc_info=True)
            raise


class SafetyEventObjectStoreWriter:
    """Specialized writer for object store backends (S3, GCS, Azure Blob).
    
    This class provides efficient batch writes to object stores,
    with support for parquet format and partitioning.
    """
    
    def __init__(
        self,
        bucket_name: str,
        prefix: str = "safety_events",
        format: str = "parquet",
    ):
        """Initialize object store writer.
        
        Args:
            bucket_name: Name of S3/GCS/Azure bucket
            prefix: Prefix/path for event files
            format: File format (parquet, json, csv)
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.format = format
        
    async def write_events(self, rows: List[Dict[str, Any]]) -> None:
        """Write events to object store.
        
        Args:
            rows: List of event row dictionaries
        """
        if not rows:
            return
        
        try:
            from datetime import datetime
            import io
            
            # Create partition path by date
            now = datetime.utcnow()
            partition_path = f"{self.prefix}/year={now.year}/month={now.month:02d}/day={now.day:02d}"
            
            # Generate file name
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            file_name = f"{partition_path}/events_{timestamp}.{self.format}"
            
            # Convert to appropriate format
            if self.format == "parquet":
                buffer = await self._write_parquet(rows)
            elif self.format == "json":
                import json
                buffer = io.BytesIO(json.dumps(rows, default=str).encode('utf-8'))
            elif self.format == "csv":
                buffer = await self._write_csv(rows)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
            
            # Write to object store (implementation depends on cloud provider)
            await self._upload_to_store(file_name, buffer)
            
            logger.debug(
                f"Wrote {len(rows)} safety events to {self.bucket_name}/{file_name}"
            )
            
        except Exception as e:
            logger.error(f"Object store write failed: {e}", exc_info=True)
            raise
    
    async def _write_parquet(self, rows: List[Dict[str, Any]]) -> io.BytesIO:
        """Convert rows to parquet format."""
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
            
            # Create table
            table = pa.Table.from_pylist(rows)
            
            # Write to buffer
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            buffer.seek(0)
            
            return buffer
        except ImportError as e:
            logger.error("pyarrow not available for parquet format - install with: pip install pyarrow")
            raise ImportError("parquet format requires pyarrow package") from e
    
    async def _write_csv(self, rows: List[Dict[str, Any]]) -> io.BytesIO:
        """Convert rows to CSV format."""
        import csv
        
        if not rows:
            return io.BytesIO()
        
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        
        # Convert string buffer to bytes
        bytes_buffer = io.BytesIO(buffer.getvalue().encode('utf-8'))
        bytes_buffer.seek(0)
        return bytes_buffer
        writer.writerows(rows)
        
        return io.BytesIO(buffer.getvalue().encode('utf-8'))
    
    async def _upload_to_store(self, file_name: str, buffer: io.BytesIO) -> None:
        """Upload buffer to object store.
        
        This is a placeholder - actual implementation depends on cloud provider.
        """
        # TODO: Implement actual upload logic based on cloud provider
        # For now, write to local filesystem as fallback
        from pathlib import Path
        
        local_path = Path(f"/tmp/{self.bucket_name}")
        local_path.mkdir(parents=True, exist_ok=True)
        
        file_path = local_path / file_name.replace('/', '_')
        with open(file_path, 'wb') as f:
            f.write(buffer.read())
        
        logger.info(f"Wrote to local fallback: {file_path}")
