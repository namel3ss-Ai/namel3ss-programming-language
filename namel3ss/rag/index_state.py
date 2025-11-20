"""Index state persistence for resumable indexing."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class IndexState:
    """
    State tracking for resumable index builds.
    
    Attributes:
        index_name: Name of the index
        dataset_name: Name of the source dataset
        last_processed_id: ID of the last successfully processed document
        processed_document_ids: Set of all processed document IDs
        total_documents: Total documents processed
        total_chunks: Total chunks created
        total_tokens: Total embedding tokens used
        started_at: ISO timestamp when indexing started
        updated_at: ISO timestamp when state was last updated
        completed: Whether indexing completed successfully
        metadata: Additional metadata (chunk_size, overlap, model, etc.)
    """
    index_name: str
    dataset_name: str
    last_processed_id: Optional[str] = None
    processed_document_ids: Set[str] = None
    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.processed_document_ids is None:
            self.processed_document_ids = set()
        if self.metadata is None:
            self.metadata = {}
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc).isoformat()
        if self.updated_at is None:
            self.updated_at = self.started_at
    
    def mark_processed(self, doc_id: str, chunks: int = 0, tokens: int = 0):
        """Mark a document as processed."""
        self.processed_document_ids.add(doc_id)
        self.last_processed_id = doc_id
        self.total_documents += 1
        self.total_chunks += chunks
        self.total_tokens += tokens
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def is_processed(self, doc_id: str) -> bool:
        """Check if a document has been processed."""
        return doc_id in self.processed_document_ids
    
    def mark_completed(self):
        """Mark indexing as completed."""
        self.completed = True
        self.updated_at = datetime.now(timezone.utc).isoformat()


class IndexStateManager:
    """
    Manages persistence of index build state.
    
    Stores state in JSON files for resumability across runs.
    """
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory for storing state files (default: ~/.namel3ss/index_states)
        """
        if state_dir is None:
            state_dir = Path.home() / ".namel3ss" / "index_states"
        
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_state_path(self, index_name: str, dataset_name: str) -> Path:
        """Get the path to the state file for an index."""
        safe_name = f"{index_name}_{dataset_name}".replace("/", "_").replace("\\", "_")
        return self.state_dir / f"{safe_name}.json"
    
    def load_state(
        self,
        index_name: str,
        dataset_name: str,
    ) -> Optional[IndexState]:
        """
        Load index state from disk.
        
        Args:
            index_name: Name of the index
            dataset_name: Name of the dataset
            
        Returns:
            IndexState if found, None otherwise
        """
        state_path = self._get_state_path(index_name, dataset_name)
        
        if not state_path.exists():
            return None
        
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert processed_document_ids from list to set
            if "processed_document_ids" in data and isinstance(data["processed_document_ids"], list):
                data["processed_document_ids"] = set(data["processed_document_ids"])
            
            state = IndexState(**data)
            logger.info(
                f"Loaded index state for '{index_name}' (dataset: '{dataset_name}'): "
                f"{state.total_documents} documents, {state.total_chunks} chunks"
            )
            return state
        
        except Exception as e:
            logger.error(f"Failed to load index state from '{state_path}': {e}")
            return None
    
    def save_state(self, state: IndexState):
        """
        Save index state to disk.
        
        Args:
            state: IndexState to save
        """
        state_path = self._get_state_path(state.index_name, state.dataset_name)
        
        try:
            # Convert state to dict and handle sets
            state_dict = asdict(state)
            if "processed_document_ids" in state_dict and isinstance(
                state_dict["processed_document_ids"], set
            ):
                state_dict["processed_document_ids"] = list(state_dict["processed_document_ids"])
            
            # Write to temp file first, then rename (atomic operation)
            temp_path = state_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
            
            temp_path.rename(state_path)
            
        except Exception as e:
            logger.error(f"Failed to save index state to '{state_path}': {e}")
            raise
    
    def delete_state(self, index_name: str, dataset_name: str):
        """
        Delete index state.
        
        Args:
            index_name: Name of the index
            dataset_name: Name of the dataset
        """
        state_path = self._get_state_path(index_name, dataset_name)
        
        try:
            if state_path.exists():
                state_path.unlink()
                logger.info(
                    f"Deleted index state for '{index_name}' (dataset: '{dataset_name}')"
                )
        except Exception as e:
            logger.error(f"Failed to delete index state at '{state_path}': {e}")
    
    def create_state(
        self,
        index_name: str,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IndexState:
        """
        Create a new index state.
        
        Args:
            index_name: Name of the index
            dataset_name: Name of the dataset
            metadata: Initial metadata
            
        Returns:
            New IndexState
        """
        return IndexState(
            index_name=index_name,
            dataset_name=dataset_name,
            metadata=metadata or {},
        )


__all__ = [
    "IndexState",
    "IndexStateManager",
]
