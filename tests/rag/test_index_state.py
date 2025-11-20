"""Tests for index state persistence."""

import pytest
from pathlib import Path

from namel3ss.rag.index_state import IndexState, IndexStateManager


class TestIndexState:
    """Tests for IndexState dataclass."""
    
    def test_create_state(self):
        """Test creating a new index state."""
        state = IndexState(
            index_name="test_index",
            dataset_name="test_dataset",
        )
        
        assert state.index_name == "test_index"
        assert state.dataset_name == "test_dataset"
        assert state.total_documents == 0
        assert state.total_chunks == 0
        assert state.total_tokens == 0
        assert len(state.processed_document_ids) == 0
        assert state.completed is False
        assert state.started_at is not None
        assert state.updated_at is not None
    
    def test_mark_processed(self):
        """Test marking documents as processed."""
        state = IndexState(
            index_name="test_index",
            dataset_name="test_dataset",
        )
        
        state.mark_processed("doc1", chunks=5, tokens=100)
        
        assert state.total_documents == 1
        assert state.total_chunks == 5
        assert state.total_tokens == 100
        assert state.last_processed_id == "doc1"
        assert "doc1" in state.processed_document_ids
    
    def test_is_processed(self):
        """Test checking if document is processed."""
        state = IndexState(
            index_name="test_index",
            dataset_name="test_dataset",
        )
        
        state.mark_processed("doc1", chunks=5, tokens=100)
        
        assert state.is_processed("doc1") is True
        assert state.is_processed("doc2") is False
    
    def test_mark_completed(self):
        """Test marking index as completed."""
        state = IndexState(
            index_name="test_index",
            dataset_name="test_dataset",
        )
        
        state.mark_completed()
        
        assert state.completed is True
    
    def test_accumulate_stats(self):
        """Test accumulating statistics across documents."""
        state = IndexState(
            index_name="test_index",
            dataset_name="test_dataset",
        )
        
        state.mark_processed("doc1", chunks=5, tokens=100)
        state.mark_processed("doc2", chunks=3, tokens=75)
        state.mark_processed("doc3", chunks=7, tokens=150)
        
        assert state.total_documents == 3
        assert state.total_chunks == 15
        assert state.total_tokens == 325
        assert len(state.processed_document_ids) == 3


class TestIndexStateManager:
    """Tests for IndexStateManager."""
    
    def test_create_manager(self, tmp_path):
        """Test creating state manager with custom directory."""
        state_dir = tmp_path / "index_states"
        manager = IndexStateManager(state_dir)
        
        assert manager.state_dir == state_dir
        assert state_dir.exists()
    
    def test_create_state(self, tmp_path):
        """Test creating a new state."""
        manager = IndexStateManager(tmp_path)
        
        state = manager.create_state(
            index_name="test_index",
            dataset_name="test_dataset",
            metadata={"embedding_model": "text-embedding-3-small"},
        )
        
        assert state.index_name == "test_index"
        assert state.dataset_name == "test_dataset"
        assert state.metadata["embedding_model"] == "text-embedding-3-small"
    
    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state."""
        manager = IndexStateManager(tmp_path)
        
        # Create and save state
        state = manager.create_state("test_index", "test_dataset")
        state.mark_processed("doc1", chunks=5, tokens=100)
        state.mark_processed("doc2", chunks=3, tokens=75)
        manager.save_state(state)
        
        # Load state
        loaded_state = manager.load_state("test_index", "test_dataset")
        
        assert loaded_state is not None
        assert loaded_state.index_name == "test_index"
        assert loaded_state.dataset_name == "test_dataset"
        assert loaded_state.total_documents == 2
        assert loaded_state.total_chunks == 8
        assert loaded_state.total_tokens == 175
        assert "doc1" in loaded_state.processed_document_ids
        assert "doc2" in loaded_state.processed_document_ids
    
    def test_load_nonexistent_state(self, tmp_path):
        """Test loading state that doesn't exist."""
        manager = IndexStateManager(tmp_path)
        
        state = manager.load_state("nonexistent_index", "nonexistent_dataset")
        
        assert state is None
    
    def test_delete_state(self, tmp_path):
        """Test deleting state."""
        manager = IndexStateManager(tmp_path)
        
        # Create and save state
        state = manager.create_state("test_index", "test_dataset")
        manager.save_state(state)
        
        # Verify it exists
        loaded_state = manager.load_state("test_index", "test_dataset")
        assert loaded_state is not None
        
        # Delete state
        manager.delete_state("test_index", "test_dataset")
        
        # Verify it's gone
        loaded_state = manager.load_state("test_index", "test_dataset")
        assert loaded_state is None
    
    def test_resume_scenario(self, tmp_path):
        """Test resuming index build scenario."""
        manager = IndexStateManager(tmp_path)
        
        # First run: process some documents
        state = manager.create_state("docs_index", "articles")
        state.mark_processed("doc1", chunks=5, tokens=100)
        state.mark_processed("doc2", chunks=3, tokens=75)
        manager.save_state(state)
        
        # Resume: load existing state
        resumed_state = manager.load_state("docs_index", "articles")
        assert resumed_state is not None
        assert resumed_state.total_documents == 2
        assert resumed_state.is_processed("doc1")
        assert resumed_state.is_processed("doc2")
        assert not resumed_state.is_processed("doc3")
        
        # Continue processing
        resumed_state.mark_processed("doc3", chunks=7, tokens=150)
        resumed_state.mark_completed()
        manager.save_state(resumed_state)
        
        # Verify final state
        final_state = manager.load_state("docs_index", "articles")
        assert final_state.total_documents == 3
        assert final_state.completed is True
    
    def test_force_rebuild_scenario(self, tmp_path):
        """Test force rebuild scenario."""
        manager = IndexStateManager(tmp_path)
        
        # First build
        state = manager.create_state("docs_index", "articles")
        state.mark_processed("doc1", chunks=5, tokens=100)
        state.mark_completed()
        manager.save_state(state)
        
        # Force rebuild: delete state
        manager.delete_state("docs_index", "articles")
        
        # Create fresh state
        new_state = manager.create_state("docs_index", "articles")
        assert new_state.total_documents == 0
        assert len(new_state.processed_document_ids) == 0
    
    def test_multiple_indices(self, tmp_path):
        """Test managing multiple indices."""
        manager = IndexStateManager(tmp_path)
        
        # Create states for different indices
        state1 = manager.create_state("index1", "dataset1")
        state1.mark_processed("doc1", chunks=5, tokens=100)
        manager.save_state(state1)
        
        state2 = manager.create_state("index2", "dataset2")
        state2.mark_processed("doc2", chunks=3, tokens=75)
        manager.save_state(state2)
        
        # Load them back
        loaded1 = manager.load_state("index1", "dataset1")
        loaded2 = manager.load_state("index2", "dataset2")
        
        assert loaded1.index_name == "index1"
        assert loaded2.index_name == "index2"
        assert loaded1.total_documents == 1
        assert loaded2.total_documents == 1
    
    def test_state_path_sanitization(self, tmp_path):
        """Test that state paths are sanitized properly."""
        manager = IndexStateManager(tmp_path)
        
        # Names with special characters
        state = manager.create_state("index/with/slashes", "dataset\\with\\backslashes")
        manager.save_state(state)
        
        # Should be able to load it back
        loaded = manager.load_state("index/with/slashes", "dataset\\with\\backslashes")
        assert loaded is not None
        assert loaded.index_name == "index/with/slashes"


class TestIndexStateEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_state_with_large_processed_set(self, tmp_path):
        """Test state with many processed document IDs."""
        manager = IndexStateManager(tmp_path)
        
        state = manager.create_state("test_index", "test_dataset")
        
        # Process many documents
        for i in range(10000):
            state.mark_processed(f"doc{i}", chunks=1, tokens=50)
        
        manager.save_state(state)
        
        # Load and verify
        loaded = manager.load_state("test_index", "test_dataset")
        assert loaded is not None
        assert loaded.total_documents == 10000
        assert len(loaded.processed_document_ids) == 10000
    
    def test_state_with_metadata(self, tmp_path):
        """Test state with custom metadata."""
        manager = IndexStateManager(tmp_path)
        
        metadata = {
            "embedding_model": "text-embedding-3-small",
            "chunk_size": 512,
            "overlap": 64,
            "custom_field": "custom_value",
        }
        
        state = manager.create_state("test_index", "test_dataset", metadata=metadata)
        manager.save_state(state)
        
        loaded = manager.load_state("test_index", "test_dataset")
        assert loaded.metadata == metadata
