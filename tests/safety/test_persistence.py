"""Tests for safety event persistence layer."""

import asyncio
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from namel3ss.safety.persistence import (
    SafetyEventDatabaseWriter,
    SafetyEventObjectStoreWriter,
    SafetyEventSink,
)
from namel3ss.safety.runtime import SafetyAction


# Mock SafetyEvent class for testing
@dataclass
class MockSafetyEvent:
    """Mock safety event for testing."""
    event_id: str
    timestamp: datetime
    policy_name: str
    action: SafetyAction
    direction: str
    latency_ms: float
    violation_category: Optional[str] = None
    violation_severity: Optional[str] = None
    violation_confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "policy_name": self.policy_name,
            "action": self.action.value if isinstance(self.action, SafetyAction) else self.action,
            "direction": self.direction,
            "latency_ms": self.latency_ms,
            "violation_category": self.violation_category,
            "violation_severity": self.violation_severity,
            "violation_confidence": self.violation_confidence,
        }


class MockAdapter:
    """Mock dataset adapter for testing."""
    
    def __init__(self, has_insert: bool = True, has_append: bool = False):
        self.inserted_rows = []
        self.appended_data = []
        self.write_calls = []
        
        # Dynamically add methods based on what should be available
        if has_insert:
            self.insert = self._insert_impl
        if has_append:
            self.append = self._append_impl
            
    def _insert_impl(self, rows: List[Dict[str, Any]]) -> None:
        """Insert implementation."""
        self.inserted_rows.extend(rows)
        
    def _append_impl(self, data: str) -> None:
        """Append implementation."""
        self.appended_data.append(data)
        
    def write(self, data: Any) -> None:
        """Write method (always available as final fallback)."""
        self.write_calls.append(data)


class MockAdapterRegistry:
    """Mock adapter registry for testing."""
    
    def __init__(self, adapter: Optional[MockAdapter] = None):
        self.adapter = adapter or MockAdapter()
        self.get_calls = []
        
    def get(self, dataset_name: str) -> MockAdapter:
        """Get adapter for dataset."""
        self.get_calls.append(dataset_name)
        return self.adapter


@pytest.fixture
def sample_events() -> List[MockSafetyEvent]:
    """Create sample safety events for testing."""
    now = datetime.utcnow()
    return [
        MockSafetyEvent(
            event_id="evt-001",
            timestamp=now,
            policy_name="input_pii_policy",
            action=SafetyAction.REDACT,
            direction="input",
            latency_ms=12.5,
            violation_category="PII",
            violation_severity="MEDIUM",
            violation_confidence=0.95,
        ),
        MockSafetyEvent(
            event_id="evt-002",
            timestamp=now + timedelta(seconds=1),
            policy_name="output_toxicity_policy",
            action=SafetyAction.BLOCK,
            direction="output",
            latency_ms=8.3,
            violation_category="TOXICITY",
            violation_severity="HIGH",
            violation_confidence=0.89,
        ),
        MockSafetyEvent(
            event_id="evt-003",
            timestamp=now + timedelta(seconds=2),
            policy_name="input_prompt_injection",
            action=SafetyAction.ALLOW,
            direction="input",
            latency_ms=5.1,
        ),
    ]


@pytest.mark.asyncio
class TestSafetyEventSink:
    """Tests for SafetyEventSink class."""
    
    async def test_init_creates_fallback_directory(self):
        """Test that fallback directory is created on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "fallback"
            registry = MockAdapterRegistry()
            
            sink = SafetyEventSink(
                dataset_name="test_events",
                adapter_registry=registry,
                fallback_path=str(fallback_path),
            )
            
            assert fallback_path.exists()
            assert fallback_path.is_dir()
    
    async def test_write_events_uses_insert_method(self, sample_events):
        """Test that events are written using insert method when available."""
        adapter = MockAdapter(has_insert=True)
        registry = MockAdapterRegistry(adapter)
        
        sink = SafetyEventSink(
            dataset_name="test_events",
            adapter_registry=registry,
        )
        
        await sink.write_events(sample_events)
        
        # Verify adapter was retrieved
        assert "test_events" in registry.get_calls
        
        # Verify events were inserted
        assert len(adapter.inserted_rows) == 3
        assert adapter.inserted_rows[0]["event_id"] == "evt-001"
        assert adapter.inserted_rows[1]["event_id"] == "evt-002"
        assert adapter.inserted_rows[2]["event_id"] == "evt-003"
    
    async def test_write_events_uses_append_method_fallback(self, sample_events):
        """Test that append method is used when insert is not available."""
        adapter = MockAdapter(has_insert=False, has_append=True)
        registry = MockAdapterRegistry(adapter)
        
        sink = SafetyEventSink(
            dataset_name="test_events",
            adapter_registry=registry,
        )
        
        await sink.write_events(sample_events)
        
        # Verify events were appended
        assert len(adapter.appended_data) == 3
        assert "evt-001" in adapter.appended_data[0]
        assert "evt-002" in adapter.appended_data[1]
    
    async def test_write_events_uses_write_method_fallback(self, sample_events):
        """Test that write method is used when insert/append not available."""
        adapter = MockAdapter(has_insert=False, has_append=False)
        registry = MockAdapterRegistry(adapter)
        
        sink = SafetyEventSink(
            dataset_name="test_events",
            adapter_registry=registry,
        )
        
        await sink.write_events(sample_events)
        
        # Verify write was called once with all rows
        assert len(adapter.write_calls) == 1
        assert len(adapter.write_calls[0]) == 3  # All 3 events in one call
    
    async def test_write_events_retries_on_failure(self, sample_events):
        """Test retry logic with exponential backoff."""
        call_count = 0
        
        class FailingAdapter(MockAdapter):
            def _insert_impl(self, rows):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("Temporary failure")
                self.inserted_rows.extend(rows)
        
        adapter = FailingAdapter()
        registry = MockAdapterRegistry(adapter)
        
        sink = SafetyEventSink(
            dataset_name="test_events",
            adapter_registry=registry,
            max_retries=3,
            retry_delay_seconds=0.01,  # Fast retries for testing
        )
        
        await sink.write_events(sample_events)
        
        # Verify retries occurred
        assert call_count == 3
        # Verify events were eventually written
        assert len(adapter.inserted_rows) == 3
    
    async def test_write_events_fallback_on_max_retries(self, sample_events):
        """Test fallback file persistence after max retries exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "fallback"
            
            class AlwaysFailingAdapter(MockAdapter):
                def _insert_impl(self, rows):
                    raise Exception("Persistent failure")
            
            adapter = AlwaysFailingAdapter()
            registry = MockAdapterRegistry(adapter)
            
            sink = SafetyEventSink(
                dataset_name="test_events",
                adapter_registry=registry,
                fallback_path=str(fallback_path),
                max_retries=2,
                retry_delay_seconds=0.01,
            )
            
            await sink.write_events(sample_events)
            
            # Verify fallback files were created
            fallback_files = list(fallback_path.glob("safety_events_*.json"))
            assert len(fallback_files) >= 1
            
            # Verify fallback file contains events
            with open(fallback_files[0]) as f:
                fallback_data = json.load(f)
                assert len(fallback_data) == 3
                assert fallback_data[0]["event_id"] == "evt-001"
    
    async def test_event_to_row_conversion(self, sample_events):
        """Test that events are correctly converted to row format."""
        adapter = MockAdapter()
        registry = MockAdapterRegistry(adapter)
        
        sink = SafetyEventSink(
            dataset_name="test_events",
            adapter_registry=registry,
        )
        
        await sink.write_events([sample_events[0]])
        
        row = adapter.inserted_rows[0]
        
        # Verify all fields are present
        assert row["event_id"] == "evt-001"
        assert row["policy_name"] == "input_pii_policy"
        assert row["action"].lower() == "redact"  # Action is lowercase from to_dict()
        assert row["direction"] == "input"
        assert row["latency_ms"] == 12.5
        assert row["violation_category"] == "PII"
        assert row["violation_severity"] == "MEDIUM"
        assert row["violation_confidence"] == 0.95
    
    async def test_empty_events_list(self):
        """Test that empty events list is handled gracefully."""
        adapter = MockAdapter()
        registry = MockAdapterRegistry(adapter)
        
        sink = SafetyEventSink(
            dataset_name="test_events",
            adapter_registry=registry,
        )
        
        await sink.write_events([])
        
        # Verify no writes occurred
        assert len(adapter.inserted_rows) == 0
    
    async def test_adapter_not_found_uses_fallback(self, sample_events):
        """Test fallback when adapter cannot be found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "fallback"
            
            class FailingRegistry:
                def get(self, dataset_name):
                    raise KeyError(f"Dataset {dataset_name} not found")
            
            registry = FailingRegistry()
            
            sink = SafetyEventSink(
                dataset_name="missing_dataset",
                adapter_registry=registry,
                fallback_path=str(fallback_path),
            )
            
            await sink.write_events(sample_events)
            
            # Verify fallback files were created
            fallback_files = list(fallback_path.glob("safety_events_*.json"))
            assert len(fallback_files) >= 1


@pytest.mark.asyncio
class TestSafetyEventDatabaseWriter:
    """Tests for SafetyEventDatabaseWriter class."""
    
    async def test_init_requires_connection_string(self):
        """Test that database writer requires connection string."""
        writer = SafetyEventDatabaseWriter(
            connection_string="sqlite:///:memory:",
            table_name="safety_events",
        )
        
        assert writer.connection_string == "sqlite:///:memory:"
        assert writer.table_name == "safety_events"
    
    async def test_write_events_empty_list(self):
        """Test that empty list is handled gracefully."""
        writer = SafetyEventDatabaseWriter(
            connection_string="sqlite:///:memory:",
            table_name="safety_events",
        )
        
        # Should not raise
        await writer.write_events([])


@pytest.mark.asyncio
class TestSafetyEventObjectStoreWriter:
    """Tests for SafetyEventObjectStoreWriter class."""
    
    async def test_init_sets_format(self):
        """Test that initialization sets format and bucket correctly."""
        writer = SafetyEventObjectStoreWriter(
            bucket_name="test-bucket",
            prefix="safety/events",
            format="json",
        )
        
        assert writer.bucket_name == "test-bucket"
        assert writer.prefix == "safety/events"
        assert writer.format == "json"
    
    async def test_write_events_empty_list(self):
        """Test that empty list is handled gracefully."""
        writer = SafetyEventObjectStoreWriter(
            bucket_name="test-bucket",
            prefix="safety/events",
            format="json",
        )
        
        # Should not raise
        await writer.write_events([])
    
    async def test_csv_format_serialization(self, sample_events):
        """Test CSV format serialization."""
        writer = SafetyEventObjectStoreWriter(
            bucket_name="test-bucket",
            prefix="safety/events",
            format="csv",
        )
        
        rows = [event.to_dict() for event in sample_events]
        buffer = await writer._write_csv(rows)
        
        # Verify buffer contains valid CSV
        buffer.seek(0)
        csv_data = buffer.read().decode('utf-8')
        assert "event_id" in csv_data  # Header
        assert "evt-001" in csv_data
        assert "evt-002" in csv_data
    
    async def test_parquet_format_requires_pyarrow(self, sample_events):
        """Test that parquet format requires pyarrow."""
        writer = SafetyEventObjectStoreWriter(
            bucket_name="test-bucket",
            prefix="safety/events",
            format="parquet",
        )
        
        rows = [event.to_dict() for event in sample_events]
        
        # This will fail if pyarrow is not installed
        with pytest.raises(ImportError):
            await writer._write_parquet(rows)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
