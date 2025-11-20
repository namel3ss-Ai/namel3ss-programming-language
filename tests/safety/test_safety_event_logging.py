"""Integration tests for safety event logging with persistence."""

import asyncio
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from namel3ss.safety.logging import (
    SafetyEvent,
    SafetyEventLogger,
    get_global_logger,
    set_global_logger,
)
from namel3ss.safety.persistence import SafetyEventSink
from namel3ss.safety.runtime import SafetyAction


class MockAdapter:
    """Mock dataset adapter for testing."""
    
    def __init__(self, fail_times: int = 0):
        self.inserted_rows = []
        self.call_count = 0
        self.fail_times = fail_times
        
    def insert(self, rows: List[Dict[str, Any]]) -> None:
        """Mock insert method."""
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise Exception(f"Simulated failure {self.call_count}")
        self.inserted_rows.extend(rows)


class MockAdapterRegistry:
    """Mock adapter registry for testing."""
    
    def __init__(self, adapter: Optional[MockAdapter] = None):
        self.adapter = adapter or MockAdapter()
        
    def get(self, dataset_name: str) -> MockAdapter:
        """Get adapter for dataset."""
        return self.adapter


@pytest.fixture
def mock_registry():
    """Create mock adapter registry."""
    return MockAdapterRegistry()


@pytest.fixture
def sample_event():
    """Create sample safety event."""
    return SafetyEvent(
        event_id="test-001",
        timestamp=datetime.utcnow(),
        policy_name="test_policy",
        action=SafetyAction.ALLOW,
        direction="input",
        latency_ms=10.5,
        violation_category=None,
        violation_severity=None,
        violation_confidence=None,
        chain_name="test_chain",
        user_id="user-123",
        session_id="session-456",
        metadata={"test": "data"},
    )


@pytest.mark.asyncio
class TestSafetyEventLoggerIntegration:
    """Integration tests for SafetyEventLogger with persistence."""
    
    async def test_logger_persists_events_on_buffer_full(self, mock_registry, sample_event):
        """Test that events are persisted when buffer fills up."""
        adapter = mock_registry.adapter
        
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=3,
            adapter_registry=mock_registry,
        )
        
        await logger.start()
        
        try:
            # Log 3 events to fill buffer
            for i in range(3):
                event = SafetyEvent(
                    event_id=f"evt-{i}",
                    timestamp=datetime.utcnow(),
                    policy_name="test_policy",
                    action=SafetyAction.ALLOW,
                    direction="input",
                    latency_ms=10.0,
                )
                await logger.log(event)
            
            # Wait a bit for async flush
            await asyncio.sleep(0.1)
            
            # Verify events were persisted
            assert len(adapter.inserted_rows) == 3
            assert adapter.inserted_rows[0]["event_id"] == "evt-0"
            assert adapter.inserted_rows[1]["event_id"] == "evt-1"
            assert adapter.inserted_rows[2]["event_id"] == "evt-2"
            
        finally:
            await logger.stop()
    
    async def test_logger_periodic_flush(self, mock_registry, sample_event):
        """Test that events are flushed periodically."""
        adapter = mock_registry.adapter
        
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=100,  # Large buffer
            flush_interval_seconds=0.2,  # Fast flush for testing
            adapter_registry=mock_registry,
        )
        
        await logger.start()
        
        try:
            # Log 2 events (less than buffer size)
            await logger.log(sample_event)
            
            event2 = SafetyEvent(
                event_id="test-002",
                timestamp=datetime.utcnow(),
                policy_name="test_policy",
                action=SafetyAction.BLOCK,
                direction="output",
                latency_ms=15.3,
            )
            await logger.log(event2)
            
            # Wait for periodic flush
            await asyncio.sleep(0.3)
            
            # Verify events were persisted
            assert len(adapter.inserted_rows) == 2
            
        finally:
            await logger.stop()
    
    async def test_logger_explicit_flush(self, mock_registry, sample_event):
        """Test explicit flush operation."""
        adapter = mock_registry.adapter
        
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=100,
            flush_interval_seconds=60.0,  # Long interval
            adapter_registry=mock_registry,
        )
        
        await logger.log(sample_event)
        
        # Explicitly flush
        await logger.flush()
        
        # Verify event was persisted
        assert len(adapter.inserted_rows) == 1
        assert adapter.inserted_rows[0]["event_id"] == "test-001"
    
    async def test_logger_stop_flushes_remaining_events(self, mock_registry, sample_event):
        """Test that stop() flushes any remaining buffered events."""
        adapter = mock_registry.adapter
        
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=100,
            flush_interval_seconds=60.0,
            adapter_registry=mock_registry,
        )
        
        await logger.start()
        
        # Log events without triggering flush
        await logger.log(sample_event)
        
        event2 = SafetyEvent(
            event_id="test-002",
            timestamp=datetime.utcnow(),
            policy_name="test_policy",
            action=SafetyAction.REDACT,
            direction="input",
            latency_ms=8.7,
        )
        await logger.log(event2)
        
        # Stop should flush remaining events
        await logger.stop()
        
        # Verify events were persisted
        assert len(adapter.inserted_rows) == 2
    
    async def test_logger_overflow_drop_oldest(self, mock_registry):
        """Test drop_oldest overflow strategy."""
        adapter = mock_registry.adapter
        
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=3,
            flush_interval_seconds=60.0,
            overflow_strategy="drop_oldest",
            adapter_registry=mock_registry,
        )
        
        # Log 4 events (more than buffer size)
        for i in range(4):
            event = SafetyEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.utcnow(),
                policy_name="test_policy",
                action=SafetyAction.ALLOW,
                direction="input",
                latency_ms=10.0,
            )
            await logger.log(event)
        
        # Flush
        await logger.flush()
        
        # Should have events 1, 2, 3 (evt-0 was dropped)
        assert len(adapter.inserted_rows) == 3
        event_ids = [row["event_id"] for row in adapter.inserted_rows]
        assert "evt-0" not in event_ids
        assert "evt-1" in event_ids
        assert "evt-2" in event_ids
        assert "evt-3" in event_ids
        
        # Check stats
        stats = logger.get_stats()
        assert stats["total_logged"] == 4
        assert stats["total_dropped"] == 1
    
    async def test_logger_overflow_drop_newest(self, mock_registry):
        """Test drop_newest overflow strategy."""
        adapter = mock_registry.adapter
        
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=3,
            flush_interval_seconds=60.0,
            overflow_strategy="drop_newest",
            adapter_registry=mock_registry,
        )
        
        # Log 4 events
        for i in range(4):
            event = SafetyEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.utcnow(),
                policy_name="test_policy",
                action=SafetyAction.ALLOW,
                direction="input",
                latency_ms=10.0,
            )
            await logger.log(event)
        
        # Flush
        await logger.flush()
        
        # Should have events 0, 1, 2 (evt-3 was dropped)
        assert len(adapter.inserted_rows) == 3
        event_ids = [row["event_id"] for row in adapter.inserted_rows]
        assert "evt-0" in event_ids
        assert "evt-1" in event_ids
        assert "evt-2" in event_ids
        assert "evt-3" not in event_ids
        
        # Check stats
        stats = logger.get_stats()
        assert stats["total_logged"] == 4
        assert stats["total_dropped"] == 1
    
    async def test_logger_overflow_block(self, mock_registry):
        """Test block overflow strategy."""
        adapter = mock_registry.adapter
        
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=3,
            flush_interval_seconds=60.0,
            overflow_strategy="block",
            adapter_registry=mock_registry,
        )
        
        # Log 4 events (should trigger flush on overflow)
        for i in range(4):
            event = SafetyEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.utcnow(),
                policy_name="test_policy",
                action=SafetyAction.ALLOW,
                direction="input",
                latency_ms=10.0,
            )
            await logger.log(event)
        
        # All events should be persisted (with intermediate flush)
        stats = logger.get_stats()
        assert stats["total_logged"] == 4
        assert stats["total_dropped"] == 0
        
        # Flush remaining
        await logger.flush()
        
        # All events should be in adapter
        assert len(adapter.inserted_rows) == 4
    
    async def test_logger_retries_on_adapter_failure(self, mock_registry):
        """Test that logger retries on adapter failures."""
        # Create adapter that fails twice then succeeds
        failing_adapter = MockAdapter(fail_times=2)
        mock_registry.adapter = failing_adapter
        
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=2,
            adapter_registry=mock_registry,
        )
        
        # Configure sink with fast retries
        logger._sink.max_retries = 3
        logger._sink.retry_delay = 0.01
        
        event = SafetyEvent(
            event_id="test-retry",
            timestamp=datetime.utcnow(),
            policy_name="test_policy",
            action=SafetyAction.ALLOW,
            direction="input",
            latency_ms=10.0,
        )
        
        await logger.log(event)
        await logger.flush()
        
        # Verify event was eventually persisted
        assert len(failing_adapter.inserted_rows) == 1
        assert failing_adapter.call_count == 3  # Failed twice, succeeded on third
    
    async def test_logger_fallback_on_persistent_failure(self, mock_registry):
        """Test fallback file persistence on persistent adapter failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "fallback"
            
            # Create adapter that always fails
            always_failing = MockAdapter(fail_times=999)
            mock_registry.adapter = always_failing
            
            logger = SafetyEventLogger(
                target_dataset="test_events",
                buffer_size=2,
                adapter_registry=mock_registry,
                fallback_path=str(fallback_path),
            )
            
            # Configure sink with fast retries
            logger._sink.max_retries = 2
            logger._sink.retry_delay = 0.01
            
            event = SafetyEvent(
                event_id="test-fallback",
                timestamp=datetime.utcnow(),
                policy_name="test_policy",
                action=SafetyAction.BLOCK,
                direction="output",
                latency_ms=12.0,
            )
            
            await logger.log(event)
            await logger.flush()
            
            # Verify fallback file was created
            fallback_files = list(fallback_path.glob("safety_events_*.json"))
            assert len(fallback_files) >= 1
            
            # Verify event is in fallback file
            with open(fallback_files[0]) as f:
                data = json.load(f)
                assert len(data) >= 1
                assert any(evt["event_id"] == "test-fallback" for evt in data)
    
    async def test_logger_stats_tracking(self, mock_registry, sample_event):
        """Test that logger tracks statistics correctly."""
        logger = SafetyEventLogger(
            target_dataset="test_events",
            buffer_size=2,
            adapter_registry=mock_registry,
        )
        
        # Initial stats
        stats = logger.get_stats()
        assert stats["total_logged"] == 0
        assert stats["total_flushed"] == 0
        assert stats["total_dropped"] == 0
        assert stats["current_buffer_size"] == 0
        
        # Log events
        await logger.log(sample_event)
        
        stats = logger.get_stats()
        assert stats["total_logged"] == 1
        assert stats["current_buffer_size"] == 1
        
        # Flush
        await logger.flush()
        
        stats = logger.get_stats()
        assert stats["total_flushed"] == 1
        assert stats["current_buffer_size"] == 0
    
    async def test_global_logger_singleton(self):
        """Test global logger singleton pattern."""
        logger1 = get_global_logger()
        logger2 = get_global_logger()
        
        assert logger1 is logger2
    
    async def test_set_global_logger(self, mock_registry):
        """Test setting custom global logger."""
        custom_logger = SafetyEventLogger(
            target_dataset="custom_events",
            buffer_size=50,
            adapter_registry=mock_registry,
        )
        
        set_global_logger(custom_logger)
        
        retrieved = get_global_logger()
        assert retrieved is custom_logger
        assert retrieved.target_dataset == "custom_events"
        assert retrieved.buffer_size == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
