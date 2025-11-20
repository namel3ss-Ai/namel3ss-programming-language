"""Safety event logging to frames/datasets for audit trails.

This module provides logging of safety violations and policy enforcement
actions to the Namel3ss frames/datasets system for audit and compliance.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .runtime import SafetyAction, SafetyViolation
from .persistence import SafetyEventSink

logger = logging.getLogger(__name__)


@dataclass
class SafetyEvent:
    """A single safety-related event for audit logging."""
    
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Context
    app_name: Optional[str] = None
    chain_name: Optional[str] = None
    agent_name: Optional[str] = None
    policy_name: Optional[str] = None
    
    # Direction and action
    direction: str = "input"  # input or output
    action: SafetyAction = SafetyAction.ALLOW
    
    # Violation details (if any)
    categories: list[str] = field(default_factory=list)
    severity: str = "none"
    confidence: float = 1.0
    reason: Optional[str] = None
    
    # Performance
    latency_ms: float = 0.0
    
    # Correlation
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Metadata (never include sensitive content)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "app_name": self.app_name,
            "chain_name": self.chain_name,
            "agent_name": self.agent_name,
            "policy_name": self.policy_name,
            "direction": self.direction,
            "action": self.action.value if isinstance(self.action, SafetyAction) else self.action,
            "categories": self.categories,
            "severity": self.severity,
            "confidence": self.confidence,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


class SafetyEventLogger:
    """Logger for safety events with async buffering and frame/dataset support.
    
    This logger provides:
    - Buffered event collection with automatic flushing
    - Integration with frames/datasets for persistence
    - Back-pressure handling to prevent unbounded memory growth
    - Graceful degradation on persistence failures
    - Configurable retention and overflow strategies
    """
    
    def __init__(
        self,
        target_frame: Optional[str] = None,
        target_dataset: Optional[str] = None,
        buffer_size: int = 100,
        flush_interval_seconds: float = 10.0,
        event_sink: Optional[SafetyEventSink] = None,
        adapter_registry: Optional[Any] = None,
        overflow_strategy: str = "drop_oldest",  # drop_oldest, drop_newest, block
        fallback_path: Optional[str] = None,
    ):
        """Initialize safety event logger.
        
        Args:
            target_frame: Name of frame to log to (if using frames)
            target_dataset: Name of dataset to log to (recommended: "safety_events")
            buffer_size: Number of events to buffer before flushing
            flush_interval_seconds: Time between automatic flushes
            event_sink: Pre-configured SafetyEventSink instance
            adapter_registry: Dataset adapter registry for persistence
            overflow_strategy: How to handle buffer overflow (drop_oldest, drop_newest, block)
            fallback_path: Path for fallback file persistence on errors
        """
        self.target_frame = target_frame
        self.target_dataset = target_dataset or "safety_events"
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_seconds
        self.overflow_strategy = overflow_strategy
        
        # Initialize event sink
        if event_sink is not None:
            self._sink = event_sink
        else:
            from namel3ss.codegen.backend.core.runtime.logic_adapters import AdapterRegistry
            
            registry = adapter_registry or AdapterRegistry()
            self._sink = SafetyEventSink(
                dataset_name=self.target_dataset,
                adapter_registry=registry,
                fallback_path=fallback_path or "/tmp/namel3ss_safety_fallback",
            )
        
        self._buffer: List[SafetyEvent] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._total_logged = 0
        self._total_flushed = 0
        self._total_dropped = 0
        
    async def start(self):
        """Start the background flush task."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._periodic_flush())
    
    async def stop(self):
        """Stop the background flush task and flush remaining events."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        
        # Flush any remaining events
        await self.flush()
    
    async def log(self, event: SafetyEvent) -> None:
        """Log a safety event (buffered).
        
        This is best-effort and will not raise exceptions that would
        disrupt the main execution path.
        
        Args:
            event: Safety event to log
        """
        try:
            async with self._lock:
                self._total_logged += 1
                
                # Handle buffer overflow
                if len(self._buffer) >= self.buffer_size:
                    if self.overflow_strategy == "drop_oldest":
                        self._buffer.pop(0)
                        self._total_dropped += 1
                        logger.warning(
                            "Safety event buffer overflow, dropping oldest event",
                            extra={"total_dropped": self._total_dropped}
                        )
                    elif self.overflow_strategy == "drop_newest":
                        self._total_dropped += 1
                        logger.warning(
                            "Safety event buffer overflow, dropping newest event",
                            extra={"total_dropped": self._total_dropped}
                        )
                        return  # Don't add the new event
                    elif self.overflow_strategy == "block":
                        # Force flush to make room
                        await self._flush_internal()
                
                self._buffer.append(event)
                
                # Flush if buffer is still full after handling overflow
                if len(self._buffer) >= self.buffer_size:
                    await self._flush_internal()
                    
        except Exception as e:
            # Never crash the main path due to logging errors
            logger.error(f"Failed to buffer safety event: {e}", exc_info=True)
    
    async def flush(self) -> None:
        """Explicitly flush buffered events."""
        async with self._lock:
            await self._flush_internal()
    
    async def _flush_internal(self) -> None:
        """Internal flush implementation (must be called with lock held)."""
        if not self._buffer:
            return
        
        events_to_flush = self._buffer[:]
        self._buffer.clear()
        
        try:
            await self._write_events(events_to_flush)
            self._total_flushed += len(events_to_flush)
        except Exception as e:
            logger.error(f"Failed to flush {len(events_to_flush)} safety events: {e}", exc_info=True)
    
    async def _write_events(self, events: List[SafetyEvent]) -> None:
        """Write events to the target dataset via SafetyEventSink.
        
        This integrates with the frames/datasets system for production-grade
        persistence. Events are written via dataset adapters with automatic
        retry logic and fallback file persistence on failures.
        
        Args:
            events: List of safety events to persist
        """
        try:
            # Use SafetyEventSink for production-grade persistence
            await self._sink.write_events(events)
            
        except Exception as e:
            # Log error but don't crash - SafetyEventSink has fallback handling
            logger.error(
                f"Failed to persist {len(events)} safety events via sink: {e}",
                exc_info=True,
                extra={
                    "event_count": len(events),
                    "first_event_id": events[0].event_id if events else None,
                    "dataset": self.target_dataset,
                }
            )
    
    async def _periodic_flush(self) -> None:
        """Background task to periodically flush events."""
        try:
            while True:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the logger's operation.
        
        Returns:
            Dictionary with statistics including total logged, flushed, dropped,
            current buffer size, and overflow strategy.
        """
        return {
            "total_logged": self._total_logged,
            "total_flushed": self._total_flushed,
            "total_dropped": self._total_dropped,
            "current_buffer_size": len(self._buffer),
            "max_buffer_size": self.buffer_size,
            "overflow_strategy": self.overflow_strategy,
            "flush_interval_seconds": self.flush_interval,
            "target_dataset": self.target_dataset,
        }


# Global logger instance
_global_logger: Optional[SafetyEventLogger] = None


def get_global_logger() -> SafetyEventLogger:
    """Get or create the global safety event logger.
    
    The global logger uses default configuration. For custom configuration,
    create a SafetyEventLogger instance directly and manage its lifecycle.
    
    Returns:
        Global SafetyEventLogger instance with default settings
    """
    global _global_logger
    if _global_logger is None:
        # Initialize with default configuration
        # In production, this should be configured via app config/DSL
        _global_logger = SafetyEventLogger(
            target_dataset="safety_events",
            buffer_size=100,
            flush_interval_seconds=10.0,
            overflow_strategy="drop_oldest",
        )
    return _global_logger


def set_global_logger(logger: SafetyEventLogger) -> None:
    """Set a custom global safety event logger.
    
    This allows applications to configure the global logger with
    custom settings (dataset, buffer size, adapter registry, etc.).
    
    Args:
        logger: Configured SafetyEventLogger instance
    """
    global _global_logger
    _global_logger = logger


async def log_safety_event(event: SafetyEvent) -> None:
    """Log a safety event using the global logger.
    
    This is a convenience function for logging safety events.
    It will not raise exceptions that would disrupt execution.
    
    Args:
        event: Safety event to log
    """
    try:
        logger_instance = get_global_logger()
        await logger_instance.log(event)
    except Exception as e:
        # Fail gracefully - never crash on logging
        logger.error(f"Failed to log safety event: {e}")


def create_safety_event_from_enforcement(
    direction: str,
    action: SafetyAction,
    violation: Optional[SafetyViolation],
    latency_ms: float,
    policy_name: Optional[str] = None,
    chain_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    trace_id: Optional[str] = None,
    **kwargs: Any,
) -> SafetyEvent:
    """Create a SafetyEvent from enforcement results.
    
    Helper function to construct events from enforce_input_policy or
    enforce_output_policy results.
    """
    categories = violation.categories if violation else []
    severity = violation.severity if violation else "none"
    confidence = violation.confidence if violation else 1.0
    reason = violation.reason if violation else None
    
    return SafetyEvent(
        direction=direction,
        action=action,
        categories=categories,
        severity=severity,
        confidence=confidence,
        reason=reason,
        latency_ms=latency_ms,
        policy_name=policy_name,
        chain_name=chain_name,
        agent_name=agent_name,
        trace_id=trace_id,
        **kwargs,
    )
