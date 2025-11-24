"""
Debug-specific metrics and profiling for namel3ss debugging system.

Extends the core observability.metrics with debugging-focused measurements
including execution timing, memory usage, and debug event statistics.
"""

from __future__ import annotations

import time
import threading
import psutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from contextlib import contextmanager

from namel3ss.observability.metrics import emit_metric, record_metric
from namel3ss.debugging import TraceEventType


@dataclass
class DebugMetrics:
    """
    Container for debug-specific metrics and statistics.
    
    Tracks execution performance, resource usage, and debug system overhead.
    """
    
    # Execution metrics
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0
    
    # Event metrics
    total_events_captured: int = 0
    events_by_type: Dict[TraceEventType, int] = field(default_factory=lambda: defaultdict(int))
    events_by_component: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Memory metrics
    peak_memory_usage_mb: float = 0.0
    avg_memory_usage_mb: float = 0.0
    memory_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    # Tracing overhead metrics
    trace_write_time_ms: float = 0.0
    trace_file_size_bytes: int = 0
    trace_buffer_flushes: int = 0
    
    # Error metrics
    trace_errors: int = 0
    validation_failures: int = 0
    llm_failures: int = 0
    tool_failures: int = 0
    
    def update_execution_timing(self, duration_ms: float):
        """Update execution timing statistics."""
        self.execution_count += 1
        self.total_execution_time_ms += duration_ms
        self.avg_execution_time_ms = self.total_execution_time_ms / self.execution_count
        self.max_execution_time_ms = max(self.max_execution_time_ms, duration_ms)
    
    def record_event(self, event_type: TraceEventType, component: str):
        """Record a trace event for statistics."""
        self.total_events_captured += 1
        self.events_by_type[event_type] += 1
        self.events_by_component[component] += 1
    
    def update_memory_usage(self, memory_mb: float):
        """Update memory usage statistics."""
        self.memory_samples.append(memory_mb)
        self.peak_memory_usage_mb = max(self.peak_memory_usage_mb, memory_mb)
        if self.memory_samples:
            self.avg_memory_usage_mb = sum(self.memory_samples) / len(self.memory_samples)
    
    def record_trace_operation(self, operation_time_ms: float, file_size_bytes: int = 0):
        """Record trace file operation metrics."""
        self.trace_write_time_ms += operation_time_ms
        self.trace_file_size_bytes += file_size_bytes
        if operation_time_ms > 0:  # Only count actual write operations
            self.trace_buffer_flushes += 1
    
    def record_error(self, error_type: str):
        """Record an error by type."""
        if error_type == "validation":
            self.validation_failures += 1
        elif error_type == "llm":
            self.llm_failures += 1
        elif error_type == "tool":
            self.tool_failures += 1
        else:
            self.trace_errors += 1
    
    def to_dict(self) -> Dict[str, any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "execution": {
                "count": self.execution_count,
                "total_time_ms": self.total_execution_time_ms,
                "avg_time_ms": self.avg_execution_time_ms,
                "max_time_ms": self.max_execution_time_ms,
            },
            "events": {
                "total_captured": self.total_events_captured,
                "by_type": {str(k): v for k, v in self.events_by_type.items()},
                "by_component": dict(self.events_by_component),
            },
            "memory": {
                "peak_mb": self.peak_memory_usage_mb,
                "avg_mb": self.avg_memory_usage_mb,
                "samples_count": len(self.memory_samples),
            },
            "tracing_overhead": {
                "write_time_ms": self.trace_write_time_ms,
                "file_size_bytes": self.trace_file_size_bytes,
                "buffer_flushes": self.trace_buffer_flushes,
            },
            "errors": {
                "trace_errors": self.trace_errors,
                "validation_failures": self.validation_failures,
                "llm_failures": self.llm_failures,
                "tool_failures": self.tool_failures,
            },
        }


class DebugProfiler:
    """
    Profiler for debugging system performance and resource usage.
    
    Tracks execution timing, memory usage, and system overhead
    with minimal impact on traced execution.
    """
    
    def __init__(self):
        self.metrics = DebugMetrics()
        self._lock = threading.RLock()
        self._start_times: Dict[str, float] = {}
    
    @contextmanager
    def profile_execution(self, execution_id: str):
        """
        Context manager for profiling a complete execution.
        
        Usage:
            with profiler.profile_execution("exec_123"):
                # Execution code here
                pass
        """
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            memory_after = self._get_memory_usage()
            
            duration_ms = (end_time - start_time) * 1000
            
            with self._lock:
                self.metrics.update_execution_timing(duration_ms)
                if memory_after:
                    self.metrics.update_memory_usage(memory_after)
            
            # Emit metrics to the global system
            emit_metric("debug.execution.duration", {"value": duration_ms}, {"execution_id": execution_id})
            emit_metric("debug.execution.memory", {"value": memory_after or 0}, {"execution_id": execution_id})
    
    @contextmanager
    def profile_trace_operation(self, operation: str):
        """
        Context manager for profiling trace operations (file writes, etc).
        
        Usage:
            with profiler.profile_trace_operation("file_write"):
                # Trace operation code here
                pass
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            with self._lock:
                self.metrics.record_trace_operation(duration_ms)
            
            # Emit overhead metrics
            emit_metric("debug.tracing.overhead", {"value": duration_ms}, {"operation": operation})
    
    def record_event_capture(self, event_type: TraceEventType, component: str, payload_size: int = 0):
        """Record statistics for a captured trace event."""
        with self._lock:
            self.metrics.record_event(event_type, component)
        
        # Emit event metrics
        emit_metric("debug.events.captured", {"value": 1}, {
            "event_type": event_type.value,
            "component": component,
            "payload_size_kb": str(payload_size // 1024),
        })
    
    def record_trace_file_write(self, file_size_bytes: int, event_count: int):
        """Record trace file write statistics."""
        with self._lock:
            self.metrics.record_trace_operation(0, file_size_bytes)  # Duration tracked separately
        
        emit_metric("debug.trace_file.write", {
            "size_bytes": file_size_bytes,
            "event_count": event_count,
            "size_kb": file_size_bytes / 1024,
        })
    
    def record_debug_error(self, error_type: str, component: str, details: Optional[str] = None):
        """Record debug system errors."""
        with self._lock:
            self.metrics.record_error(error_type)
        
        emit_metric("debug.errors", {"value": 1}, {
            "error_type": error_type,
            "component": component,
            "has_details": "true" if details else "false",
        })
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Get a summary of performance metrics."""
        with self._lock:
            return self.metrics.to_dict()
    
    def get_overhead_percentage(self) -> float:
        """
        Calculate debugging overhead as percentage of total execution time.
        
        Returns:
            Overhead percentage (0.0-100.0)
        """
        with self._lock:
            if self.metrics.total_execution_time_ms == 0:
                return 0.0
            
            return (self.metrics.trace_write_time_ms / self.metrics.total_execution_time_ms) * 100
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self._lock:
            self.metrics = DebugMetrics()
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return None


# Global debug profiler instance
_debug_profiler: Optional[DebugProfiler] = None
_profiler_lock = threading.RLock()


def get_debug_profiler() -> DebugProfiler:
    """Get the global debug profiler instance."""
    global _debug_profiler
    
    with _profiler_lock:
        if _debug_profiler is None:
            _debug_profiler = DebugProfiler()
        return _debug_profiler


def reset_debug_profiler():
    """Reset the global debug profiler."""
    global _debug_profiler
    
    with _profiler_lock:
        if _debug_profiler is not None:
            _debug_profiler.reset_metrics()


def record_debug_metric(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """
    Record a debug-specific metric.
    
    Convenience function for recording debug metrics with automatic prefixing.
    
    Args:
        metric_name: Name of the metric (will be prefixed with 'debug.')
        value: Metric value
        labels: Optional labels/tags
    """
    full_name = f"debug.{metric_name}" if not metric_name.startswith("debug.") else metric_name
    record_metric(full_name, value, labels)


def profile_debug_execution(execution_id: str):
    """
    Context manager for profiling debug execution.
    
    Usage:
        with profile_debug_execution("exec_123"):
            # Execution code
            pass
    """
    return get_debug_profiler().profile_execution(execution_id)


def profile_trace_operation(operation: str):
    """
    Context manager for profiling trace operations.
    
    Usage:
        with profile_trace_operation("file_write"):
            # Trace operation
            pass
    """
    return get_debug_profiler().profile_trace_operation(operation)


class MemoryTracker:
    """
    Lightweight memory usage tracker for debugging.
    
    Provides periodic memory sampling and peak detection
    with minimal overhead.
    """
    
    def __init__(self, sample_interval_ms: int = 1000):
        self.sample_interval_ms = sample_interval_ms
        self.samples: Deque[tuple[float, float]] = deque(maxlen=1000)  # (timestamp, memory_mb)
        self.peak_memory = 0.0
        self.baseline_memory = 0.0
        self._lock = threading.RLock()
        
    def start_tracking(self):
        """Start memory tracking with baseline measurement."""
        current_memory = self._get_memory_usage()
        if current_memory:
            self.baseline_memory = current_memory
            self.peak_memory = current_memory
            self._record_sample(current_memory)
    
    def record_checkpoint(self, checkpoint_name: str):
        """Record a memory checkpoint with name."""
        current_memory = self._get_memory_usage()
        if current_memory:
            self._record_sample(current_memory)
            
            # Emit checkpoint metric
            emit_metric("debug.memory.checkpoint", {"value": current_memory}, {
                "checkpoint": checkpoint_name,
                "delta_from_baseline": str(current_memory - self.baseline_memory),
            })
    
    def get_memory_report(self) -> Dict[str, float]:
        """Get memory usage report."""
        with self._lock:
            current_memory = self._get_memory_usage() or 0.0
            
            return {
                "current_mb": current_memory,
                "peak_mb": self.peak_memory,
                "baseline_mb": self.baseline_memory,
                "delta_from_baseline_mb": current_memory - self.baseline_memory,
                "peak_delta_mb": self.peak_memory - self.baseline_memory,
                "sample_count": len(self.samples),
            }
    
    def _record_sample(self, memory_mb: float):
        """Record a memory sample."""
        with self._lock:
            timestamp = time.time()
            self.samples.append((timestamp, memory_mb))
            self.peak_memory = max(self.peak_memory, memory_mb)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return None


__all__ = [
    "DebugMetrics",
    "DebugProfiler",
    "MemoryTracker",
    "get_debug_profiler",
    "reset_debug_profiler",
    "record_debug_metric",
    "profile_debug_execution",
    "profile_trace_operation",
]