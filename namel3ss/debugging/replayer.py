"""
Deterministic replay system for namel3ss execution traces.

Provides the ability to replay previous executions from trace files for debugging,
testing, and analysis purposes. Supports step-by-step replay, breakpoints, and
mock response injection for deterministic debugging.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union, Callable
from collections import defaultdict

from . import (
    TraceEvent,
    TraceEventType,
    TraceExecutionContext,
    TraceFilter,
)

logger = logging.getLogger(__name__)


@dataclass
class ReplayBreakpoint:
    """
    Configuration for breakpoints during trace replay.
    
    Allows pausing execution at specific events for debugging.
    """
    
    event_type: Optional[TraceEventType] = None
    event_index: Optional[int] = None  # Stop at Nth occurrence
    component: Optional[str] = None
    component_name: Optional[str] = None
    condition: Optional[Callable[[TraceEvent], bool]] = None
    
    def matches(self, event: TraceEvent, event_count: int) -> bool:
        """Check if this breakpoint matches the given event."""
        
        if self.event_type and event.event_type != self.event_type:
            return False
        
        if self.event_index and event_count != self.event_index:
            return False
        
        if self.component and event.component != self.component:
            return False
        
        if self.component_name and event.component_name != self.component_name:
            return False
        
        if self.condition and not self.condition(event):
            return False
        
        return True


@dataclass
class ReplayState:
    """
    Current state of trace replay execution.
    
    Tracks progress, context, and debugging information.
    """
    
    # Replay progress
    current_event_index: int = 0
    total_events: int = 0
    paused: bool = False
    completed: bool = False
    
    # Current event context
    current_event: Optional[TraceEvent] = None
    event_stack: List[TraceEvent] = field(default_factory=list)
    
    # Execution context
    execution_context: Optional[TraceExecutionContext] = None
    
    # Mock responses for deterministic replay
    mock_responses: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    event_type_counts: Dict[TraceEventType, int] = field(default_factory=lambda: defaultdict(int))
    component_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_events: List[TraceEvent] = field(default_factory=list)
    
    def get_progress_percentage(self) -> float:
        """Get replay progress as percentage."""
        if self.total_events == 0:
            return 0.0
        return (self.current_event_index / self.total_events) * 100


class ExecutionReplayer:
    """
    Replay engine for namel3ss execution traces.
    
    Provides deterministic replay of previous executions with debugging capabilities:
    - Step-by-step execution control
    - Breakpoint support
    - Mock response injection
    - Execution analysis and statistics
    """
    
    def __init__(
        self,
        trace_file: Path,
        *,
        filter: Optional[TraceFilter] = None,
        breakpoints: Optional[List[ReplayBreakpoint]] = None,
        mock_responses: Optional[Dict[str, Any]] = None,
    ):
        self.trace_file = trace_file
        self.filter = filter
        self.breakpoints = breakpoints or []
        self.mock_responses = mock_responses or {}
        
        # Load and parse trace
        self.events = self._load_trace_events()
        self.state = ReplayState(
            total_events=len(self.events),
            mock_responses=self.mock_responses.copy()
        )
        
        # Extract execution context from trace
        self.state.execution_context = self._extract_execution_context()
        
        logger.info(f"Loaded trace with {len(self.events)} events from {trace_file}")
    
    def replay_full(self) -> ReplayState:
        """
        Replay the complete trace without interruption.
        
        Returns:
            Final replay state with statistics and results
        """
        logger.info("Starting full trace replay")
        
        for event in self.events:
            self._replay_event(event)
            self.state.current_event_index += 1
        
        self.state.completed = True
        
        logger.info(f"Completed replay of {len(self.events)} events")
        return self.state
    
    def replay_step(self) -> Optional[TraceEvent]:
        """
        Replay the next event in the trace.
        
        Returns:
            Next event that was replayed, or None if replay is complete
        """
        if self.state.completed or self.state.current_event_index >= len(self.events):
            self.state.completed = True
            return None
        
        event = self.events[self.state.current_event_index]
        self._replay_event(event)
        self.state.current_event_index += 1
        
        # Check for breakpoints
        if self._check_breakpoints(event, self.state.current_event_index):
            self.state.paused = True
            logger.info(f"Hit breakpoint at event {self.state.current_event_index}: {event.event_type}")
        
        return event
    
    def replay_until_breakpoint(self) -> Optional[TraceEvent]:
        """
        Replay events until hitting a breakpoint or completion.
        
        Returns:
            Event where breakpoint was hit, or None if replay completed
        """
        self.state.paused = False
        
        while not self.state.completed and not self.state.paused:
            event = self.replay_step()
            if not event:
                break
        
        return self.state.current_event
    
    def replay_until_event_type(self, event_type: TraceEventType) -> Optional[TraceEvent]:
        """
        Replay events until hitting the specified event type.
        
        Args:
            event_type: Event type to stop at
        
        Returns:
            Event of specified type, or None if not found
        """
        while not self.state.completed:
            event = self.replay_step()
            if not event:
                break
            
            if event.event_type == event_type:
                return event
        
        return None
    
    def seek_to_event(self, event_index: int) -> Optional[TraceEvent]:
        """
        Seek to a specific event index in the trace.
        
        Args:
            event_index: Index of event to seek to
        
        Returns:
            Event at the specified index, or None if invalid index
        """
        if event_index < 0 or event_index >= len(self.events):
            return None
        
        self.state.current_event_index = event_index
        self.state.current_event = self.events[event_index]
        self.state.completed = False
        self.state.paused = True
        
        return self.state.current_event
    
    def get_events_by_component(self, component: str) -> List[TraceEvent]:
        """Get all events for a specific component."""
        return [event for event in self.events if event.component == component]
    
    def get_events_by_type(self, event_type: TraceEventType) -> List[TraceEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_error_events(self) -> List[TraceEvent]:
        """Get all events that represent errors."""
        return [event for event in self.events if event.status == "failed" or event.error]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of the traced execution.
        
        Returns:
            Dictionary with execution statistics and analysis
        """
        # Count events by type and component
        type_counts = defaultdict(int)
        component_counts = defaultdict(int)
        error_count = 0
        total_duration = 0.0
        total_tokens = 0
        total_cost = 0.0
        
        for event in self.events:
            type_counts[event.event_type] += 1
            component_counts[event.component] += 1
            
            if event.status == "failed" or event.error:
                error_count += 1
            
            if event.duration_ms:
                total_duration += event.duration_ms
            
            if event.tokens_used:
                total_tokens += event.tokens_used
            
            if event.cost_estimate:
                total_cost += event.cost_estimate
        
        # Find execution span
        start_time = min(event.timestamp for event in self.events) if self.events else 0
        end_time = max(event.timestamp for event in self.events) if self.events else 0
        
        return {
            "execution_overview": {
                "total_events": len(self.events),
                "execution_duration_seconds": end_time - start_time,
                "total_processing_duration_ms": total_duration,
                "error_count": error_count,
                "success_rate": (len(self.events) - error_count) / len(self.events) if self.events else 0,
            },
            "resource_usage": {
                "total_tokens": total_tokens,
                "estimated_cost": total_cost,
                "avg_memory_usage_mb": self._calculate_avg_memory_usage(),
            },
            "component_breakdown": dict(component_counts),
            "event_type_breakdown": {str(k): v for k, v in type_counts.items()},
            "execution_context": {
                "execution_id": self.state.execution_context.execution_id if self.state.execution_context else "unknown",
                "app_name": self.state.execution_context.app_name if self.state.execution_context else None,
            },
            "trace_file": str(self.trace_file),
        }
    
    def inject_mock_response(self, component_name: str, response: Any):
        """
        Inject a mock response for a specific component during replay.
        
        Args:
            component_name: Name of component to mock
            response: Mock response data
        """
        self.state.mock_responses[component_name] = response
    
    def add_breakpoint(self, breakpoint: ReplayBreakpoint):
        """Add a breakpoint to the replay session."""
        self.breakpoints.append(breakpoint)
    
    def remove_breakpoint(self, breakpoint: ReplayBreakpoint):
        """Remove a breakpoint from the replay session."""
        if breakpoint in self.breakpoints:
            self.breakpoints.remove(breakpoint)
    
    def _load_trace_events(self) -> List[TraceEvent]:
        """Load and parse events from the trace file."""
        events = []
        
        try:
            with open(self.trace_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event_data = json.loads(line)
                        event = TraceEvent.from_dict(event_data)
                        
                        # Apply filter if specified
                        if not self.filter or self.filter.should_capture(event):
                            events.append(event)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {self.trace_file}: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to create event from line {line_num}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load trace file {self.trace_file}: {e}")
            raise
        
        return events
    
    def _extract_execution_context(self) -> Optional[TraceExecutionContext]:
        """Extract execution context from loaded events."""
        if not self.events:
            return None
        
        # Find app load events to extract context
        app_start_events = [e for e in self.events if e.event_type == TraceEventType.APP_LOAD_START]
        
        if app_start_events:
            start_event = app_start_events[0]
            return TraceExecutionContext(
                execution_id=start_event.execution_id,
                app_name=start_event.component_name,
                start_time=start_event.timestamp,
                namel3ss_version=start_event.metadata.get("namel3ss_version"),
                python_version=start_event.metadata.get("python_version"),
            )
        
        # Fallback: create context from first event
        first_event = self.events[0]
        return TraceExecutionContext(
            execution_id=first_event.execution_id,
            start_time=first_event.timestamp,
        )
    
    def _replay_event(self, event: TraceEvent):
        """Process a single event during replay."""
        self.state.current_event = event
        
        # Update statistics
        self.state.event_type_counts[event.event_type] += 1
        self.state.component_counts[event.component] += 1
        
        if event.status == "failed" or event.error:
            self.state.error_events.append(event)
        
        # Manage event stack for hierarchical events
        if event.event_type.name.endswith("_START"):
            self.state.event_stack.append(event)
        elif event.event_type.name.endswith("_END") and self.state.event_stack:
            self.state.event_stack.pop()
        
        # Apply mock responses if configured
        self._apply_mock_response(event)
        
        logger.debug(f"Replayed event: {event.event_type} - {event.component}/{event.component_name}")
    
    def _apply_mock_response(self, event: TraceEvent):
        """Apply mock responses for deterministic replay."""
        if not self.state.mock_responses:
            return
        
        # Check for component-specific mock
        component_key = f"{event.component}/{event.component_name}"
        if component_key in self.state.mock_responses:
            mock_response = self.state.mock_responses[component_key]
            event.outputs = {"mock_response": mock_response, "original_outputs": event.outputs}
            logger.debug(f"Applied mock response to {component_key}")
    
    def _check_breakpoints(self, event: TraceEvent, event_count: int) -> bool:
        """Check if any breakpoints match the current event."""
        for breakpoint in self.breakpoints:
            if breakpoint.matches(event, event_count):
                return True
        return False
    
    def _calculate_avg_memory_usage(self) -> Optional[float]:
        """Calculate average memory usage across all events."""
        memory_events = [e for e in self.events if e.memory_usage_mb is not None]
        if not memory_events:
            return None
        
        total_memory = sum(e.memory_usage_mb for e in memory_events)  # type: ignore
        return total_memory / len(memory_events)


class TraceAnalyzer:
    """
    Advanced analysis utilities for execution traces.
    
    Provides detailed insights into execution patterns, performance bottlenecks,
    and debugging information.
    """
    
    def __init__(self, trace_file: Path):
        self.trace_file = trace_file
        self.replayer = ExecutionReplayer(trace_file)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze performance characteristics of the execution.
        
        Returns:
            Performance analysis report
        """
        events = self.replayer.events
        
        # Find performance bottlenecks
        slow_events = []
        llm_events = []
        tool_events = []
        
        for event in events:
            if event.duration_ms and event.duration_ms > 1000:  # Events > 1 second
                slow_events.append(event)
            
            if event.component == "llm":
                llm_events.append(event)
            
            if event.component == "tool":
                tool_events.append(event)
        
        # Calculate timing statistics
        llm_times = [e.duration_ms for e in llm_events if e.duration_ms]
        tool_times = [e.duration_ms for e in tool_events if e.duration_ms]
        
        return {
            "slow_operations": [
                {
                    "event_type": e.event_type,
                    "component": e.component,
                    "component_name": e.component_name,
                    "duration_ms": e.duration_ms,
                }
                for e in slow_events
            ],
            "llm_performance": {
                "total_calls": len(llm_events),
                "avg_duration_ms": sum(llm_times) / len(llm_times) if llm_times else 0,
                "max_duration_ms": max(llm_times) if llm_times else 0,
                "total_tokens": sum(e.tokens_used or 0 for e in llm_events),
            },
            "tool_performance": {
                "total_calls": len(tool_events),
                "avg_duration_ms": sum(tool_times) / len(tool_times) if tool_times else 0,
                "max_duration_ms": max(tool_times) if tool_times else 0,
            },
        }
    
    def analyze_errors(self) -> Dict[str, Any]:
        """
        Analyze error patterns in the execution.
        
        Returns:
            Error analysis report
        """
        error_events = self.replayer.get_error_events()
        
        # Categorize errors
        error_categories = defaultdict(list)
        for event in error_events:
            error_type = "unknown"
            if event.error:
                # Try to extract error type from error message
                if "ValidationError" in event.error:
                    error_type = "validation"
                elif "LLMError" in event.error:
                    error_type = "llm"
                elif "ToolError" in event.error:
                    error_type = "tool"
                else:
                    error_type = "runtime"
            
            error_categories[error_type].append(event)
        
        return {
            "total_errors": len(error_events),
            "error_categories": {
                category: {
                    "count": len(events),
                    "examples": [
                        {
                            "event_type": e.event_type,
                            "component": e.component,
                            "component_name": e.component_name,
                            "error": e.error,
                            "timestamp": e.timestamp,
                        }
                        for e in events[:3]  # Show first 3 examples
                    ]
                }
                for category, events in error_categories.items()
            }
        }
    
    def find_execution_path(self, component_name: str) -> List[TraceEvent]:
        """
        Find the execution path for a specific component.
        
        Args:
            component_name: Name of component to trace
        
        Returns:
            List of events forming the execution path
        """
        return [
            event for event in self.replayer.events
            if event.component_name == component_name
        ]


__all__ = [
    "ReplayBreakpoint",
    "ReplayState", 
    "ExecutionReplayer",
    "TraceAnalyzer",
]