"""
Namel3ss Debugging & Observability System

Comprehensive execution tracing, replay, and debugging infrastructure for namel3ss applications.
Provides step-by-step execution visibility, deterministic replay capabilities, and CLI debug tooling.

Core Components:
- TraceEvent: Structured execution event data model
- Tracer: Runtime execution event capture and logging
- Replayer: Deterministic replay from trace files
- DebugCLI: Command-line debugging interface

Architecture Integration Points:
- AgentRuntime.aact() - Agent execution with turns, LLM calls, tool invocations
- execute_structured_prompt() - Prompt execution with validation  
- run_chain() - Chain execution with sequential steps
- CLI loading.py - Application and runtime module loading
- Tool execution - Both real and mock tool invocations

Event Types Captured:
- agent_execution_start/end
- agent_turn_start/end  
- llm_call_start/end
- tool_call_start/end
- prompt_execution_start/end
- chain_execution_start/end
- chain_step_start/end
- error_occurred
- memory_operation
- validation_event

Trace File Format:
JSON lines format with timestamped execution events, structured for replay and analysis.

CLI Debug Commands:
- namel3ss debug trace <app> [--output=trace.json] [--filter=agent|prompt|chain]
- namel3ss debug replay <trace-file> [--step] [--breakpoint=event_type:index]
- namel3ss debug analyze <trace-file> [--performance] [--errors] [--summary]
- namel3ss debug inspect <trace-file> [--event=123] [--agent=AgentName] [--chain=ChainName]

Configuration:
- Debug settings in namel3ss config
- Environment variable controls
- Runtime trace filtering and verbosity levels
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Literal, Set
from datetime import datetime
from pathlib import Path
from enum import Enum


class TraceEventType(str, Enum):
    """Types of execution events captured in traces."""
    
    # Application lifecycle
    APP_LOAD_START = "app_load_start"
    APP_LOAD_END = "app_load_end"
    
    # Agent execution events
    AGENT_EXECUTION_START = "agent_execution_start"
    AGENT_EXECUTION_END = "agent_execution_end"
    AGENT_TURN_START = "agent_turn_start"
    AGENT_TURN_END = "agent_turn_end"
    
    # LLM interaction events
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    
    # Tool execution events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    
    # Prompt execution events
    PROMPT_EXECUTION_START = "prompt_execution_start"
    PROMPT_EXECUTION_END = "prompt_execution_end"
    
    # Chain execution events
    CHAIN_EXECUTION_START = "chain_execution_start"
    CHAIN_EXECUTION_END = "chain_execution_end"
    CHAIN_STEP_START = "chain_step_start"
    CHAIN_STEP_END = "chain_step_end"
    
    # Memory and state events
    MEMORY_OPERATION = "memory_operation"
    MEMORY_SUMMARIZATION = "memory_summarization"
    
    # Validation and error events
    VALIDATION_START = "validation_start"
    VALIDATION_END = "validation_end"
    ERROR_OCCURRED = "error_occurred"
    
    # Performance events
    PERFORMANCE_MARKER = "performance_marker"


@dataclass
class TraceEvent:
    """
    Structured representation of a runtime execution event.
    
    Core event data model for tracing all namel3ss execution paths.
    Designed for serialization, replay, and analysis.
    """
    
    # Event identification
    event_type: TraceEventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    parent_event_id: Optional[str] = None
    
    # Execution context
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    component: Literal["agent", "prompt", "chain", "tool", "app", "llm"] = "app"
    component_name: Optional[str] = None
    
    # Event data payload
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution timing and state
    duration_ms: Optional[float] = None
    status: Literal["started", "completed", "failed", "skipped"] = "started"
    error: Optional[str] = None
    
    # Resource usage (for performance analysis)
    memory_usage_mb: Optional[float] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraceEvent:
        """Create event from dictionary."""
        # Convert string event_type back to enum
        if "event_type" in data:
            data["event_type"] = TraceEventType(data["event_type"])
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> TraceEvent:
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TraceExecutionContext:
    """
    Context information for a complete execution trace.
    
    Maintains execution hierarchy and relationships between events.
    """
    
    execution_id: str
    app_name: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Event hierarchy tracking
    root_event_id: Optional[str] = None
    current_event_stack: List[str] = field(default_factory=list)
    
    # Execution metadata
    namel3ss_version: Optional[str] = None
    python_version: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    
    # Performance totals
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tokens_used: int = 0
    total_cost_estimate: float = 0.0
    
    def get_duration_ms(self) -> Optional[float]:
        """Get total execution duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class TraceFilter:
    """
    Configuration for filtering trace events during capture or analysis.
    
    Allows selective tracing to reduce overhead and focus on specific components.
    """
    
    def __init__(
        self,
        *,
        components: Optional[Set[str]] = None,
        event_types: Optional[Set[TraceEventType]] = None,
        component_names: Optional[Set[str]] = None,
        min_duration_ms: Optional[float] = None,
        include_performance: bool = True,
        include_memory: bool = True,
        include_errors_only: bool = False,
    ):
        self.components = components or set()
        self.event_types = event_types or set()
        self.component_names = component_names or set()
        self.min_duration_ms = min_duration_ms
        self.include_performance = include_performance
        self.include_memory = include_memory
        self.include_errors_only = include_errors_only
    
    def should_capture(self, event: TraceEvent) -> bool:
        """Determine if an event should be captured based on filter criteria."""
        
        # Error-only filter
        if self.include_errors_only and event.status != "failed":
            return False
        
        # Component filter
        if self.components and event.component not in self.components:
            return False
        
        # Event type filter
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Component name filter
        if self.component_names and event.component_name not in self.component_names:
            return False
        
        # Duration filter (only applies to completed events)
        if (self.min_duration_ms is not None and 
            event.duration_ms is not None and 
            event.duration_ms < self.min_duration_ms):
            return False
        
        return True


@dataclass
class DebugConfiguration:
    """
    Debug system configuration settings.
    
    Controls tracing behavior, output locations, and performance impact.
    """
    
    # Output configuration
    enabled: bool = False
    trace_output_dir: Path = Path("./debug/traces")
    auto_trace_filename: bool = True  # Generate timestamp-based filenames
    
    # Tracing behavior
    buffer_events: bool = True  # Buffer events in memory before writing
    buffer_size: int = 1000
    flush_interval_seconds: float = 5.0
    
    # Filtering and performance
    trace_filter: Optional[TraceFilter] = None
    max_event_payload_size: int = 1024 * 16  # 16KB max per event payload
    capture_memory_usage: bool = True
    capture_performance_markers: bool = True
    
    # Integration settings
    trace_agent_execution: bool = True
    trace_prompt_execution: bool = True
    trace_chain_execution: bool = True
    trace_tool_calls: bool = True
    trace_llm_calls: bool = True
    trace_memory_operations: bool = False  # Can be verbose
    
    # Environment integration
    environment_variable_prefix: str = "NAMEL3SS_DEBUG"
    
    def get_trace_file_path(self, execution_id: str) -> Path:
        """Get the trace file path for a given execution."""
        if self.auto_trace_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trace_{timestamp}_{execution_id[:8]}.jsonl"
        else:
            filename = f"trace_{execution_id}.jsonl"
        
        return self.trace_output_dir / filename
    
    @classmethod
    def from_environment(cls) -> DebugConfiguration:
        """Create configuration from environment variables."""
        import os
        
        config = cls()
        
        # Check if debugging is enabled
        config.enabled = os.getenv("NAMEL3SS_DEBUG_ENABLED", "false").lower() in ("true", "1", "yes")
        
        # Output directory
        if output_dir := os.getenv("NAMEL3SS_DEBUG_OUTPUT_DIR"):
            config.trace_output_dir = Path(output_dir)
        
        # Component filtering
        if components := os.getenv("NAMEL3SS_DEBUG_COMPONENTS"):
            config.trace_filter = TraceFilter(
                components=set(components.split(","))
            )
        
        return config


def initialize_tracing(config: DebugConfiguration | None = None) -> None:
    """
    Initialize debugging and tracing for the namel3ss runtime.
    
    Args:
        config: Debug configuration to use. If None, loads from environment/workspace.
    """
    global _debug_config
    if config is None:
        from .config import get_debug_config_manager
        config_manager = get_debug_config_manager()
        config = config_manager.get_runtime_config()
    _debug_config = config

# Global debug configuration instance
_debug_config: DebugConfiguration | None = None

def get_debug_config() -> DebugConfiguration | None:
    """Get the current debug configuration."""
    return _debug_config

__all__ = [
    "TraceEventType",
    "TraceEvent", 
    "TraceExecutionContext",
    "TraceFilter",
    "DebugConfiguration",
    "initialize_tracing",
    "get_debug_config",
]