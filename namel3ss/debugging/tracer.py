"""
Runtime tracing infrastructure for capturing namel3ss execution events.

Provides the core Tracer class that integrates with namel3ss runtime components
to capture detailed execution traces for debugging and observability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import psutil
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
from datetime import datetime

from . import (
    TraceEvent,
    TraceEventType, 
    TraceExecutionContext,
    TraceFilter,
    DebugConfiguration,
)

logger = logging.getLogger(__name__)


class ExecutionTracer:
    """
    Core tracing engine that captures and persists execution events.
    
    Integrates with namel3ss runtime to provide comprehensive execution visibility:
    - Event capture with minimal performance overhead
    - Buffered writing to trace files  
    - Context management for hierarchical events
    - Integration hooks for all major execution paths
    """
    
    def __init__(self, config: DebugConfiguration):
        self.config = config
        self.current_context: Optional[TraceExecutionContext] = None
        self.event_buffer: List[TraceEvent] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Ensure output directory exists
        self.config.trace_output_dir.mkdir(parents=True, exist_ok=True)
    
    async def start_execution_trace(
        self, 
        *,
        app_name: Optional[str] = None,
        execution_id: Optional[str] = None
    ) -> TraceExecutionContext:
        """
        Initialize a new execution trace context.
        
        Args:
            app_name: Name of the application being traced
            execution_id: Unique identifier for this execution (auto-generated if None)
        
        Returns:
            TraceExecutionContext for this execution
        """
        if not self.config.enabled:
            # Return minimal context even when disabled
            return TraceExecutionContext(
                execution_id=execution_id or "disabled",
                app_name=app_name
            )
        
        context = TraceExecutionContext(
            execution_id=execution_id or f"exec_{int(time.time())}_{id(self):x}",
            app_name=app_name,
            namel3ss_version="0.1.0",  # TODO: Get from package
            python_version=f"{__import__('sys').version_info[:2]}"
        )
        
        self.current_context = context
        
        # Start background flush task if buffering is enabled
        if self.config.buffer_events and not self._flush_task:
            self._flush_task = asyncio.create_task(self._periodic_flush())
        
        # Emit application load start event
        await self.emit_event(
            TraceEventType.APP_LOAD_START,
            component="app",
            component_name=app_name,
            inputs={"execution_id": context.execution_id},
            metadata={
                "namel3ss_version": context.namel3ss_version,
                "python_version": context.python_version,
            }
        )
        
        return context
    
    async def end_execution_trace(self) -> Optional[Path]:
        """
        Finalize the current execution trace and write remaining events.
        
        Returns:
            Path to the written trace file, or None if tracing was disabled
        """
        if not self.config.enabled or not self.current_context:
            return None
        
        # Emit application load end event
        await self.emit_event(
            TraceEventType.APP_LOAD_END,
            component="app",
            inputs={},
            outputs={"execution_summary": self._get_execution_summary()},
            status="completed"
        )
        
        # Finalize context
        self.current_context.end_time = time.time()
        
        # Stop background flush task
        if self._flush_task:
            self._shutdown = True
            try:
                await asyncio.wait_for(self._flush_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._flush_task.cancel()
            self._flush_task = None
        
        # Final flush of all buffered events
        trace_file = await self._flush_events()
        
        logger.info(f"Execution trace written to: {trace_file}")
        return trace_file
    
    async def emit_event(
        self,
        event_type: TraceEventType,
        *,
        component: str = "app",
        component_name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
        status: str = "started",
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> TraceEvent:
        """
        Emit a trace event with the given details.
        
        Args:
            event_type: Type of event being emitted
            component: Component generating the event (agent, prompt, chain, etc.)
            component_name: Specific name of the component instance
            inputs: Input data for the event
            outputs: Output data for the event  
            metadata: Additional metadata
            parent_event_id: ID of parent event in execution hierarchy
            status: Event status (started, completed, failed, skipped)
            error: Error message if status is failed
            duration_ms: Event duration in milliseconds
        
        Returns:
            The created TraceEvent
        """
        if not self.config.enabled:
            # Return minimal event when disabled
            return TraceEvent(
                event_type=event_type,
                component=component,  # type: ignore
                status=status  # type: ignore
            )
        
        # Use current context or create minimal one
        execution_id = self.current_context.execution_id if self.current_context else "unknown"
        
        # Auto-detect parent from context stack
        if parent_event_id is None and self.current_context and self.current_context.current_event_stack:
            parent_event_id = self.current_context.current_event_stack[-1]
        
        # Capture memory usage if enabled
        memory_usage_mb = None
        if self.config.capture_memory_usage:
            try:
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except Exception:
                pass  # Ignore memory capture errors
        
        # Truncate large payloads to prevent trace file bloat
        inputs = self._truncate_payload(inputs or {})
        outputs = self._truncate_payload(outputs or {})
        metadata = self._truncate_payload(metadata or {})
        
        event = TraceEvent(
            event_type=event_type,
            execution_id=execution_id,
            component=component,  # type: ignore
            component_name=component_name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            parent_event_id=parent_event_id,
            status=status,  # type: ignore
            error=error,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
        )
        
        # Apply filtering
        if self.config.trace_filter and not self.config.trace_filter.should_capture(event):
            return event
        
        # Add to buffer or write immediately
        if self.config.buffer_events:
            self.event_buffer.append(event)
            
            # Force flush if buffer is full
            if len(self.event_buffer) >= self.config.buffer_size:
                await self._flush_events()
        else:
            await self._write_event(event)
        
        return event
    
    @asynccontextmanager
    async def trace_execution_context(
        self,
        event_type: TraceEventType,
        *,
        component: str,
        component_name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[TraceEvent]:
        """
        Context manager for tracing a complete execution block.
        
        Automatically emits start/end events and measures duration.
        Manages execution hierarchy by pushing/popping from context stack.
        
        Usage:
            async with tracer.trace_execution_context(
                TraceEventType.AGENT_EXECUTION_START,
                component="agent", 
                component_name="MyAgent"
            ) as start_event:
                # Agent execution code here
                result = await agent.run()
                return result
        """
        start_time = time.time()
        
        # Emit start event
        start_event = await self.emit_event(
            event_type,
            component=component,
            component_name=component_name,
            inputs=inputs,
            metadata=metadata,
            status="started"
        )
        
        # Push to context stack
        if self.current_context:
            self.current_context.current_event_stack.append(start_event.event_id)
        
        error_msg: Optional[str] = None
        outputs: Dict[str, Any] = {}
        
        try:
            yield start_event
        except Exception as e:
            error_msg = str(e)
            outputs["error_type"] = type(e).__name__
            raise
        finally:
            # Pop from context stack
            if self.current_context and self.current_context.current_event_stack:
                self.current_context.current_event_stack.pop()
            
            # Emit end event
            duration_ms = (time.time() - start_time) * 1000
            end_event_type = self._get_end_event_type(event_type)
            
            await self.emit_event(
                end_event_type,
                component=component,
                component_name=component_name,
                inputs={},
                outputs=outputs,
                metadata=metadata,
                parent_event_id=start_event.event_id,
                status="failed" if error_msg else "completed",
                error=error_msg,
                duration_ms=duration_ms
            )
    
    @contextmanager
    def trace_execution_context_sync(
        self,
        event_type: TraceEventType,
        *,
        component: str,
        component_name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[TraceEvent]:
        """
        Synchronous version of trace_execution_context.
        
        For use in synchronous code paths that can't use async context managers.
        """
        # Convert to async internally
        async def _async_trace():
            async with self.trace_execution_context(
                event_type, 
                component=component,
                component_name=component_name,
                inputs=inputs,
                metadata=metadata
            ) as event:
                return event
        
        # Run in event loop (simplified for sync usage)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, can't use sync version
                raise RuntimeError("Cannot use sync tracer in async context")
            event = loop.run_until_complete(_async_trace())
        except RuntimeError:
            # Create minimal event for sync usage
            event = TraceEvent(
                event_type=event_type,
                component=component,  # type: ignore
                component_name=component_name,
                inputs=inputs or {},
                metadata=metadata or {},
            )
        
        yield event
    
    async def _periodic_flush(self):
        """Background task that periodically flushes buffered events."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                if self.event_buffer:
                    await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    async def _flush_events(self) -> Optional[Path]:
        """Flush all buffered events to trace file."""
        if not self.event_buffer or not self.current_context:
            return None
        
        trace_file = self.config.get_trace_file_path(self.current_context.execution_id)
        
        try:
            with open(trace_file, "a") as f:
                for event in self.event_buffer:
                    f.write(json.dumps(event.to_dict()) + "\n")
            
            logger.debug(f"Flushed {len(self.event_buffer)} events to {trace_file}")
            self.event_buffer.clear()
            return trace_file
            
        except Exception as e:
            logger.error(f"Failed to flush events to {trace_file}: {e}")
            return None
    
    async def _write_event(self, event: TraceEvent):
        """Write a single event immediately to trace file."""
        if not self.current_context:
            return
        
        trace_file = self.config.get_trace_file_path(self.current_context.execution_id)
        
        try:
            with open(trace_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write event to {trace_file}: {e}")
    
    def _truncate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate large payloads to prevent trace file bloat."""
        payload_str = json.dumps(payload)
        if len(payload_str) <= self.config.max_event_payload_size:
            return payload
        
        # Truncate and add marker
        truncated_str = payload_str[:self.config.max_event_payload_size - 50]
        return {
            "_truncated": True,
            "_original_size": len(payload_str),
            "_data": truncated_str + "... [TRUNCATED]"
        }
    
    def _get_end_event_type(self, start_event_type: TraceEventType) -> TraceEventType:
        """Get the corresponding end event type for a start event type."""
        end_mapping = {
            TraceEventType.APP_LOAD_START: TraceEventType.APP_LOAD_END,
            TraceEventType.AGENT_EXECUTION_START: TraceEventType.AGENT_EXECUTION_END,
            TraceEventType.AGENT_TURN_START: TraceEventType.AGENT_TURN_END,
            TraceEventType.LLM_CALL_START: TraceEventType.LLM_CALL_END,
            TraceEventType.TOOL_CALL_START: TraceEventType.TOOL_CALL_END,
            TraceEventType.PROMPT_EXECUTION_START: TraceEventType.PROMPT_EXECUTION_END,
            TraceEventType.CHAIN_EXECUTION_START: TraceEventType.CHAIN_EXECUTION_END,
            TraceEventType.CHAIN_STEP_START: TraceEventType.CHAIN_STEP_END,
            TraceEventType.VALIDATION_START: TraceEventType.VALIDATION_END,
        }
        return end_mapping.get(start_event_type, start_event_type)
    
    def _get_execution_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current execution."""
        if not self.current_context:
            return {}
        
        return {
            "execution_id": self.current_context.execution_id,
            "duration_ms": self.current_context.get_duration_ms(),
            "total_events": len(self.event_buffer),
            "llm_calls": self.current_context.total_llm_calls,
            "tool_calls": self.current_context.total_tool_calls,
            "tokens_used": self.current_context.total_tokens_used,
            "cost_estimate": self.current_context.total_cost_estimate,
        }


# Global tracer instance
_global_tracer: Optional[ExecutionTracer] = None


def get_global_tracer() -> Optional[ExecutionTracer]:
    """Get the global tracer instance."""
    return _global_tracer


def set_global_tracer(tracer: ExecutionTracer):
    """Set the global tracer instance."""
    global _global_tracer
    _global_tracer = tracer


def initialize_tracing(config: Optional[DebugConfiguration] = None) -> ExecutionTracer:
    """
    Initialize global tracing with the given configuration.
    
    Args:
        config: Debug configuration (defaults to environment-based config)
    
    Returns:
        Initialized ExecutionTracer instance
    """
    if config is None:
        config = DebugConfiguration.from_environment()
    
    tracer = ExecutionTracer(config)
    set_global_tracer(tracer)
    
    return tracer


__all__ = [
    "ExecutionTracer",
    "get_global_tracer",
    "set_global_tracer", 
    "initialize_tracing",
]