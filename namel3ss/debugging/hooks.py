"""
Runtime integration hooks for namel3ss debugging system.

Provides decorators and context managers to integrate execution tracing 
into existing namel3ss runtime components with minimal code changes.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Dict, Optional, Callable, TypeVar, ParamSpec
from contextlib import asynccontextmanager

from . import TraceEventType
from .tracer import get_global_tracer, ExecutionTracer

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def trace_agent_execution(
    agent_name: Optional[str] = None,
    *,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
    capture_metadata: bool = True,
):
    """
    Decorator to trace agent execution methods.
    
    Usage:
        @trace_agent_execution("MyAgent")
        async def aact(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
            # Agent execution code
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_global_tracer()
            if not tracer or not tracer.config.trace_agent_execution:
                return await func(*args, **kwargs)
            
            # Extract agent name from self if not provided
            component_name = agent_name
            if not component_name and args:
                agent_instance = args[0]
                if hasattr(agent_instance, 'agent_def') and hasattr(agent_instance.agent_def, 'name'):
                    component_name = agent_instance.agent_def.name
                elif hasattr(agent_instance, 'name'):
                    component_name = agent_instance.name
                else:
                    component_name = type(agent_instance).__name__
            
            # Prepare inputs
            inputs = {}
            if capture_inputs and args:
                inputs = {
                    "user_input": args[1] if len(args) > 1 else None,
                    "context": kwargs.get("context"),
                    "max_turns": kwargs.get("max_turns"),
                }
            
            async with tracer.trace_execution_context(
                TraceEventType.AGENT_EXECUTION_START,
                component="agent",
                component_name=component_name,
                inputs=inputs if capture_inputs else {},
                metadata={"function": func.__name__} if capture_metadata else {}
            ):
                result = await func(*args, **kwargs)
                
                # Capture outputs
                if capture_outputs:
                    await tracer.emit_event(
                        TraceEventType.AGENT_EXECUTION_END,
                        component="agent", 
                        component_name=component_name,
                        outputs={
                            "status": getattr(result, "status", None),
                            "final_response": getattr(result, "final_response", None),
                            "total_turns": len(getattr(result, "turns", [])),
                        } if hasattr(result, "status") else {"result": str(result)},
                        status="completed"
                    )
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # For sync functions, try to trace if possible
            tracer = get_global_tracer()
            if not tracer:
                return func(*args, **kwargs)
            
            # Use sync context manager for limited tracing
            component_name = agent_name or "UnknownAgent"
            
            with tracer.trace_execution_context_sync(
                TraceEventType.AGENT_EXECUTION_START,
                component="agent",
                component_name=component_name,
                inputs={} if not capture_inputs else {"sync_call": True},
            ):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


def trace_prompt_execution(
    prompt_name: Optional[str] = None,
    *,
    capture_inputs: bool = True,
    capture_outputs: bool = True,
):
    """
    Decorator to trace prompt execution methods.
    
    Usage:
        @trace_prompt_execution("MyPrompt")
        async def execute_structured_prompt(prompt_def, llm, args):
            # Prompt execution code
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_global_tracer()
            if not tracer or not tracer.config.trace_prompt_execution:
                return await func(*args, **kwargs)
            
            # Extract prompt name from arguments
            component_name = prompt_name
            if not component_name and args:
                prompt_def = args[0] if args else kwargs.get("prompt_def")
                if prompt_def and hasattr(prompt_def, "name"):
                    component_name = prompt_def.name
                else:
                    component_name = "UnknownPrompt"
            
            # Prepare inputs
            inputs = {}
            if capture_inputs:
                inputs = {
                    "args": args[2] if len(args) > 2 else kwargs.get("args", {}),
                    "llm_provider": str(args[1]) if len(args) > 1 else str(kwargs.get("llm", "")),
                }
            
            async with tracer.trace_execution_context(
                TraceEventType.PROMPT_EXECUTION_START,
                component="prompt",
                component_name=component_name,
                inputs=inputs if capture_inputs else {}
            ):
                result = await func(*args, **kwargs)
                
                # Capture outputs
                if capture_outputs:
                    await tracer.emit_event(
                        TraceEventType.PROMPT_EXECUTION_END,
                        component="prompt",
                        component_name=component_name,
                        outputs={
                            "output": getattr(result, "output", None),
                            "latency_ms": getattr(result, "latency_ms", None),
                            "tokens_used": getattr(result, "prompt_tokens", 0) + getattr(result, "completion_tokens", 0),
                            "model": getattr(result, "model", None),
                            "provider": getattr(result, "provider", None),
                        } if hasattr(result, "output") else {"result": str(result)},
                        status="completed"
                    )
                
                return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return func  # type: ignore
    
    return decorator


def trace_chain_execution(
    chain_name: Optional[str] = None,
    *,
    capture_steps: bool = True,
):
    """
    Decorator to trace chain execution methods.
    
    Usage:
        @trace_chain_execution("MyChain") 
        def run_chain(name: str, payload: Dict[str, Any]):
            # Chain execution code
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_global_tracer()
            if not tracer or not tracer.config.trace_chain_execution:
                return await func(*args, **kwargs)
            
            # Extract chain name from arguments
            component_name = chain_name
            if not component_name and args:
                component_name = args[0] if isinstance(args[0], str) else kwargs.get("name", "UnknownChain")
            
            inputs = {
                "payload": args[1] if len(args) > 1 else kwargs.get("payload", {}),
            }
            
            async with tracer.trace_execution_context(
                TraceEventType.CHAIN_EXECUTION_START,
                component="chain",
                component_name=component_name,
                inputs=inputs
            ):
                result = await func(*args, **kwargs)
                
                # Capture chain result
                await tracer.emit_event(
                    TraceEventType.CHAIN_EXECUTION_END,
                    component="chain",
                    component_name=component_name,
                    outputs={
                        "status": result.get("status") if isinstance(result, dict) else "unknown",
                        "steps_executed": len(result.get("steps", [])) if isinstance(result, dict) else 0,
                        "result": result.get("result") if isinstance(result, dict) else str(result),
                    },
                    status="completed"
                )
                
                return result
        
        @functools.wraps(func) 
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_global_tracer()
            if not tracer or not tracer.config.trace_chain_execution:
                return func(*args, **kwargs)
            
            # Extract chain name from arguments
            component_name = chain_name
            if not component_name and args:
                component_name = args[0] if isinstance(args[0], str) else kwargs.get("name", "UnknownChain")
            
            # Simple sync tracing (limited functionality)
            logger.debug(f"Tracing chain execution: {component_name}")
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


def trace_llm_call(
    provider_name: Optional[str] = None,
):
    """
    Decorator to trace LLM API calls.
    
    Usage:
        @trace_llm_call("openai")
        async def generate(self, prompt: str, **kwargs):
            # LLM call code
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_global_tracer()
            if not tracer or not tracer.config.trace_llm_calls:
                return await func(*args, **kwargs)
            
            # Extract provider name
            component_name = provider_name
            if not component_name and args:
                llm_instance = args[0]
                component_name = getattr(llm_instance, "provider_name", type(llm_instance).__name__)
            
            inputs = {
                "prompt_length": len(str(args[1])) if len(args) > 1 else 0,
                "model": kwargs.get("model"),
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
            }
            
            async with tracer.trace_execution_context(
                TraceEventType.LLM_CALL_START,
                component="llm",
                component_name=component_name,
                inputs=inputs
            ):
                result = await func(*args, **kwargs)
                
                # Capture LLM response
                await tracer.emit_event(
                    TraceEventType.LLM_CALL_END,
                    component="llm",
                    component_name=component_name,
                    outputs={
                        "response_length": len(getattr(result, "text", "")),
                        "prompt_tokens": getattr(result, "prompt_tokens", 0),
                        "completion_tokens": getattr(result, "completion_tokens", 0),
                        "model": getattr(result, "model", ""),
                        "finish_reason": getattr(result, "finish_reason", ""),
                    },
                    tokens_used=getattr(result, "prompt_tokens", 0) + getattr(result, "completion_tokens", 0),
                    status="completed"
                )
                
                return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return func  # type: ignore
    
    return decorator


def trace_tool_call(
    tool_name: Optional[str] = None,
):
    """
    Decorator to trace tool invocations.
    
    Usage:
        @trace_tool_call("search_tool")
        async def execute(self, **inputs):
            # Tool execution code
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_global_tracer()
            if not tracer or not tracer.config.trace_tool_calls:
                return await func(*args, **kwargs)
            
            # Extract tool name
            component_name = tool_name
            if not component_name and args:
                tool_instance = args[0]
                component_name = getattr(tool_instance, "name", type(tool_instance).__name__)
            
            inputs = {
                "inputs": kwargs,
                "arg_count": len(args),
            }
            
            async with tracer.trace_execution_context(
                TraceEventType.TOOL_CALL_START,
                component="tool",
                component_name=component_name,
                inputs=inputs
            ):
                result = await func(*args, **kwargs)
                
                # Capture tool result
                await tracer.emit_event(
                    TraceEventType.TOOL_CALL_END,
                    component="tool",
                    component_name=component_name,
                    outputs={
                        "result": str(result)[:500],  # Truncate large results
                        "result_type": type(result).__name__,
                    },
                    status="completed"
                )
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_global_tracer()
            if not tracer:
                return func(*args, **kwargs)
            
            component_name = tool_name or "UnknownTool"
            logger.debug(f"Tracing tool call: {component_name}")
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator


@asynccontextmanager
async def trace_agent_turn(
    agent_name: str,
    turn_number: int,
    user_input: Optional[str] = None
):
    """
    Context manager for tracing individual agent turns.
    
    Usage:
        async with trace_agent_turn("MyAgent", 1, "user message"):
            # Agent turn processing
            pass
    """
    tracer = get_global_tracer()
    if not tracer:
        yield
        return
    
    async with tracer.trace_execution_context(
        TraceEventType.AGENT_TURN_START,
        component="agent",
        component_name=agent_name,
        inputs={
            "turn_number": turn_number,
            "user_input": user_input,
        },
        metadata={"turn": turn_number}
    ):
        yield


async def trace_memory_operation(
    operation_type: str,
    agent_name: str,
    details: Optional[Dict[str, Any]] = None
):
    """
    Trace memory operations during agent execution.
    
    Args:
        operation_type: Type of memory operation (add, retrieve, summarize, etc.)
        agent_name: Name of the agent performing the operation
        details: Additional operation details
    """
    tracer = get_global_tracer()
    if not tracer or not tracer.config.trace_memory_operations:
        return
    
    await tracer.emit_event(
        TraceEventType.MEMORY_OPERATION,
        component="agent",
        component_name=agent_name,
        inputs={
            "operation_type": operation_type,
            "details": details or {},
        },
        metadata={"memory_operation": operation_type}
    )


async def trace_validation_event(
    validator_name: str,
    input_data: Any,
    result: Any,
    success: bool,
    errors: Optional[List[str]] = None
):
    """
    Trace validation events during prompt execution.
    
    Args:
        validator_name: Name/type of validator
        input_data: Data being validated
        result: Validation result
        success: Whether validation succeeded
        errors: List of validation errors if any
    """
    tracer = get_global_tracer()
    if not tracer:
        return
    
    await tracer.emit_event(
        TraceEventType.VALIDATION_START,
        component="prompt",
        component_name=validator_name,
        inputs={"input_size": len(str(input_data))},
        outputs={
            "success": success,
            "errors": errors or [],
            "result_type": type(result).__name__,
        },
        status="completed" if success else "failed",
        error="; ".join(errors) if errors else None
    )


async def trace_error(
    component: str,
    component_name: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
):
    """
    Trace error events across all components.
    
    Args:
        component: Component type where error occurred
        component_name: Specific component instance name
        error: Exception that occurred
        context: Additional error context
    """
    tracer = get_global_tracer()
    if not tracer:
        return
    
    await tracer.emit_event(
        TraceEventType.ERROR_OCCURRED,
        component=component,  # type: ignore
        component_name=component_name,
        inputs=context or {},
        metadata={
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
        status="failed",
        error=str(error)
    )


__all__ = [
    "trace_agent_execution",
    "trace_prompt_execution", 
    "trace_chain_execution",
    "trace_llm_call",
    "trace_tool_call",
    "trace_agent_turn",
    "trace_memory_operation",
    "trace_validation_event",
    "trace_error",
]