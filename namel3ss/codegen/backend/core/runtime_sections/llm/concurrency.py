"""Concurrency utilities for parallel chain step execution."""

from textwrap import dedent

CONCURRENCY = dedent(
    '''
import asyncio
from typing import Any, Dict, List, Optional, Tuple

# Semaphore for rate limiting concurrent LLM calls
_LLM_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Get or create the global LLM rate limiting semaphore."""
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        max_concurrent = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "50"))
        _LLM_SEMAPHORE = asyncio.Semaphore(max_concurrent)
    return _LLM_SEMAPHORE


async def _execute_step_with_semaphore(
    step: Dict[str, Any],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any, bool]:
    """Execute a single step with semaphore-based rate limiting."""
    semaphore = _get_llm_semaphore()
    
    async with semaphore:
        return await _execute_workflow_step(
            step,
            context=context,
            chain_scope=chain_scope,
            args=args,
            working=working,
            memory_state=memory_state,
            allow_stubs=allow_stubs,
            steps_history=steps_history,
            chain_name=chain_name,
        )


async def _execute_parallel_steps(
    steps: List[Dict[str, Any]],
    *,
    context: Dict[str, Any],
    chain_scope: Dict[str, Any],
    args: Dict[str, Any],
    working: Any,
    memory_state: Dict[str, Dict[str, Any]],
    allow_stubs: bool,
    steps_history: List[Dict[str, Any]],
    chain_name: str,
) -> Tuple[str, Any, Any]:
    """
    Execute multiple independent steps in parallel using asyncio.gather().
    
    Returns the status, result, and working value from the final step.
    """
    if not steps:
        return "partial", None, working
    
    # Create tasks for all steps
    tasks = []
    for step in steps:
        task = _execute_step_with_semaphore(
            step,
            context=context,
            chain_scope=chain_scope,
            args=args,
            working=working,
            memory_state=memory_state,
            allow_stubs=allow_stubs,
            steps_history=steps_history,
            chain_name=chain_name,
        )
        tasks.append(task)
    
    # Execute all steps concurrently
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as exc:
        _record_runtime_error(
            context,
            code="parallel_execution_error",
            message=f"Parallel step execution failed: {exc}",
            scope=chain_name,
            source="concurrency",
        )
        return "error", {"error": str(exc)}, working
    
    # Process results
    final_status = "partial"
    final_result = None
    final_working = working
    encountered_error = False
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            encountered_error = True
            logger.error(f"Step {i} in parallel group failed: {result}")
            if not allow_stubs:
                return "error", {"error": str(result)}, working
        else:
            step_status, step_working, step_result, stop_on_error = result
            
            if step_status == "error":
                encountered_error = True
                if stop_on_error:
                    return "error", step_result, step_working
            elif step_status == "ok":
                final_status = "ok"
                final_result = step_result
                final_working = step_working
            elif step_status == "stub" and final_status != "error" and final_result is None:
                final_result = step_result
    
    if encountered_error and final_status != "error":
        final_status = "error"
    
    return final_status, final_result, final_working


async def _execute_with_timeout(
    coro,
    timeout_seconds: Optional[float] = None,
    chain_name: str = "unknown",
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Execute a coroutine with optional timeout.
    
    Falls back to environment variable CHAIN_TIMEOUT_SECONDS if not specified.
    """
    if timeout_seconds is None:
        timeout_seconds = float(os.getenv("CHAIN_TIMEOUT_SECONDS", "300"))  # 5 minutes default
    
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        error_msg = f"Chain '{chain_name}' execution timed out after {timeout_seconds} seconds"
        logger.error(error_msg)
        if context:
            _record_runtime_error(
                context,
                code="chain_timeout",
                message=error_msg,
                scope=chain_name,
                source="concurrency",
            )
        raise TimeoutError(error_msg)


def _detect_parallel_groups(steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Detect groups of steps that can be executed in parallel.
    
    Steps are considered parallelizable if they:
    1. Don't have explicit 'depends_on' fields
    2. Are not control flow nodes (if/for/while)
    3. Are marked with 'parallel: true' in their config
    
    Returns list of groups, where each group is a list of steps to execute in parallel.
    """
    groups = []
    current_sequential = []
    parallel_group = []
    
    for step in steps:
        node_type = str(step.get("type") or "step").lower()
        is_parallel = step.get("parallel") is True
        has_dependencies = bool(step.get("depends_on"))
        
        # Control flow nodes must be sequential
        if node_type in ["if", "for", "while"]:
            # Flush parallel group
            if parallel_group:
                groups.append(parallel_group)
                parallel_group = []
            # Add control flow node
            groups.append([step])
            continue
        
        # Steps with dependencies must be sequential
        if has_dependencies:
            # Flush parallel group
            if parallel_group:
                groups.append(parallel_group)
                parallel_group = []
            groups.append([step])
            continue
        
        # Parallel-marked steps
        if is_parallel:
            parallel_group.append(step)
        else:
            # Sequential step - flush parallel group first
            if parallel_group:
                groups.append(parallel_group)
                parallel_group = []
            groups.append([step])
    
    # Flush remaining parallel group
    if parallel_group:
        groups.append(parallel_group)
    
    return groups
'''
).strip()

__all__ = ['CONCURRENCY']
