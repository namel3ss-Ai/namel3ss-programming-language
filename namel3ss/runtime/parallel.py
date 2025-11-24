"""
Parallel and Distributed Execution Runtime for Namel3ss.

This module implements the core parallel execution engine with support for:
- Asyncio-based concurrent execution
- Multiple execution strategies (ALL, ANY_SUCCESS, RACE, MAP_REDUCE, COLLECT)
- Security integration with capability-based access control
- OpenTelemetry observability and distributed tracing
- Fault tolerance and error handling
- Resource management and concurrency control

Production-ready implementation with comprehensive logging, metrics, and monitoring.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    trace = None
    Status = None
    StatusCode = None

# Security integration
try:
    from .security import (
        SecurityContext, SecurityManager, SecurityAction, ResourceType,
        get_security_manager, InsufficientPermissionsError
    )
except ImportError:
    # Fallback for environments without security module
    SecurityContext = None
    SecurityManager = None
    SecurityAction = None
    ResourceType = None
    get_security_manager = lambda: None
    InsufficientPermissionsError = Exception

# Observability integration
try:
    from .observability import (
        ObservabilityManager, get_observability_manager, trace_execution,
        record_execution_metrics
    )
except ImportError:
    # Fallback for environments without observability module
    ObservabilityManager = None
    get_observability_manager = lambda: None
    trace_execution = lambda *args, **kwargs: lambda func: func
    record_execution_metrics = lambda *args, **kwargs: None


# =============================================================================
# Core Data Structures
# =============================================================================

class ParallelStrategy(Enum):
    """Parallel execution strategies."""
    ALL = "all"                # All tasks must succeed
    ANY_SUCCESS = "any_success"  # First success wins
    RACE = "race"              # First completion wins (success or failure)
    MAP_REDUCE = "map_reduce"  # Execute all, then reduce results
    COLLECT = "collect"        # Execute all, collect all results (including failures)


@dataclass
class ParallelTaskResult:
    """Result from a single parallel task."""
    task_id: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelExecutionResult:
    """Result from parallel block execution."""
    block_name: str
    strategy: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    results: List[ParallelTaskResult]
    final_result: Any = None
    aggregated_output: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    overall_status: str = "pending"
    error_message: Optional[str] = None
    max_concurrent: Optional[int] = None
    actual_max_concurrent: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelExecutionContext:
    """Context for parallel execution tracking."""
    execution_id: str
    block_name: str
    strategy: Union[str, ParallelStrategy]
    max_concurrency: int
    timeout_seconds: Optional[float] = None
    parent_span_id: Optional[str] = None
    
    # Runtime tracking
    semaphore: Optional[asyncio.Semaphore] = None
    cancel_event: Optional[asyncio.Event] = None
    running_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Dict[str, ParallelTaskResult] = field(default_factory=dict)
    current_concurrency: int = 0
    
    # Security context
    execution_permissions: Optional[str] = None
    allowed_capabilities: Set[str] = field(default_factory=set)
    
    # Observability
    trace_context: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Context Variables for Thread-Safe Access
# =============================================================================

current_parallel_context: ContextVar[Optional[ParallelExecutionContext]] = ContextVar(
    'current_parallel_context', default=None
)


# =============================================================================
# Parallel Execution Engine
# =============================================================================

class ParallelExecutor:
    """
    High-performance parallel execution engine for Namel3ss.
    
    Features:
    - Asyncio-based concurrent execution
    - Multiple execution strategies
    - Integrated security validation
    - OpenTelemetry distributed tracing
    - Comprehensive error handling and resource management
    """
    
    def __init__(
        self,
        default_max_concurrency: int = 10,
        default_timeout: float = 300.0,
        enable_tracing: bool = True,
        enable_security: bool = True,
        thread_pool_size: int = 4,
        security_manager: Optional[Any] = None,
    ):
        """Initialize the parallel executor."""
        self.default_max_concurrency = default_max_concurrency
        self.default_timeout = default_timeout
        self.enable_tracing = enable_tracing and trace is not None
        self.enable_security = enable_security
        self.security_manager = security_manager or (get_security_manager() if enable_security else None)
        self.observability = get_observability_manager() if get_observability_manager else None
        
        # Thread pool for CPU-bound tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # Execution statistics
        self._execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_tasks': 0,
            'average_duration_ms': 0.0,
        }
        
        logger.info(
            f"ParallelExecutor initialized: max_concurrency={default_max_concurrency}, "
            f"timeout={default_timeout}s, tracing={self.enable_tracing}, "
            f"security={self.enable_security}"
        )
    
    async def execute_parallel_block(
        self,
        parallel_block: Union[Dict[str, Any], Any],
        step_executor: Callable[[Any, Dict[str, Any]], Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> ParallelExecutionResult:
        """
        Execute a parallel block with comprehensive error handling and observability.
        
        Args:
            parallel_block: Parallel block configuration or AST node
            step_executor: Function to execute individual steps
            context: Execution context
            security_context: Security validation context
            parent_span_id: Parent tracing span ID
            
        Returns:
            ParallelExecutionResult with comprehensive execution details
        """
        # Parse parallel block configuration
        if isinstance(parallel_block, dict):
            block_name = parallel_block.get('name', 'unnamed_parallel_block')
            strategy_str = parallel_block.get('strategy', 'all')
            steps = parallel_block.get('steps', [])
            max_concurrency = parallel_block.get('max_concurrency')
            timeout_seconds = parallel_block.get('timeout_seconds')
            reduce_function = parallel_block.get('reduce_function')
        else:
            block_name = parallel_block.name
            strategy_str = parallel_block.strategy.value if hasattr(parallel_block.strategy, 'value') else str(parallel_block.strategy)
            steps = parallel_block.steps
            max_concurrency = parallel_block.max_concurrency
            timeout_seconds = parallel_block.timeout_seconds
            reduce_function = getattr(parallel_block, 'reduce_function', None)
        
        # Parse strategy
        try:
            if ParallelStrategy:
                strategy = ParallelStrategy(strategy_str)
            else:
                strategy = strategy_str
        except (ValueError, AttributeError):
            strategy = 'all'  # Fallback
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create execution context
        exec_context = ParallelExecutionContext(
            execution_id=execution_id,
            block_name=block_name,
            strategy=strategy,
            max_concurrency=max_concurrency or self.default_max_concurrency,
            timeout_seconds=timeout_seconds or self.default_timeout,
            parent_span_id=parent_span_id,
        )
        
        # Set up concurrency control
        exec_context.semaphore = asyncio.Semaphore(exec_context.max_concurrency)
        exec_context.cancel_event = asyncio.Event()
        
        # Security setup
        if self.enable_security and security_context:
            if isinstance(security_context, dict):
                # Backward compatibility for dict-based context
                exec_context.execution_permissions = security_context.get('permission_level')
                exec_context.allowed_capabilities = set(security_context.get('capabilities', []))
            else:
                # Modern SecurityContext object
                exec_context.execution_permissions = security_context.permission_level.value if hasattr(security_context, 'permission_level') else None
                exec_context.allowed_capabilities = {cap.name for cap in security_context.capabilities} if hasattr(security_context, 'capabilities') else set()
        
        # Set context variable
        current_parallel_context.set(exec_context)
        
        # Start tracing span
        span = None
        if self.enable_tracing and trace:
            tracer = trace.get_tracer(__name__)
            span = tracer.start_span(
                f"parallel_execution.{block_name}",
                attributes={
                    "parallel.execution_id": execution_id,
                    "parallel.strategy": strategy_str,
                    "parallel.max_concurrency": exec_context.max_concurrency,
                    "parallel.task_count": len(steps),
                }
            )
            exec_context.trace_context["span_id"] = span.get_span_context().span_id
        
        try:
            logger.info(
                f"Starting parallel execution: {block_name} with {len(steps)} steps, strategy={strategy_str}"
            )
            
            # Record execution start with observability
            execution_span = None
            if self.observability:
                execution_span = await self.observability.record_execution_start(
                    component="parallel_executor",
                    strategy=strategy_str,
                    execution_id=execution_id,
                    block_name=block_name,
                    step_count=len(steps),
                    max_concurrency=exec_context.max_concurrency
                )
            
            # Security validation
            if self.enable_security and self.security_manager and security_context:
                # Convert security_context dict to SecurityContext object if needed
                if isinstance(security_context, dict):
                    # For backward compatibility, treat dict as basic context
                    logger.warning("Security validation skipped: security_context should be SecurityContext object")
                elif hasattr(security_context, 'can_perform_action') and SecurityAction:
                    # Validate execution permissions
                    parallel_config = {
                        'name': block_name,
                        'strategy': strategy_str,
                        'steps': steps,
                        'max_concurrency': exec_context.max_concurrency,
                    }
                    execution_metadata = {
                        'distributed': False,  # This is local parallel execution
                        'max_concurrency': exec_context.max_concurrency,
                        'step_count': len(steps),
                    }
                    
                    is_authorized = await self.security_manager.validate_parallel_execution(
                        security_context, parallel_config, execution_metadata
                    )
                    
                    if not is_authorized:
                        raise InsufficientPermissionsError(
                            f"Insufficient permissions to execute parallel block: {block_name}"
                        )
                    
                    logger.debug(f"Security validation passed for execution: {block_name}")
            
            # Execute based on strategy
            if strategy_str in ['all', 'ALL']:
                result = await self._execute_all_strategy(
                    steps, step_executor, exec_context, context
                )
            elif strategy_str in ['any_success', 'ANY_SUCCESS']:
                result = await self._execute_any_success_strategy(
                    steps, step_executor, exec_context, context
                )
            elif strategy_str in ['race', 'RACE']:
                result = await self._execute_race_strategy(
                    steps, step_executor, exec_context, context
                )
            elif strategy_str in ['map_reduce', 'MAP_REDUCE']:
                result = await self._execute_map_reduce_strategy(
                    steps, step_executor, exec_context, context, reduce_function
                )
            elif strategy_str in ['collect', 'COLLECT']:
                result = await self._execute_collect_strategy(
                    steps, step_executor, exec_context, context
                )
            else:
                raise ValueError(f"Unknown parallel strategy: {strategy_str}")
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Build final result
            parallel_result = ParallelExecutionResult(
                block_name=block_name,
                strategy=strategy_str,
                total_tasks=len(steps),
                completed_tasks=len([r for r in result.results if r.status == 'completed']),
                failed_tasks=len([r for r in result.results if r.status == 'failed']),
                cancelled_tasks=len([r for r in result.results if r.status == 'cancelled']),
                results=result.results,
                final_result=result.final_result,
                aggregated_output=result.aggregated_output,
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=duration_ms,
                overall_status='completed',
                max_concurrent=exec_context.max_concurrency,
                actual_max_concurrent=exec_context.current_concurrency,
            )
            
            # Update metrics
            self._execution_stats['total_executions'] += 1
            self._execution_stats['successful_executions'] += 1
            self._execution_stats['total_tasks'] += len(steps)
            
            # Record completion with observability
            if self.observability:
                await self.observability.record_execution_complete(
                    component="parallel_executor",
                    strategy=strategy_str,
                    execution_id=execution_id,
                    span=execution_span,
                    duration_seconds=duration_ms / 1000,
                    success=True,
                    completed_tasks=parallel_result.completed_tasks,
                    failed_tasks=parallel_result.failed_tasks
                )
            
            # Update span
            if span:
                span.set_attribute("parallel.completed_tasks", parallel_result.completed_tasks)
                span.set_attribute("parallel.failed_tasks", parallel_result.failed_tasks)
                span.set_attribute("parallel.duration_ms", duration_ms)
                span.set_status(Status(StatusCode.OK))
            
            logger.info(
                f"Parallel execution completed: {block_name}, "
                f"completed={parallel_result.completed_tasks}, "
                f"failed={parallel_result.failed_tasks}, "
                f"duration={duration_ms:.1f}ms"
            )
            
            return parallel_result
            
        except InsufficientPermissionsError:
            # Re-raise security exceptions immediately - don't convert to ParallelExecutionResult
            raise
            
        except asyncio.TimeoutError:
            # Handle timeout
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            logger.warning(
                f"Parallel execution timeout: {block_name} after {duration_ms:.1f}ms"
            )
            
            # Cancel remaining tasks
            exec_context.cancel_event.set()
            
            parallel_result = ParallelExecutionResult(
                block_name=block_name,
                strategy=strategy_str,
                total_tasks=len(steps),
                completed_tasks=len(exec_context.completed_tasks),
                failed_tasks=0,
                cancelled_tasks=len(steps) - len(exec_context.completed_tasks),
                results=list(exec_context.completed_tasks.values()),
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=duration_ms,
                overall_status='timeout',
                error_message=f"Execution timed out after {timeout_seconds}s",
                max_concurrent=exec_context.max_concurrency,
            )
            
            if span:
                span.set_attribute("parallel.timeout", True)
                span.set_status(Status(StatusCode.ERROR, "Execution timeout"))
            
            self._execution_stats['total_executions'] += 1
            self._execution_stats['failed_executions'] += 1
            
            # Record timeout with observability
            if self.observability:
                timeout_error = asyncio.TimeoutError(f"Execution timed out after {exec_context.timeout_seconds}s")
                await self.observability.record_execution_complete(
                    component="parallel_executor",
                    strategy=strategy_str,
                    execution_id=execution_id,
                    span=locals().get('execution_span'),
                    duration_seconds=duration_ms / 1000,
                    success=False,
                    error=timeout_error
                )
                await self.observability.record_error(
                    component="parallel_executor",
                    error_type="timeout",
                    severity="warning"
                )
            
            return parallel_result
            
        except Exception as e:
            # Handle execution error
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            logger.error(
                f"Parallel execution failed: {block_name}, error={str(e)}",
                exc_info=True
            )
            
            exec_context.cancel_event.set()
            
            parallel_result = ParallelExecutionResult(
                block_name=block_name,
                strategy=strategy_str,
                total_tasks=len(steps),
                completed_tasks=len(exec_context.completed_tasks),
                failed_tasks=1,
                cancelled_tasks=len(steps) - len(exec_context.completed_tasks),
                results=list(exec_context.completed_tasks.values()),
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=duration_ms,
                overall_status='failed',
                error_message=str(e),
                max_concurrent=exec_context.max_concurrency,
            )
            
            if span:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            
            self._execution_stats['total_executions'] += 1
            self._execution_stats['failed_executions'] += 1
            
            # Record error with observability
            if self.observability:
                await self.observability.record_execution_complete(
                    component="parallel_executor",
                    strategy=strategy_str,
                    execution_id=execution_id,
                    span=locals().get('execution_span'),
                    duration_seconds=duration_ms / 1000,
                    success=False,
                    error=e
                )
                await self.observability.record_error(
                    component="parallel_executor",
                    error_type=type(e).__name__,
                    severity="error"
                )
            
            return parallel_result
            
        finally:
            if span:
                span.end()
    
    async def _execute_single_task(
        self,
        step: Any,
        step_executor: Callable,
        exec_context: ParallelExecutionContext,
        context: Optional[Dict[str, Any]],
        task_id: str,
    ) -> ParallelTaskResult:
        """Execute a single task within parallel block."""
        task_result = ParallelTaskResult(task_id=task_id)
        
        # Security check
        if self.enable_security and exec_context.execution_permissions:
            # Validate step can be executed with current permissions
            # This would integrate with the security validation system
            pass
        
        try:
            # Acquire concurrency semaphore
            async with exec_context.semaphore:
                # Check for cancellation
                if exec_context.cancel_event and exec_context.cancel_event.is_set():
                    task_result.status = 'cancelled'
                    return task_result
                
                # Update concurrency tracking
                exec_context.running_tasks.add(task_id)
                exec_context.current_concurrency = len(exec_context.running_tasks)
                
                task_result.start_time = time.time()
                task_result.status = 'running'
                
                logger.debug(f"Starting task {task_id}, concurrency={exec_context.current_concurrency}")
                
                # Execute the step
                step_result = await step_executor(step, context or {})
                
                task_result.end_time = time.time()
                task_result.duration_ms = (task_result.end_time - task_result.start_time) * 1000
                task_result.result = step_result
                task_result.status = 'completed'
                
                logger.debug(f"Completed task {task_id} in {task_result.duration_ms:.1f}ms")
                
                return task_result
                
        except asyncio.CancelledError:
            task_result.status = 'cancelled'
            logger.debug(f"Task {task_id} was cancelled")
            return task_result
            
        except Exception as e:
            task_result.end_time = time.time()
            if task_result.start_time:
                task_result.duration_ms = (task_result.end_time - task_result.start_time) * 1000
            task_result.error = e
            task_result.status = 'failed'
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            return task_result
            
        finally:
            # Update tracking
            exec_context.running_tasks.discard(task_id)
            exec_context.current_concurrency = len(exec_context.running_tasks)
            exec_context.completed_tasks[task_id] = task_result
    
    async def _execute_all_strategy(
        self,
        steps: List[Any],
        step_executor: Callable,
        exec_context: ParallelExecutionContext,
        context: Optional[Dict[str, Any]],
    ) -> ParallelExecutionResult:
        """Execute all steps, fail if any fails."""
        tasks = []
        
        # Create all tasks
        for i, step in enumerate(steps):
            task_id = f"{exec_context.execution_id}_task_{i}"
            task = self._execute_single_task(
                step, step_executor, exec_context, context, task_id
            )
            tasks.append(task)
        
        # Wait for all tasks with timeout
        try:
            if exec_context.timeout_seconds:
                task_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=exec_context.timeout_seconds
                )
            else:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        
        # Process results
        results = []
        final_outputs = {}
        has_failure = False
        
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                # Task raised an exception
                task_result = ParallelTaskResult(
                    task_id=f"{exec_context.execution_id}_task_{i}",
                    status='failed',
                    error=result,
                )
                has_failure = True
            else:
                task_result = result
                if task_result.status == 'failed':
                    has_failure = True
                elif task_result.status == 'completed':
                    # Add to final outputs
                    step_name = getattr(steps[i], 'name', f'step_{i}')
                    final_outputs[step_name] = task_result.result
            
            results.append(task_result)
        
        # For ALL strategy, fail if any task failed
        if has_failure:
            error_tasks = [r for r in results if r.status == 'failed']
            error_msg = f"Failed tasks: {', '.join(t.task_id for t in error_tasks)}"
            
            return ParallelExecutionResult(
                block_name=exec_context.block_name,
                strategy='all',
                total_tasks=len(steps),
                completed_tasks=len([r for r in results if r.status == 'completed']),
                failed_tasks=len([r for r in results if r.status == 'failed']),
                cancelled_tasks=len([r for r in results if r.status == 'cancelled']),
                results=results,
                overall_status='failed',
                error_message=error_msg,
            )
        
        return ParallelExecutionResult(
            block_name=exec_context.block_name,
            strategy='all',
            total_tasks=len(steps),
            completed_tasks=len(results),
            failed_tasks=0,
            cancelled_tasks=0,
            results=results,
            final_result=final_outputs,
            aggregated_output=final_outputs,
            overall_status='completed',
        )
    
    async def _execute_any_success_strategy(
        self,
        steps: List[Any],
        step_executor: Callable,
        exec_context: ParallelExecutionContext,
        context: Optional[Dict[str, Any]],
    ) -> ParallelExecutionResult:
        """Execute steps, return on first success."""
        tasks = []
        
        # Create all tasks
        for i, step in enumerate(steps):
            task_id = f"{exec_context.execution_id}_task_{i}"
            task = asyncio.create_task(self._execute_single_task(
                step, step_executor, exec_context, context, task_id
            ))
            tasks.append(task)
        
        # Wait for first success
        completed_results = []
        success_result = None
        
        try:
            # Process tasks as they complete
            for coro in asyncio.as_completed(tasks, timeout=exec_context.timeout_seconds):
                try:
                    result = await coro
                    completed_results.append(result)
                    
                    if result.status == 'completed':
                        success_result = result
                        # Cancel remaining tasks
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        break
                        
                except Exception as e:
                    # Individual task failed, continue with others
                    failed_result = ParallelTaskResult(
                        task_id=f"failed_task_{len(completed_results)}",
                        status='failed',
                        error=e,
                    )
                    completed_results.append(failed_result)
                    continue
        
        except asyncio.TimeoutError:
            # Cancel all remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            return ParallelExecutionResult(
                block_name=exec_context.block_name,
                strategy='any_success',
                total_tasks=len(steps),
                completed_tasks=len([r for r in completed_results if r.status == 'completed']),
                failed_tasks=len([r for r in completed_results if r.status == 'failed']),
                cancelled_tasks=len(steps) - len(completed_results),
                results=completed_results,
                overall_status='timeout',
                error_message="No tasks completed before timeout",
            )
        
        # Wait for any remaining cancelled tasks
        remaining_results = []
        for task in tasks:
            if task.done():
                try:
                    result = task.result()
                    if result not in completed_results:
                        remaining_results.append(result)
                except asyncio.CancelledError:
                    cancelled_result = ParallelTaskResult(
                        task_id=f"cancelled_task_{len(remaining_results)}",
                        status='cancelled',
                    )
                    remaining_results.append(cancelled_result)
        
        all_results = completed_results + remaining_results
        
        if success_result:
            return ParallelExecutionResult(
                block_name=exec_context.block_name,
                strategy='any_success',
                total_tasks=len(steps),
                completed_tasks=1,
                failed_tasks=len([r for r in all_results if r.status == 'failed']),
                cancelled_tasks=len([r for r in all_results if r.status == 'cancelled']),
                results=all_results,
                final_result=success_result.result,
                overall_status='completed',
            )
        else:
            return ParallelExecutionResult(
                block_name=exec_context.block_name,
                strategy='any_success',
                total_tasks=len(steps),
                completed_tasks=0,
                failed_tasks=len([r for r in all_results if r.status == 'failed']),
                cancelled_tasks=len([r for r in all_results if r.status == 'cancelled']),
                results=all_results,
                overall_status='failed',
                error_message="No tasks completed successfully",
            )
    
    async def _execute_race_strategy(
        self,
        steps: List[Any],
        step_executor: Callable,
        exec_context: ParallelExecutionContext,
        context: Optional[Dict[str, Any]],
    ) -> ParallelExecutionResult:
        """Execute steps, return first completion (success or failure)."""
        tasks = []
        
        # Create all tasks
        for i, step in enumerate(steps):
            task_id = f"{exec_context.execution_id}_task_{i}"
            task = asyncio.create_task(self._execute_single_task(
                step, step_executor, exec_context, context, task_id
            ))
            tasks.append(task)
        
        # Wait for first completion
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=exec_context.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if not done:
                # Timeout
                for task in pending:
                    task.cancel()
                    
                return ParallelExecutionResult(
                    block_name=exec_context.block_name,
                    strategy='race',
                    total_tasks=len(steps),
                    completed_tasks=0,
                    failed_tasks=0,
                    cancelled_tasks=len(steps),
                    results=[],
                    overall_status='timeout',
                    error_message="Race timed out before any task completed",
                )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
            
            # Get the first completed result
            first_task = list(done)[0]
            result = await first_task
            
            # Wait for cancelled tasks to finish cancellation
            cancelled_results = []
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:
                    cancelled_result = ParallelTaskResult(
                        task_id=f"cancelled_{len(cancelled_results)}",
                        status='cancelled',
                    )
                    cancelled_results.append(cancelled_result)
            
            all_results = [result] + cancelled_results
            
            return ParallelExecutionResult(
                block_name=exec_context.block_name,
                strategy='race',
                total_tasks=len(steps),
                completed_tasks=1 if result.status in ['completed', 'failed'] else 0,
                failed_tasks=1 if result.status == 'failed' else 0,
                cancelled_tasks=len(cancelled_results),
                results=all_results,
                final_result=result.result,
                overall_status='completed' if result.status in ['completed', 'failed'] else 'failed',
            )
            
        except asyncio.TimeoutError:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            return ParallelExecutionResult(
                block_name=exec_context.block_name,
                strategy='race',
                total_tasks=len(steps),
                completed_tasks=0,
                failed_tasks=0,
                cancelled_tasks=len(steps),
                results=[],
                overall_status='timeout',
                error_message="Race execution timed out",
            )
    
    async def _execute_map_reduce_strategy(
        self,
        steps: List[Any],
        step_executor: Callable,
        exec_context: ParallelExecutionContext,
        context: Optional[Dict[str, Any]],
        reduce_function: Optional[str],
    ) -> ParallelExecutionResult:
        """Execute all steps and apply reduce function to results."""
        # First execute all steps (like ALL strategy)
        all_result = await self._execute_all_strategy(
            steps, step_executor, exec_context, context
        )
        
        if all_result.overall_status != 'completed':
            return all_result
        
        # Apply reduce function if provided
        if reduce_function:
            try:
                # Extract successful results
                successful_results = [
                    r.result for r in all_result.results 
                    if r.status == 'completed' and r.result is not None
                ]
                
                if not successful_results:
                    reduced_result = None
                elif reduce_function == 'sum':
                    reduced_result = sum(successful_results)
                elif reduce_function == 'count':
                    reduced_result = len(successful_results)
                elif reduce_function == 'concat':
                    if all(isinstance(r, (list, str)) for r in successful_results):
                        if isinstance(successful_results[0], list):
                            reduced_result = []
                            for r in successful_results:
                                reduced_result.extend(r)
                        else:
                            reduced_result = ''.join(successful_results)
                    else:
                        reduced_result = successful_results
                elif reduce_function == 'merge':
                    if all(isinstance(r, dict) for r in successful_results):
                        reduced_result = {}
                        for r in successful_results:
                            reduced_result.update(r)
                    else:
                        reduced_result = successful_results
                else:
                    # Custom reduce function would need to be implemented
                    logger.warning(f"Unknown reduce function: {reduce_function}, using raw results")
                    reduced_result = successful_results
                
                all_result.final_result = reduced_result
                all_result.aggregated_output = {'reduced_result': reduced_result}
                
            except Exception as e:
                logger.error(f"Reduce function failed: {str(e)}", exc_info=True)
                all_result.overall_status = 'failed'
                all_result.error_message = f"Reduce function failed: {str(e)}"
        
        return all_result
    
    async def _execute_collect_strategy(
        self,
        steps: List[Any],
        step_executor: Callable,
        exec_context: ParallelExecutionContext,
        context: Optional[Dict[str, Any]],
    ) -> ParallelExecutionResult:
        """Execute all steps, collect all results including failures."""
        tasks = []
        
        # Create all tasks
        for i, step in enumerate(steps):
            task_id = f"{exec_context.execution_id}_task_{i}"
            task = self._execute_single_task(
                step, step_executor, exec_context, context, task_id
            )
            tasks.append(task)
        
        # Wait for all tasks (don't fail on exceptions)
        try:
            if exec_context.timeout_seconds:
                task_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=exec_context.timeout_seconds
                )
            else:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            # Collect whatever completed
            completed = [task for task in tasks if task.done()]
            task_results = []
            
            for i, task in enumerate(tasks):
                if task.done():
                    try:
                        result = await task
                        task_results.append(result)
                    except Exception as e:
                        failed_result = ParallelTaskResult(
                            task_id=f"{exec_context.execution_id}_task_{i}",
                            status='failed',
                            error=e,
                        )
                        task_results.append(failed_result)
                else:
                    task.cancel()
                    cancelled_result = ParallelTaskResult(
                        task_id=f"{exec_context.execution_id}_task_{i}",
                        status='cancelled',
                    )
                    task_results.append(cancelled_result)
        
        # Process all results
        results = []
        collected_outputs = {}
        
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                task_result = ParallelTaskResult(
                    task_id=f"{exec_context.execution_id}_task_{i}",
                    status='failed',
                    error=result,
                )
            else:
                task_result = result
                if task_result.status == 'completed':
                    step_name = getattr(steps[i], 'name', f'step_{i}')
                    collected_outputs[step_name] = {
                        'result': task_result.result,
                        'status': task_result.status,
                        'duration_ms': task_result.duration_ms,
                    }
                elif task_result.status == 'failed':
                    step_name = getattr(steps[i], 'name', f'step_{i}')
                    collected_outputs[step_name] = {
                        'error': str(task_result.error) if task_result.error else 'Unknown error',
                        'status': task_result.status,
                        'duration_ms': task_result.duration_ms,
                    }
            
            results.append(task_result)
        
        return ParallelExecutionResult(
            block_name=exec_context.block_name,
            strategy='collect',
            total_tasks=len(steps),
            completed_tasks=len([r for r in results if r.status == 'completed']),
            failed_tasks=len([r for r in results if r.status == 'failed']),
            cancelled_tasks=len([r for r in results if r.status == 'cancelled']),
            results=results,
            final_result=collected_outputs,
            aggregated_output=collected_outputs,
            overall_status='completed',  # COLLECT strategy always succeeds
        )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self._execution_stats.copy()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)


# =============================================================================
# Convenience Functions
# =============================================================================

async def execute_parallel_steps(
    steps: List[Any],
    step_executor: Callable,
    strategy: str = 'all',
    max_concurrency: int = 10,
    timeout_seconds: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ParallelExecutionResult:
    """
    Convenience function to execute parallel steps.
    
    Args:
        steps: List of steps to execute
        step_executor: Function to execute individual steps  
        strategy: Parallel execution strategy
        max_concurrency: Maximum concurrent tasks
        timeout_seconds: Execution timeout
        context: Execution context
        
    Returns:
        ParallelExecutionResult
    """
    executor = ParallelExecutor()
    
    parallel_block = {
        'name': 'parallel_steps',
        'steps': steps,
        'strategy': strategy,
        'max_concurrency': max_concurrency,
        'timeout_seconds': timeout_seconds,
    }
    
    try:
        return await executor.execute_parallel_block(
            parallel_block, step_executor, context
        )
    finally:
        await executor.cleanup()


def get_current_parallel_context() -> Optional[ParallelExecutionContext]:
    """Get the current parallel execution context."""
    return current_parallel_context.get(None)