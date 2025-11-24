"""
Distributed Execution Coordinator for Namel3ss.

This module integrates parallel execution with distributed task queues to provide:
- Seamless scaling from single-node to multi-node execution
- Intelligent task distribution and load balancing
- Fault tolerance and automatic failover
- Resource-aware scheduling
- Integration with existing parallel execution patterns

Main entry point for distributed parallel execution in Namel3ss.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable

from .parallel import (
    ParallelExecutor, ParallelExecutionResult, ParallelExecutionContext,
    ParallelStrategy, current_parallel_context
)
from .distributed import (
    DistributedTaskQueue, DistributedTask, TaskStatus, WorkerNode, 
    create_distributed_queue, create_message_broker
)

logger = logging.getLogger(__name__)


# =============================================================================
# Distributed Parallel Execution Coordinator
# =============================================================================

class DistributedParallelExecutor:
    """
    Distributed parallel execution coordinator.
    
    Combines parallel execution patterns with distributed task queues for:
    - Automatic scaling across multiple nodes
    - Intelligent task distribution
    - Fault tolerance and recovery
    - Resource optimization
    """
    
    def __init__(
        self,
        broker_type: str = 'redis',
        broker_config: Optional[Dict[str, Any]] = None,
        default_queue: str = 'namel3ss_parallel',
        fallback_to_local: bool = True,
        max_distributed_tasks: int = 1000,
        distributed_timeout: float = 600.0,
        enable_auto_scaling: bool = True,
    ):
        """Initialize distributed parallel executor."""
        self.broker_type = broker_type
        self.broker_config = broker_config or {}
        self.default_queue = default_queue
        self.fallback_to_local = fallback_to_local
        self.max_distributed_tasks = max_distributed_tasks
        self.distributed_timeout = distributed_timeout
        self.enable_auto_scaling = enable_auto_scaling
        
        # Components
        self.local_executor = ParallelExecutor()
        self.distributed_queue: Optional[DistributedTaskQueue] = None
        
        # Execution strategy decisions
        self.distribution_threshold = 5  # Tasks needed to consider distribution
        self.worker_availability_threshold = 0.8  # Max worker utilization
        
        # Statistics and monitoring
        self.execution_stats = {
            'total_executions': 0,
            'local_executions': 0,
            'distributed_executions': 0,
            'failed_distributions': 0,
            'average_distribution_overhead_ms': 0.0,
        }
        
        logger.info(
            f"DistributedParallelExecutor initialized: broker={broker_type}, "
            f"queue={default_queue}, fallback={fallback_to_local}"
        )
    
    async def initialize(self) -> None:
        """Initialize distributed components."""
        try:
            # Create distributed task queue
            self.distributed_queue = await create_distributed_queue(
                broker_type=self.broker_type,
                queue_name=self.default_queue,
                **self.broker_config
            )
            
            logger.info("Distributed task queue initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize distributed queue: {e}")
            if not self.fallback_to_local:
                raise
            logger.info("Falling back to local execution only")
    
    async def cleanup(self) -> None:
        """Clean up distributed components."""
        if self.distributed_queue:
            await self.distributed_queue.stop()
        await self.local_executor.cleanup()
    
    async def execute_parallel_block(
        self,
        parallel_block: Union[Dict[str, Any], Any],
        step_executor: Callable[[Any, Dict[str, Any]], Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[Dict[str, Any]] = None,
        force_distribution: bool = False,
        force_local: bool = False,
    ) -> ParallelExecutionResult:
        """
        Execute parallel block with intelligent distribution strategy.
        
        Args:
            parallel_block: Parallel block configuration or AST node
            step_executor: Function to execute individual steps
            context: Execution context
            security_context: Security validation context
            force_distribution: Force distributed execution
            force_local: Force local execution
            
        Returns:
            ParallelExecutionResult with execution details
        """
        self.execution_stats['total_executions'] += 1
        
        # Parse block configuration
        if isinstance(parallel_block, dict):
            steps = parallel_block.get('steps', [])
            strategy = parallel_block.get('strategy', 'all')
            max_concurrency = parallel_block.get('max_concurrency')
            block_name = parallel_block.get('name', 'unnamed_parallel_block')
        else:
            steps = parallel_block.steps
            strategy = parallel_block.strategy.value if hasattr(parallel_block.strategy, 'value') else str(parallel_block.strategy)
            max_concurrency = parallel_block.max_concurrency
            block_name = parallel_block.name
        
        # Decision: Local vs Distributed execution
        should_distribute = await self._should_distribute_execution(
            len(steps), strategy, force_distribution, force_local
        )
        
        if should_distribute:
            logger.info(f"Executing {block_name} with distributed strategy")
            return await self._execute_distributed(
                parallel_block, step_executor, context, security_context
            )
        else:
            logger.info(f"Executing {block_name} with local strategy")
            return await self._execute_local(
                parallel_block, step_executor, context, security_context
            )
    
    async def _should_distribute_execution(
        self,
        task_count: int,
        strategy: str,
        force_distribution: bool,
        force_local: bool,
    ) -> bool:
        """Decide whether to use distributed execution."""
        if force_local:
            return False
        
        if force_distribution:
            return True
        
        # Check if distributed queue is available
        if not self.distributed_queue:
            return False
        
        # Check task count threshold
        if task_count < self.distribution_threshold:
            return False
        
        # Check worker availability
        try:
            queue_status = await self.distributed_queue.get_queue_status()
            active_workers = queue_status['active_workers']
            busy_workers = queue_status['busy_workers']
            
            if active_workers == 0:
                return False
            
            # Check worker utilization
            utilization = busy_workers / active_workers if active_workers > 0 else 1.0
            if utilization > self.worker_availability_threshold:
                return False
            
            # Strategies that benefit from distribution
            beneficial_strategies = ['all', 'collect', 'map_reduce']
            if strategy.lower() not in beneficial_strategies:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking worker availability: {e}")
            return False
    
    async def _execute_local(
        self,
        parallel_block: Union[Dict[str, Any], Any],
        step_executor: Callable[[Any, Dict[str, Any]], Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[Dict[str, Any]] = None,
    ) -> ParallelExecutionResult:
        """Execute using local parallel executor."""
        self.execution_stats['local_executions'] += 1
        
        return await self.local_executor.execute_parallel_block(
            parallel_block, step_executor, context, security_context
        )
    
    async def _execute_distributed(
        self,
        parallel_block: Union[Dict[str, Any], Any],
        step_executor: Callable[[Any, Dict[str, Any]], Any],
        context: Optional[Dict[str, Any]] = None,
        security_context: Optional[Dict[str, Any]] = None,
    ) -> ParallelExecutionResult:
        """Execute using distributed task queue."""
        start_time = time.time()
        
        try:
            self.execution_stats['distributed_executions'] += 1
            
            # Parse block configuration
            if isinstance(parallel_block, dict):
                steps = parallel_block.get('steps', [])
                strategy = parallel_block.get('strategy', 'all')
                block_name = parallel_block.get('name', 'unnamed_parallel_block')
                timeout_seconds = parallel_block.get('timeout_seconds', self.distributed_timeout)
            else:
                steps = parallel_block.steps
                strategy = parallel_block.strategy.value if hasattr(parallel_block.strategy, 'value') else str(parallel_block.strategy)
                block_name = parallel_block.name
                timeout_seconds = parallel_block.timeout_seconds or self.distributed_timeout
            
            logger.info(f"Distributing {len(steps)} tasks for block {block_name}")
            
            # Submit tasks to distributed queue
            task_configs = []
            for i, step in enumerate(steps):
                # Serialize step for distributed execution
                step_data = self._serialize_step(step, context)
                
                task_config = {
                    'task_type': 'namel3ss_step_execution',
                    'payload': {
                        'step_index': i,
                        'step_data': step_data,
                        'execution_context': context or {},
                        'security_context': security_context or {},
                        'block_name': block_name,
                    }
                }
                task_configs.append(task_config)
            
            # Submit batch of tasks
            task_ids = await self.distributed_queue.submit_batch(
                task_configs,
                timeout_seconds=timeout_seconds,
            )
            
            # Wait for results based on strategy
            if strategy.lower() in ['all', 'collect', 'map_reduce']:
                # Wait for all tasks
                results = await self.distributed_queue.get_batch_results(
                    task_ids, timeout_seconds
                )
                
                return self._aggregate_distributed_results(
                    block_name, strategy, steps, results, start_time
                )
                
            elif strategy.lower() == 'any_success':
                # Wait for first success
                return await self._wait_for_first_success(
                    block_name, strategy, steps, task_ids, timeout_seconds, start_time
                )
                
            elif strategy.lower() == 'race':
                # Wait for first completion
                return await self._wait_for_first_completion(
                    block_name, strategy, steps, task_ids, timeout_seconds, start_time
                )
                
            else:
                raise ValueError(f"Unknown distributed strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Distributed execution failed: {e}")
            self.execution_stats['failed_distributions'] += 1
            
            # Fallback to local execution if enabled
            if self.fallback_to_local:
                logger.info("Falling back to local execution")
                return await self._execute_local(
                    parallel_block, step_executor, context, security_context
                )
            else:
                raise
        
        finally:
            # Update distribution overhead statistics
            end_time = time.time()
            overhead_ms = (end_time - start_time) * 1000
            current_avg = self.execution_stats['average_distribution_overhead_ms']
            total_distributed = self.execution_stats['distributed_executions']
            
            # Running average
            if total_distributed > 0:
                self.execution_stats['average_distribution_overhead_ms'] = (
                    (current_avg * (total_distributed - 1) + overhead_ms) / total_distributed
                )
    
    def _serialize_step(self, step: Any, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Serialize step for distributed execution."""
        # This is a simplified serialization - in practice, you'd need
        # more sophisticated serialization for complex step objects
        
        if hasattr(step, '__dict__'):
            # Object with attributes
            return {
                'type': 'object',
                'class_name': step.__class__.__name__,
                'attributes': {
                    key: value for key, value in step.__dict__.items()
                    if not key.startswith('_') and not callable(value)
                }
            }
        elif isinstance(step, dict):
            # Dictionary step
            return {
                'type': 'dict',
                'data': step
            }
        else:
            # Simple value
            return {
                'type': 'value',
                'data': step
            }
    
    def _deserialize_step(self, step_data: Dict[str, Any]) -> Any:
        """Deserialize step from distributed execution."""
        step_type = step_data.get('type')
        
        if step_type == 'dict':
            return step_data['data']
        elif step_type == 'value':
            return step_data['data']
        elif step_type == 'object':
            # For object deserialization, you'd need to reconstruct
            # the object based on class_name and attributes
            # This is simplified for demonstration
            return step_data['attributes']
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    def _aggregate_distributed_results(
        self,
        block_name: str,
        strategy: str,
        steps: List[Any],
        results: Dict[str, DistributedTask],
        start_time: float,
    ) -> ParallelExecutionResult:
        """Aggregate results from distributed execution."""
        from .parallel import ParallelTaskResult, ParallelExecutionResult
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Convert distributed results to parallel task results
        task_results = []
        completed_count = 0
        failed_count = 0
        cancelled_count = 0
        final_outputs = {}
        
        for i, (task_id, distributed_task) in enumerate(results.items()):
            # Map distributed task status to parallel task status
            if distributed_task.status == TaskStatus.COMPLETED:
                status = 'completed'
                completed_count += 1
                step_name = getattr(steps[i], 'name', f'step_{i}')
                final_outputs[step_name] = distributed_task.result
            elif distributed_task.status == TaskStatus.FAILED:
                status = 'failed'
                failed_count += 1
            elif distributed_task.status == TaskStatus.CANCELLED:
                status = 'cancelled'
                cancelled_count += 1
            else:
                status = 'failed'  # Treat other statuses as failed
                failed_count += 1
            
            task_result = ParallelTaskResult(
                task_id=task_id,
                status=status,
                result=distributed_task.result,
                error=distributed_task.error,
                start_time=distributed_task.started_at,
                end_time=distributed_task.completed_at,
                duration_ms=(
                    (distributed_task.completed_at - distributed_task.started_at) * 1000
                    if distributed_task.started_at and distributed_task.completed_at
                    else None
                ),
                metadata={'worker_id': distributed_task.worker_id}
            )
            task_results.append(task_result)
        
        # Determine overall status
        if strategy.lower() == 'all' and failed_count > 0:
            overall_status = 'failed'
        elif strategy.lower() in ['collect'] or (strategy.lower() == 'all' and failed_count == 0):
            overall_status = 'completed'
        else:
            overall_status = 'completed' if completed_count > 0 else 'failed'
        
        return ParallelExecutionResult(
            block_name=block_name,
            strategy=strategy,
            total_tasks=len(steps),
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            cancelled_tasks=cancelled_count,
            results=task_results,
            final_result=final_outputs if final_outputs else None,
            aggregated_output=final_outputs if final_outputs else None,
            start_time=start_time,
            end_time=end_time,
            total_duration_ms=duration_ms,
            overall_status=overall_status,
            metadata={
                'execution_type': 'distributed',
                'worker_usage': self._calculate_worker_usage(task_results),
            }
        )
    
    async def _wait_for_first_success(
        self,
        block_name: str,
        strategy: str,
        steps: List[Any],
        task_ids: List[str],
        timeout_seconds: float,
        start_time: float,
    ) -> ParallelExecutionResult:
        """Wait for first successful task completion."""
        from .parallel import ParallelTaskResult, ParallelExecutionResult
        
        completed_results = []
        success_result = None
        
        # Poll for results
        poll_interval = 0.1
        elapsed = 0
        
        while elapsed < timeout_seconds and not success_result:
            for task_id in task_ids:
                try:
                    # Check if task completed
                    result = await self.distributed_queue.get_task_result(task_id, timeout=poll_interval)
                    
                    if result.status == TaskStatus.COMPLETED:
                        success_result = result
                        break
                    elif result.status in [TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        completed_results.append(result)
                        
                except asyncio.TimeoutError:
                    continue
            
            if success_result:
                break
                
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        # Build result
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if success_result:
            task_result = ParallelTaskResult(
                task_id=success_result.task_id,
                status='completed',
                result=success_result.result,
                duration_ms=(
                    (success_result.completed_at - success_result.started_at) * 1000
                    if success_result.started_at and success_result.completed_at
                    else None
                ),
                metadata={'worker_id': success_result.worker_id}
            )
            
            return ParallelExecutionResult(
                block_name=block_name,
                strategy=strategy,
                total_tasks=len(steps),
                completed_tasks=1,
                failed_tasks=len(completed_results),
                cancelled_tasks=len(task_ids) - 1 - len(completed_results),
                results=[task_result],
                final_result=success_result.result,
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=duration_ms,
                overall_status='completed',
                metadata={'execution_type': 'distributed'}
            )
        else:
            return ParallelExecutionResult(
                block_name=block_name,
                strategy=strategy,
                total_tasks=len(steps),
                completed_tasks=0,
                failed_tasks=len(completed_results),
                cancelled_tasks=len(task_ids) - len(completed_results),
                results=[],
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=duration_ms,
                overall_status='timeout',
                error_message="No successful tasks before timeout",
                metadata={'execution_type': 'distributed'}
            )
    
    async def _wait_for_first_completion(
        self,
        block_name: str,
        strategy: str,
        steps: List[Any],
        task_ids: List[str],
        timeout_seconds: float,
        start_time: float,
    ) -> ParallelExecutionResult:
        """Wait for first task completion (success or failure)."""
        from .parallel import ParallelTaskResult, ParallelExecutionResult
        
        # Poll for any completion
        poll_interval = 0.1
        elapsed = 0
        first_result = None
        
        while elapsed < timeout_seconds and not first_result:
            for task_id in task_ids:
                try:
                    result = await self.distributed_queue.get_task_result(task_id, timeout=poll_interval)
                    
                    if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        first_result = result
                        break
                        
                except asyncio.TimeoutError:
                    continue
            
            if first_result:
                break
                
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        # Build result
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if first_result:
            status = 'completed' if first_result.status == TaskStatus.COMPLETED else 'failed'
            
            task_result = ParallelTaskResult(
                task_id=first_result.task_id,
                status=status,
                result=first_result.result,
                error=first_result.error,
                duration_ms=(
                    (first_result.completed_at - first_result.started_at) * 1000
                    if first_result.started_at and first_result.completed_at
                    else None
                ),
                metadata={'worker_id': first_result.worker_id}
            )
            
            return ParallelExecutionResult(
                block_name=block_name,
                strategy=strategy,
                total_tasks=len(steps),
                completed_tasks=1 if status == 'completed' else 0,
                failed_tasks=1 if status == 'failed' else 0,
                cancelled_tasks=len(task_ids) - 1,
                results=[task_result],
                final_result=first_result.result,
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=duration_ms,
                overall_status='completed',
                metadata={'execution_type': 'distributed'}
            )
        else:
            return ParallelExecutionResult(
                block_name=block_name,
                strategy=strategy,
                total_tasks=len(steps),
                completed_tasks=0,
                failed_tasks=0,
                cancelled_tasks=len(task_ids),
                results=[],
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=duration_ms,
                overall_status='timeout',
                error_message="No tasks completed before timeout",
                metadata={'execution_type': 'distributed'}
            )
    
    def _calculate_worker_usage(self, task_results: List[Any]) -> Dict[str, int]:
        """Calculate worker usage statistics."""
        worker_usage = {}
        
        for task_result in task_results:
            worker_id = task_result.metadata.get('worker_id')
            if worker_id:
                worker_usage[worker_id] = worker_usage.get(worker_id, 0) + 1
        
        return worker_usage
    
    async def register_worker(
        self,
        worker_id: str,
        worker_type: str = 'namel3ss_worker',
        capabilities: Optional[List[str]] = None,
        max_concurrent_tasks: int = 1,
    ) -> None:
        """Register a distributed worker."""
        if not self.distributed_queue:
            await self.initialize()
        
        if self.distributed_queue:
            await self.distributed_queue.register_worker(
                worker_id=worker_id,
                worker_type=worker_type,
                capabilities=set(capabilities or []),
                max_concurrent_tasks=max_concurrent_tasks,
            )
    
    async def start_worker(
        self,
        worker_id: str,
        step_executor: Callable[[Any, Dict[str, Any]], Any],
    ) -> None:
        """Start a distributed worker."""
        if not self.distributed_queue:
            raise RuntimeError("Distributed queue not initialized")
        
        async def distributed_task_handler(task: DistributedTask) -> Any:
            """Handle distributed tasks by executing steps."""
            payload = task.payload
            step_data = payload['step_data']
            context = payload['execution_context']
            
            # Deserialize step
            step = self._deserialize_step(step_data)
            
            # Execute step using the provided executor
            return await step_executor(step, context)
        
        await self.distributed_queue.start_worker(worker_id, distributed_task_handler)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        local_stats = self.local_executor.get_execution_stats()
        
        combined_stats = {
            'distributed': self.execution_stats.copy(),
            'local': local_stats,
            'combined': {
                'total_executions': (
                    self.execution_stats['total_executions'] + 
                    local_stats.get('total_executions', 0)
                ),
                'distribution_ratio': (
                    self.execution_stats['distributed_executions'] / 
                    max(1, self.execution_stats['total_executions'])
                ),
            }
        }
        
        if self.distributed_queue:
            # Add queue status if available
            try:
                # Note: This would need to be made async in real implementation
                combined_stats['queue_status'] = {'available': True}
            except:
                combined_stats['queue_status'] = {'available': False}
        
        return combined_stats


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_distributed_executor(
    broker_type: str = 'redis',
    broker_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> DistributedParallelExecutor:
    """Create and initialize a distributed parallel executor."""
    executor = DistributedParallelExecutor(
        broker_type=broker_type,
        broker_config=broker_config,
        **kwargs
    )
    
    await executor.initialize()
    return executor


# Global instance for easy access
_global_distributed_executor: Optional[DistributedParallelExecutor] = None


async def get_distributed_executor() -> DistributedParallelExecutor:
    """Get global distributed executor instance."""
    global _global_distributed_executor
    
    if _global_distributed_executor is None:
        _global_distributed_executor = await create_distributed_executor()
    
    return _global_distributed_executor


async def execute_distributed_parallel(
    parallel_block: Union[Dict[str, Any], Any],
    step_executor: Callable[[Any, Dict[str, Any]], Any],
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ParallelExecutionResult:
    """Convenience function for distributed parallel execution."""
    executor = await get_distributed_executor()
    
    return await executor.execute_parallel_block(
        parallel_block, step_executor, context, **kwargs
    )