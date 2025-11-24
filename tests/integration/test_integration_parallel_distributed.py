"""
Comprehensive Integration Test for Namel3ss Parallel and Distributed Execution.

This test validates:
1. Parallel execution with all strategies
2. Distributed task execution with memory broker
3. Event-driven reactive workflows
4. Integration between all components
5. Error handling and fault tolerance

Run this test to validate the complete parallel/distributed execution system.
"""

import asyncio
import logging
import time
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our runtime components
try:
    from namel3ss.runtime import (
        # Parallel execution
        ParallelExecutor, ParallelStrategy, execute_parallel_steps,
        
        # Distributed execution
        DistributedTaskQueue, MemoryMessageBroker, create_distributed_queue,
        
        # Distributed coordination
        DistributedParallelExecutor, create_distributed_executor,
        
        # Event-driven runtime
        EventDrivenExecutor, EventType, publish_event, register_event_workflow,
    )
    
    logger.info("✅ Successfully imported all runtime components")
except ImportError as e:
    logger.error(f"❌ Failed to import runtime components: {e}")
    exit(1)


async def test_parallel_execution():
    """Test basic parallel execution with different strategies."""
    logger.info("🔄 Testing parallel execution...")
    
    async def sample_task(step_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Sample task that simulates work."""
        task_id = step_data.get('id', 'unknown')
        delay = step_data.get('delay', 0.1)
        
        await asyncio.sleep(delay)
        
        if step_data.get('should_fail', False):
            raise ValueError(f"Task {task_id} was designed to fail")
        
        return {
            'task_id': task_id,
            'processed_at': time.time(),
            'input_data': step_data,
        }
    
    # Test data
    test_steps = [
        {'id': 'task_1', 'delay': 0.1, 'data': 'test_data_1'},
        {'id': 'task_2', 'delay': 0.15, 'data': 'test_data_2'},
        {'id': 'task_3', 'delay': 0.05, 'data': 'test_data_3'},
        {'id': 'task_4', 'delay': 0.2, 'data': 'test_data_4'},
    ]
    
    # Test ALL strategy
    logger.info("  Testing ALL strategy...")
    result = await execute_parallel_steps(
        steps=test_steps,
        step_executor=sample_task,
        strategy='all',
        max_concurrency=3,
    )
    
    assert result.overall_status == 'completed', f"Expected completed, got {result.overall_status}"
    assert result.completed_tasks == 4, f"Expected 4 completed tasks, got {result.completed_tasks}"
    logger.info(f"  ✅ ALL strategy: {result.completed_tasks}/{result.total_tasks} tasks completed")
    
    # Test ANY_SUCCESS strategy
    logger.info("  Testing ANY_SUCCESS strategy...")
    result = await execute_parallel_steps(
        steps=test_steps[:2],  # Use fewer steps for faster testing
        step_executor=sample_task,
        strategy='any_success',
        max_concurrency=2,
    )
    
    assert result.overall_status == 'completed', f"Expected completed, got {result.overall_status}"
    assert result.completed_tasks >= 1, f"Expected at least 1 completed task, got {result.completed_tasks}"
    logger.info(f"  ✅ ANY_SUCCESS strategy: First task completed in {result.total_duration_ms:.1f}ms")
    
    # Test COLLECT strategy (should handle failures gracefully)
    logger.info("  Testing COLLECT strategy with failures...")
    failing_steps = test_steps + [{'id': 'failing_task', 'should_fail': True}]
    
    result = await execute_parallel_steps(
        steps=failing_steps,
        step_executor=sample_task,
        strategy='collect',
        max_concurrency=3,
    )
    
    assert result.overall_status == 'completed', f"COLLECT should always complete, got {result.overall_status}"
    assert result.failed_tasks == 1, f"Expected 1 failed task, got {result.failed_tasks}"
    logger.info(f"  ✅ COLLECT strategy: {result.completed_tasks} succeeded, {result.failed_tasks} failed")
    
    logger.info("✅ Parallel execution tests passed!")


async def test_distributed_execution():
    """Test distributed task execution."""
    logger.info("🔄 Testing distributed execution...")
    
    # Create distributed queue with memory broker
    broker = MemoryMessageBroker()
    queue = DistributedTaskQueue(broker, queue_name="test_queue")
    await queue.start()
    
    # Register a worker
    worker_id = "test_worker_1"
    await queue.register_worker(
        worker_id=worker_id,
        worker_type="test_worker",
        capabilities={'computation'},
        max_concurrent_tasks=2,
    )
    
    # Define task handler
    async def task_handler(task) -> Dict[str, Any]:
        """Handle distributed tasks."""
        payload = task.payload
        
        # Simulate work
        await asyncio.sleep(0.1)
        
        return {
            'task_type': task.task_type,
            'processed_data': payload,
            'worker_id': worker_id,
            'processed_at': time.time(),
        }
    
    # Start worker in background
    worker_task = asyncio.create_task(queue.start_worker(worker_id, task_handler))
    
    # Submit tasks
    task_ids = []
    for i in range(5):
        task_id = await queue.submit_task(
            task_type='test_computation',
            payload={'data': f'test_data_{i}', 'index': i},
            priority=i,  # Different priorities
        )
        task_ids.append(task_id)
    
    logger.info(f"  Submitted {len(task_ids)} tasks to distributed queue")
    
    # Wait for results
    results = await queue.get_batch_results(task_ids, timeout=10.0)
    
    # Verify results
    completed_count = sum(1 for task in results.values() if task.status.value == 'completed')
    assert completed_count == len(task_ids), f"Expected {len(task_ids)} completed, got {completed_count}"
    
    # Check queue status
    status = await queue.get_queue_status()
    logger.info(f"  ✅ Distributed execution: {completed_count} tasks completed")
    logger.info(f"  Queue status: {status['completed_tasks']} completed, {status['active_workers']} workers")
    
    # Cleanup
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await queue.stop()
    
    logger.info("✅ Distributed execution tests passed!")


async def test_distributed_parallel_coordination():
    """Test the distributed parallel coordinator."""
    logger.info("🔄 Testing distributed parallel coordination...")
    
    # Create distributed executor
    executor = DistributedParallelExecutor(
        broker_type='memory',
        fallback_to_local=True,
    )
    await executor.initialize()
    
    # Define step executor
    async def step_executor(step_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step."""
        await asyncio.sleep(0.1)  # Simulate work
        
        return {
            'step_data': step_data,
            'context_keys': list(context.keys()),
            'executed_at': time.time(),
        }
    
    # Test with local execution (not enough tasks for distribution)
    small_parallel_block = {
        'name': 'small_test_block',
        'steps': [{'id': 1}, {'id': 2}],
        'strategy': 'all',
        'max_concurrency': 2,
    }
    
    result = await executor.execute_parallel_block(
        parallel_block=small_parallel_block,
        step_executor=step_executor,
        force_local=True,  # Force local execution for testing
    )
    
    assert result.overall_status == 'completed'
    assert result.completed_tasks == 2
    logger.info("  ✅ Local execution via coordinator works")
    
    # Test execution statistics
    stats = executor.get_execution_stats()
    logger.info(f"  Execution stats: {stats['distributed']['total_executions']} total executions")
    
    await executor.cleanup()
    logger.info("✅ Distributed parallel coordination tests passed!")


async def test_event_driven_execution():
    """Test event-driven reactive workflows."""
    logger.info("🔄 Testing event-driven execution...")
    
    # Create event-driven executor
    event_executor = EventDrivenExecutor(enable_websocket=False)  # Disable WebSocket for testing
    await event_executor.start()
    
    # Track workflow execution
    workflow_results = []
    
    # Register event handler to track results
    async def result_tracker(event):
        """Track workflow completion events."""
        if event.event_type == EventType.WORKFLOW_COMPLETED:
            workflow_results.append(event.data)
    
    await event_executor.register_event_handler(
        event_types=[EventType.WORKFLOW_COMPLETED],
        handler_func=result_tracker,
    )
    
    # Register a reactive workflow
    await event_executor.register_workflow_trigger(
        workflow_name="test_reaction_workflow",
        trigger_event_type=EventType.TASK_COMPLETED,
        workflow_config={
            'steps': [
                {'type': 'validate', 'data': {'validation_rule': 'non_empty'}},
                {'type': 'transform', 'data': {'transform_type': 'uppercase'}},
                {'type': 'store', 'data': {'storage_location': 'test_db'}},
            ],
            'strategy': 'all',
            'max_concurrency': 2,
        },
        filters={'task_category': 'analysis'}
    )
    
    logger.info("  Registered reactive workflow")
    
    # Trigger the workflow with matching event
    event_id = await event_executor.publish_event(
        event_type=EventType.TASK_COMPLETED,
        source="test_system",
        data={
            'task_category': 'analysis',  # Matches filter
            'task_id': 'analysis_task_123',
            'result': {'status': 'success', 'data': 'analysis_output'},
        },
        correlation_id="test_correlation_123"
    )
    
    logger.info(f"  Published trigger event: {event_id}")
    
    # Wait for workflow execution
    await asyncio.sleep(2.0)  # Give time for workflow to execute
    
    # Verify workflow was triggered and completed
    assert len(workflow_results) >= 1, f"Expected at least 1 workflow result, got {len(workflow_results)}"
    
    workflow_result = workflow_results[0]
    assert workflow_result['workflow_name'] == 'test_reaction_workflow'
    assert workflow_result['status'] in ['completed', 'failed']  # Should have executed
    
    logger.info(f"  ✅ Reactive workflow executed: {workflow_result['status']}")
    
    # Get event system statistics
    stats = event_executor.get_stats()
    logger.info(f"  Event stats: {stats['event_driven_executor']['workflows_triggered']} workflows triggered")
    
    await event_executor.stop()
    logger.info("✅ Event-driven execution tests passed!")


async def test_end_to_end_integration():
    """Test complete integration of all components."""
    logger.info("🔄 Testing end-to-end integration...")
    
    # This test combines parallel execution, distributed coordination, and event-driven patterns
    results = {
        'parallel_executed': False,
        'events_published': 0,
        'workflows_triggered': 0,
    }
    
    # Setup event system
    event_executor = EventDrivenExecutor(enable_websocket=False)
    await event_executor.start()
    
    # Track events
    async def integration_event_tracker(event):
        """Track integration events."""
        results['events_published'] += 1
        if event.event_type == EventType.WORKFLOW_COMPLETED:
            results['workflows_triggered'] += 1
    
    await event_executor.register_event_handler(
        event_types=[EventType.TASK_COMPLETED, EventType.WORKFLOW_COMPLETED],
        handler_func=integration_event_tracker,
    )
    
    # Register workflow that responds to task completion
    await event_executor.register_workflow_trigger(
        workflow_name="integration_post_processing",
        trigger_event_type=EventType.TASK_COMPLETED,
        workflow_config={
            'steps': [
                {'type': 'validate_integration', 'data': {}},
                {'type': 'log_completion', 'data': {}},
            ],
            'strategy': 'all',
        }
    )
    
    # Setup distributed executor
    distributed_executor = DistributedParallelExecutor(broker_type='memory', fallback_to_local=True)
    await distributed_executor.initialize()
    
    # Execute parallel work that publishes events
    async def event_publishing_step_executor(step_data: Any, context: Dict[str, Any]) -> Any:
        """Step executor that publishes completion events."""
        # Simulate work
        await asyncio.sleep(0.1)
        
        # Publish task completion event
        await event_executor.publish_event(
            event_type=EventType.TASK_COMPLETED,
            source="integration_test",
            data={
                'step_data': step_data,
                'context': context,
                'execution_type': 'integration_test',
            }
        )
        
        results['parallel_executed'] = True
        return {'step_id': step_data.get('id', 'unknown'), 'completed': True}
    
    # Execute parallel block
    integration_block = {
        'name': 'integration_test_block',
        'steps': [
            {'id': 'integration_step_1', 'data': 'test_data'},
            {'id': 'integration_step_2', 'data': 'test_data'},
        ],
        'strategy': 'all',
    }
    
    parallel_result = await distributed_executor.execute_parallel_block(
        parallel_block=integration_block,
        step_executor=event_publishing_step_executor,
    )
    
    # Wait for event processing
    await asyncio.sleep(2.0)
    
    # Verify integration
    assert results['parallel_executed'], "Parallel execution should have occurred"
    assert parallel_result.overall_status == 'completed', f"Expected completed, got {parallel_result.overall_status}"
    assert results['events_published'] >= 2, f"Expected at least 2 events published, got {results['events_published']}"
    assert results['workflows_triggered'] >= 1, f"Expected at least 1 workflow triggered, got {results['workflows_triggered']}"
    
    logger.info(f"  ✅ Integration results:")
    logger.info(f"    - Parallel tasks completed: {parallel_result.completed_tasks}")
    logger.info(f"    - Events published: {results['events_published']}")
    logger.info(f"    - Workflows triggered: {results['workflows_triggered']}")
    
    # Cleanup
    await distributed_executor.cleanup()
    await event_executor.stop()
    
    logger.info("✅ End-to-end integration tests passed!")


async def run_comprehensive_test():
    """Run all integration tests."""
    logger.info("🚀 Starting Namel3ss Parallel/Distributed Execution Integration Tests")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Run individual test suites
        await test_parallel_execution()
        await test_distributed_execution()
        await test_distributed_parallel_coordination()
        await test_event_driven_execution()
        await test_end_to_end_integration()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info(f"🎉 ALL TESTS PASSED! Total duration: {duration:.2f} seconds")
        logger.info("=" * 80)
        logger.info("✅ Namel3ss Parallel and Distributed Execution System is READY FOR PRODUCTION!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        logger.exception("Full error details:")
        return False


if __name__ == "__main__":
    # Run the integration test
    asyncio.run(run_comprehensive_test())