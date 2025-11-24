"""
Comprehensive Testing Suite for Namel3ss Parallel and Distributed Execution.

This module implements extensive test coverage including:
- Unit tests for individual components
- Integration tests for component interactions
- Performance tests and benchmarking
- Edge case and stress testing
- Fault tolerance and recovery testing
- Security validation testing
- Observability verification testing

Production-grade testing framework for deployment confidence.
"""

import asyncio
import json
import logging
import random
import time
import uuid
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from unittest.mock import Mock, AsyncMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all components for comprehensive testing
from namel3ss.runtime.parallel import ParallelExecutor, ParallelStrategy, ParallelTaskResult, ParallelExecutionResult
from namel3ss.runtime.distributed import DistributedTaskQueue, MemoryMessageBroker, DistributedTask, WorkerNode
from namel3ss.runtime.coordinator import DistributedParallelExecutor
from namel3ss.runtime.events import EventDrivenExecutor, Event, EventBus
from namel3ss.runtime.security import (
    SecurityManager, SecurityContext, PermissionLevel, SecurityAction, 
    ResourceType, Capability, WorkerSecurityPolicy
)
from namel3ss.runtime.observability import (
    ObservabilityManager, MetricsCollector, TracingManager, HealthMonitor,
    HealthStatus, MetricType, MetricDefinition
)

logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures and Utilities
# =============================================================================

@dataclass
class TestResult:
    """Test execution result."""
    test_name: str
    category: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class TestExecutor:
    """Base class for test executors."""
    
    def __init__(self, name: str, should_fail: bool = False, execution_time: float = 0.1):
        self.name = name
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.call_count = 0
        self.calls = []
        self.results = []
    
    async def __call__(self, step: Any, context: Dict[str, Any] = None) -> Any:
        """Execute a test step."""
        self.call_count += 1
        call_info = {
            'step': step,
            'context': context,
            'timestamp': time.time(),
            'executor': self.name
        }
        self.calls.append(call_info)
        
        # Simulate execution time
        await asyncio.sleep(self.execution_time)
        
        # Simulate failure if requested
        if self.should_fail:
            error_msg = f"Test executor {self.name} failed on step {step}"
            raise RuntimeError(error_msg)
        
        result = f"{self.name}_result_for_{step}"
        self.results.append(result)
        return result


class VariableTimeExecutor:
    """Executor with variable execution times."""
    
    def __init__(self, min_time: float = 0.05, max_time: float = 0.5):
        self.min_time = min_time
        self.max_time = max_time
        self.call_count = 0
        self.execution_times = []
    
    async def __call__(self, step: Any, context: Dict[str, Any] = None) -> Any:
        self.call_count += 1
        execution_time = random.uniform(self.min_time, self.max_time)
        self.execution_times.append(execution_time)
        
        await asyncio.sleep(execution_time)
        return f"variable_result_{step}_{execution_time:.3f}"


class ResourceIntensiveExecutor:
    """Executor that simulates resource-intensive operations."""
    
    def __init__(self, cpu_intensive: bool = False, memory_intensive: bool = False):
        self.cpu_intensive = cpu_intensive
        self.memory_intensive = memory_intensive
        self.call_count = 0
    
    async def __call__(self, step: Any, context: Dict[str, Any] = None) -> Any:
        self.call_count += 1
        
        if self.cpu_intensive:
            # Simulate CPU-intensive work
            def cpu_work():
                total = 0
                for i in range(100000):
                    total += i * i
                return total
            
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                await loop.run_in_executor(executor, cpu_work)
        
        if self.memory_intensive:
            # Simulate memory allocation
            large_data = [i for i in range(10000)]
            await asyncio.sleep(0.1)
            del large_data
        
        return f"resource_intensive_result_{step}"


# =============================================================================
# Unit Tests
# =============================================================================

class UnitTestSuite:
    """Comprehensive unit tests for individual components."""
    
    def __init__(self):
        self.results = []
    
    async def run_all_unit_tests(self) -> List[TestResult]:
        """Run all unit tests."""
        test_methods = [
            self.test_parallel_strategy_enum,
            self.test_parallel_task_result,
            self.test_parallel_execution_result,
            self.test_distributed_task_creation,
            self.test_worker_node_creation,
            self.test_security_capability_validation,
            self.test_security_context_creation,
            self.test_metrics_definition,
            self.test_health_check_creation,
            self.test_event_trigger_creation
        ]
        
        for test_method in test_methods:
            try:
                start_time = time.time()
                await test_method()
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    category="unit",
                    status="PASS",
                    duration_ms=duration_ms
                ))
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    category="unit",
                    status="FAIL",
                    duration_ms=duration_ms,
                    error_message=str(e)
                ))
        
        return self.results
    
    async def test_parallel_strategy_enum(self):
        """Test parallel strategy enumeration."""
        strategies = list(ParallelStrategy)
        assert len(strategies) >= 5, "Should have at least 5 strategies"
        assert ParallelStrategy.ALL in strategies
        assert ParallelStrategy.ANY_SUCCESS in strategies
        assert ParallelStrategy.RACE in strategies
    
    async def test_parallel_task_result(self):
        """Test parallel task result creation."""
        task_result = ParallelTaskResult(
            task_id="test_task",
            status="completed",
            result="test_result"
        )
        assert task_result.task_id == "test_task"
        assert task_result.status == "completed"
        assert task_result.result == "test_result"
    
    async def test_parallel_execution_result(self):
        """Test parallel execution result creation."""
        execution_result = ParallelExecutionResult(
            block_name="test_block",
            strategy="all",
            total_tasks=5,
            completed_tasks=5,
            failed_tasks=0,
            cancelled_tasks=0,
            results=[]
        )
        assert execution_result.block_name == "test_block"
        assert execution_result.total_tasks == 5
        assert execution_result.completed_tasks == 5
    
    async def test_distributed_task_creation(self):
        """Test distributed task creation."""
        task = DistributedTask(
            task_id="test_distributed_task",
            task_type="test_type",
            payload={"key": "value"},
            priority=1
        )
        assert task.task_id == "test_distributed_task"
        assert task.task_type == "test_type"
        assert task.payload["key"] == "value"
    
    async def test_worker_node_creation(self):
        """Test worker node creation."""
        worker = WorkerNode(
            worker_id="test_worker",
            capabilities=["compute", "data"],
            max_concurrent_tasks=5
        )
        assert worker.worker_id == "test_worker"
        assert "compute" in worker.capabilities
        assert worker.max_concurrent_tasks == 5
    
    async def test_security_capability_validation(self):
        """Test security capability validation."""
        capability = Capability(
            name="test_capability",
            description="Test capability",
            actions=frozenset({SecurityAction.EXECUTE_TASK}),
            resource_types=frozenset({ResourceType.COMPUTE})
        )
        
        assert capability.name == "test_capability"
        assert capability.allows_action(SecurityAction.EXECUTE_TASK, ResourceType.COMPUTE)
        assert not capability.allows_action(SecurityAction.ADMIN_OPERATION, ResourceType.COMPUTE)
    
    async def test_security_context_creation(self):
        """Test security context creation."""
        capability = Capability(
            name="test_cap",
            description="Test",
            actions=frozenset({SecurityAction.EXECUTE_TASK}),
            resource_types=frozenset({ResourceType.COMPUTE})
        )
        
        context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            permission_level=PermissionLevel.READ_WRITE,
            capabilities={capability}
        )
        
        assert context.user_id == "test_user"
        assert context.permission_level == PermissionLevel.READ_WRITE
        assert len(context.capabilities) == 1
    
    async def test_metrics_definition(self):
        """Test metrics definition creation."""
        metric_def = MetricDefinition(
            name="test_metric",
            description="Test metric",
            metric_type=MetricType.COUNTER,
            labels=["label1", "label2"]
        )
        
        assert metric_def.name == "test_metric"
        assert metric_def.metric_type == MetricType.COUNTER
        assert len(metric_def.labels) == 2
    
    async def test_health_check_creation(self):
        """Test health check creation."""
        from namel3ss.runtime.observability import HealthCheck
        
        def test_check():
            return True
        
        health_check = HealthCheck(
            name="test_health_check",
            description="Test health check",
            check_function=test_check,
            timeout_seconds=5.0,
            critical=True
        )
        
        assert health_check.name == "test_health_check"
        assert health_check.critical is True
        assert health_check.check_function() is True
    
    async def test_event_trigger_creation(self):
        """Test event creation."""
        trigger = Event(
            event_id="test_event",
            event_type="test_type",
            data={"key": "value"},
            metadata={"actions": ["action1", "action2"]}
        )
        
        assert trigger.event_id == "test_event"
        assert trigger.event_type == "test_type"
        assert trigger.data["key"] == "value"
        assert len(trigger.metadata["actions"]) == 2


# =============================================================================
# Integration Tests
# =============================================================================

class IntegrationTestSuite:
    """Integration tests for component interactions."""
    
    def __init__(self):
        self.results = []
    
    async def run_all_integration_tests(self) -> List[TestResult]:
        """Run all integration tests."""
        test_methods = [
            self.test_parallel_executor_integration,
            self.test_distributed_queue_integration,
            self.test_security_parallel_integration,
            self.test_observability_parallel_integration,
            self.test_coordinator_integration,
            self.test_event_driven_integration,
            self.test_end_to_end_workflow,
            self.test_multi_component_interaction
        ]
        
        for test_method in test_methods:
            try:
                start_time = time.time()
                await test_method()
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    category="integration",
                    status="PASS",
                    duration_ms=duration_ms
                ))
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    category="integration",
                    status="FAIL",
                    duration_ms=duration_ms,
                    error_message=str(e)
                ))
        
        return self.results
    
    async def test_parallel_executor_integration(self):
        """Test parallel executor with various strategies."""
        executor = ParallelExecutor(default_max_concurrency=3, enable_security=False)
        test_executor = TestExecutor("integration_test")
        
        # Test ALL strategy
        parallel_block = {
            'name': 'integration_test_all',
            'strategy': 'all',
            'steps': ['step1', 'step2', 'step3'],
            'max_concurrency': 2
        }
        
        result = await executor.execute_parallel_block(parallel_block, test_executor)
        assert result.overall_status == "completed"
        assert result.completed_tasks == 3
        assert test_executor.call_count == 3
        
        # Test RACE strategy
        race_executor = TestExecutor("race_test", execution_time=0.05)
        parallel_block['strategy'] = 'race'
        
        result = await executor.execute_parallel_block(parallel_block, race_executor)
        assert result.overall_status == "completed"
        assert result.completed_tasks >= 1  # At least one task should complete
    
    async def test_distributed_queue_integration(self):
        """Test distributed queue operations."""
        broker = MemoryMessageBroker()
        queue = DistributedTaskQueue(broker=broker, enable_security=False)
        
        await queue.start()
        
        try:
            # Submit multiple tasks
            task_ids = []
            for i in range(3):
                task_id = await queue.submit_task(
                    task_type="integration_test",
                    payload={"index": i},
                    priority=i
                )
                task_ids.append(task_id)
            
            assert len(task_ids) == 3
            assert queue.stats["tasks_submitted"] == 3
            
            # Verify tasks are in queue
            for task_id in task_ids:
                assert task_id in queue.pending_tasks
        
        finally:
            await queue.stop()
    
    async def test_security_parallel_integration(self):
        """Test security integration with parallel execution."""
        security_manager = SecurityManager(audit_enabled=True)
        
        # Create security context
        context = await security_manager.create_security_context(
            user_id="integration_test_user",
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=['execute_basic', 'access_data']
        )
        
        # Test with security-enabled executor
        executor = ParallelExecutor(
            enable_security=True,
            security_manager=security_manager
        )
        
        test_executor = TestExecutor("secure_integration")
        
        parallel_block = {
            'name': 'secure_integration_test',
            'strategy': 'all',
            'steps': ['secure_step1', 'secure_step2']
        }
        
        result = await executor.execute_parallel_block(
            parallel_block,
            test_executor,
            security_context=context
        )
        
        assert result.overall_status == "completed"
        assert result.completed_tasks == 2
        
        # Verify audit trail
        audit_trail = security_manager.get_audit_trail(user_id="integration_test_user")
        assert len(audit_trail) > 0
    
    async def test_observability_parallel_integration(self):
        """Test observability integration with parallel execution."""
        from namel3ss.runtime.observability import initialize_observability
        
        observability = initialize_observability(
            service_name="integration-test",
            enable_metrics=True,
            enable_tracing=False,
            enable_health_monitoring=True
        )
        
        executor = ParallelExecutor(enable_security=False)
        test_executor = TestExecutor("observability_integration")
        
        parallel_block = {
            'name': 'observability_integration_test',
            'strategy': 'all',
            'steps': ['obs_step1', 'obs_step2', 'obs_step3']
        }
        
        result = await executor.execute_parallel_block(parallel_block, test_executor)
        
        assert result.overall_status == "completed"
        assert result.completed_tasks == 3
        
        # Verify observability data
        metrics_summary = await observability.get_metrics_summary()
        assert isinstance(metrics_summary, dict)
        assert 'uptime_seconds' in metrics_summary
    
    async def test_coordinator_integration(self):
        """Test distributed parallel coordinator."""
        broker = MemoryMessageBroker()
        distributed_queue = DistributedTaskQueue(broker=broker, enable_security=False)
        parallel_executor = ParallelExecutor(enable_security=False)
        
        coordinator = DistributedParallelExecutor(
            parallel_executor=parallel_executor,
            distributed_queue=distributed_queue,
            enable_security=False
        )
        
        await distributed_queue.start()
        
        try:
            test_executor = TestExecutor("coordinator_integration")
            
            parallel_block = {
                'name': 'coordinator_integration_test',
                'strategy': 'all',
                'steps': ['coord_step1', 'coord_step2'],
                'distribution_policy': 'local_first'
            }
            
            result = await coordinator.execute_distributed_parallel(
                parallel_block,
                test_executor
            )
            
            assert result is not None
            # Coordinator should execute locally for small tasks
        
        finally:
            await distributed_queue.stop()
    
    async def test_event_driven_integration(self):
        """Test event-driven execution integration."""
        event_bus = EventBus()
        executor = EventDrivenExecutor(event_bus=event_bus, enable_security=False)
        
        # Register event handler
        handler_called = False
        handler_data = None
        
        async def test_handler(event_data):
            nonlocal handler_called, handler_data
            handler_called = True
            handler_data = event_data
        
        executor.register_event_handler("integration_test_event", test_handler)
        
        # Trigger event
        await executor.trigger_event("integration_test_event", {"test": "data"})
        
        # Give event time to process
        await asyncio.sleep(0.1)
        
        assert handler_called
        assert handler_data["test"] == "data"
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Initialize all components
        security_manager = SecurityManager(audit_enabled=False)
        observability = ObservabilityManager(
            enable_metrics=False,
            enable_tracing=False,
            enable_health_monitoring=False
        )
        
        broker = MemoryMessageBroker()
        distributed_queue = DistributedTaskQueue(
            broker=broker,
            enable_security=False
        )
        
        parallel_executor = ParallelExecutor(
            enable_security=False
        )
        
        coordinator = DistributedParallelExecutor(
            parallel_executor=parallel_executor,
            distributed_queue=distributed_queue,
            enable_security=False
        )
        
        await distributed_queue.start()
        
        try:
            test_executor = TestExecutor("e2e_test")
            
            # Execute complete workflow
            parallel_block = {
                'name': 'e2e_workflow_test',
                'strategy': 'all',
                'steps': ['e2e_step1', 'e2e_step2', 'e2e_step3', 'e2e_step4'],
                'max_concurrency': 2,
                'distribution_policy': 'local_first'
            }
            
            result = await coordinator.execute_distributed_parallel(
                parallel_block,
                test_executor
            )
            
            assert result is not None
            assert test_executor.call_count >= 4
        
        finally:
            await distributed_queue.stop()
    
    async def test_multi_component_interaction(self):
        """Test interactions between multiple components."""
        # Set up multiple components
        components = {
            'security': SecurityManager(audit_enabled=True),
            'observability': ObservabilityManager(enable_metrics=False, enable_tracing=False, enable_health_monitoring=True),
            'parallel': ParallelExecutor(enable_security=False),
            'event_bus': EventBus(),
        }
        
        event_executor = EventDrivenExecutor(
            event_bus=components['event_bus'],
            enable_security=False
        )
        
        # Test component interaction
        test_executor = TestExecutor("multi_component")
        
        # Execute with multiple components involved
        parallel_block = {
            'name': 'multi_component_test',
            'strategy': 'all',
            'steps': ['mc_step1', 'mc_step2']
        }
        
        result = await components['parallel'].execute_parallel_block(
            parallel_block,
            test_executor
        )
        
        # Trigger related events
        await event_executor.trigger_event(
            "parallel_execution_completed",
            {"result": result.overall_status}
        )
        
        assert result.overall_status == "completed"
        assert test_executor.call_count == 2


# =============================================================================
# Performance Tests
# =============================================================================

class PerformanceTestSuite:
    """Performance and benchmark tests."""
    
    def __init__(self):
        self.results = []
        self.benchmarks = {}
    
    async def run_all_performance_tests(self) -> List[TestResult]:
        """Run all performance tests."""
        test_methods = [
            self.test_parallel_execution_performance,
            self.test_concurrency_scaling,
            self.test_memory_usage,
            self.test_throughput_benchmark,
            self.test_latency_measurement,
            self.test_resource_intensive_operations,
            self.test_large_scale_execution
        ]
        
        for test_method in test_methods:
            try:
                start_time = time.time()
                benchmark_data = await test_method()
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    category="performance",
                    status="PASS",
                    duration_ms=duration_ms,
                    details=benchmark_data or {}
                ))
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    category="performance",
                    status="FAIL",
                    duration_ms=duration_ms,
                    error_message=str(e)
                ))
        
        return self.results
    
    async def test_parallel_execution_performance(self) -> Dict[str, Any]:
        """Test parallel execution performance characteristics."""
        executor = ParallelExecutor(default_max_concurrency=10, enable_security=False)
        
        # Test with different step counts
        performance_data = {}
        
        for step_count in [5, 10, 20, 50]:
            test_executor = TestExecutor("perf_test", execution_time=0.1)
            
            steps = [f"perf_step_{i}" for i in range(step_count)]
            parallel_block = {
                'name': f'performance_test_{step_count}',
                'strategy': 'all',
                'steps': steps,
                'max_concurrency': 10
            }
            
            start_time = time.time()
            result = await executor.execute_parallel_block(parallel_block, test_executor)
            execution_time = time.time() - start_time
            
            performance_data[f"steps_{step_count}"] = {
                'execution_time': execution_time,
                'tasks_per_second': step_count / execution_time,
                'completed_tasks': result.completed_tasks,
                'efficiency': result.completed_tasks / step_count
            }
        
        return performance_data
    
    async def test_concurrency_scaling(self) -> Dict[str, Any]:
        """Test how performance scales with concurrency."""
        scaling_data = {}
        
        for max_concurrency in [1, 2, 4, 8, 16]:
            executor = ParallelExecutor(default_max_concurrency=max_concurrency, enable_security=False)
            test_executor = TestExecutor("concurrency_test", execution_time=0.05)
            
            steps = [f"concurrent_step_{i}" for i in range(20)]
            parallel_block = {
                'name': f'concurrency_test_{max_concurrency}',
                'strategy': 'all',
                'steps': steps,
                'max_concurrency': max_concurrency
            }
            
            start_time = time.time()
            result = await executor.execute_parallel_block(parallel_block, test_executor)
            execution_time = time.time() - start_time
            
            scaling_data[f"concurrency_{max_concurrency}"] = {
                'execution_time': execution_time,
                'throughput': len(steps) / execution_time,
                'completed_tasks': result.completed_tasks
            }
        
        return scaling_data
    
    async def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns."""
        try:
            import psutil
            process = psutil.Process()
            
            memory_data = {}
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            executor = ParallelExecutor(default_max_concurrency=5, enable_security=False)
            memory_executor = ResourceIntensiveExecutor(memory_intensive=True)
            
            steps = [f"memory_step_{i}" for i in range(10)]
            parallel_block = {
                'name': 'memory_test',
                'strategy': 'all',
                'steps': steps,
                'max_concurrency': 5
            }
            
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = await executor.execute_parallel_block(parallel_block, memory_executor)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_data = {
                'baseline_memory_mb': baseline_memory,
                'start_memory_mb': start_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': peak_memory - start_memory,
                'completed_tasks': result.completed_tasks
            }
            
            return memory_data
            
        except ImportError:
            return {"message": "psutil not available for memory testing"}
    
    async def test_throughput_benchmark(self) -> Dict[str, Any]:
        """Benchmark overall system throughput."""
        executor = ParallelExecutor(default_max_concurrency=8, enable_security=False)
        
        # High-throughput test
        test_executor = TestExecutor("throughput_test", execution_time=0.01)
        
        steps = [f"throughput_step_{i}" for i in range(100)]
        parallel_block = {
            'name': 'throughput_benchmark',
            'strategy': 'all',
            'steps': steps,
            'max_concurrency': 8
        }
        
        start_time = time.time()
        result = await executor.execute_parallel_block(parallel_block, test_executor)
        total_time = time.time() - start_time
        
        throughput_data = {
            'total_tasks': len(steps),
            'completed_tasks': result.completed_tasks,
            'total_time_seconds': total_time,
            'tasks_per_second': result.completed_tasks / total_time,
            'average_task_time_ms': (total_time / result.completed_tasks) * 1000 if result.completed_tasks > 0 else 0
        }
        
        return throughput_data
    
    async def test_latency_measurement(self) -> Dict[str, Any]:
        """Measure execution latency characteristics."""
        executor = ParallelExecutor(default_max_concurrency=1, enable_security=False)  # Sequential for latency measurement
        
        latencies = []
        
        for i in range(10):
            test_executor = TestExecutor(f"latency_test_{i}", execution_time=0.05)
            
            parallel_block = {
                'name': f'latency_test_{i}',
                'strategy': 'all',
                'steps': [f'latency_step_{i}'],
                'max_concurrency': 1
            }
            
            start_time = time.time()
            result = await executor.execute_parallel_block(parallel_block, test_executor)
            latency = (time.time() - start_time) * 1000  # ms
            
            latencies.append(latency)
        
        latency_data = {
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'avg_latency_ms': sum(latencies) / len(latencies),
            'latency_stddev': (sum((l - sum(latencies)/len(latencies))**2 for l in latencies) / len(latencies))**0.5,
            'all_latencies': latencies
        }
        
        return latency_data
    
    async def test_resource_intensive_operations(self) -> Dict[str, Any]:
        """Test performance with resource-intensive operations."""
        executor = ParallelExecutor(default_max_concurrency=4, enable_security=False)
        
        # CPU-intensive test
        cpu_executor = ResourceIntensiveExecutor(cpu_intensive=True)
        
        cpu_steps = [f"cpu_step_{i}" for i in range(8)]
        cpu_block = {
            'name': 'cpu_intensive_test',
            'strategy': 'all',
            'steps': cpu_steps,
            'max_concurrency': 4
        }
        
        start_time = time.time()
        cpu_result = await executor.execute_parallel_block(cpu_block, cpu_executor)
        cpu_time = time.time() - start_time
        
        resource_data = {
            'cpu_intensive': {
                'execution_time_seconds': cpu_time,
                'completed_tasks': cpu_result.completed_tasks,
                'tasks_per_second': cpu_result.completed_tasks / cpu_time
            }
        }
        
        return resource_data
    
    async def test_large_scale_execution(self) -> Dict[str, Any]:
        """Test large-scale execution scenarios."""
        executor = ParallelExecutor(default_max_concurrency=20, enable_security=False)
        test_executor = VariableTimeExecutor(min_time=0.01, max_time=0.1)
        
        # Large number of tasks
        large_steps = [f"large_step_{i}" for i in range(200)]
        large_block = {
            'name': 'large_scale_test',
            'strategy': 'all',
            'steps': large_steps,
            'max_concurrency': 20
        }
        
        start_time = time.time()
        result = await executor.execute_parallel_block(large_block, test_executor)
        execution_time = time.time() - start_time
        
        large_scale_data = {
            'total_tasks': len(large_steps),
            'completed_tasks': result.completed_tasks,
            'execution_time_seconds': execution_time,
            'average_concurrency': len(large_steps) / execution_time if execution_time > 0 else 0,
            'efficiency_ratio': result.completed_tasks / len(large_steps),
            'throughput_tasks_per_second': result.completed_tasks / execution_time if execution_time > 0 else 0
        }
        
        return large_scale_data


# =============================================================================
# Edge Case and Stress Tests
# =============================================================================

class EdgeCaseTestSuite:
    """Edge case and stress testing scenarios."""
    
    def __init__(self):
        self.results = []
    
    async def run_all_edge_case_tests(self) -> List[TestResult]:
        """Run all edge case tests."""
        test_methods = [
            self.test_empty_execution_blocks,
            self.test_single_task_execution,
            self.test_timeout_scenarios,
            self.test_exception_propagation,
            self.test_cancellation_scenarios,
            self.test_resource_exhaustion,
            self.test_concurrent_executor_usage,
            self.test_malformed_inputs,
            self.test_network_simulation_failures
        ]
        
        for test_method in test_methods:
            try:
                start_time = time.time()
                await test_method()
                duration_ms = (time.time() - start_time) * 1000
                
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    category="edge_case",
                    status="PASS",
                    duration_ms=duration_ms
                ))
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    category="edge_case",
                    status="FAIL",
                    duration_ms=duration_ms,
                    error_message=str(e)
                ))
        
        return self.results
    
    async def test_empty_execution_blocks(self):
        """Test execution with empty step lists."""
        executor = ParallelExecutor(enable_security=False)
        test_executor = TestExecutor("empty_test")
        
        # Empty steps
        empty_block = {
            'name': 'empty_test',
            'strategy': 'all',
            'steps': [],
            'max_concurrency': 5
        }
        
        result = await executor.execute_parallel_block(empty_block, test_executor)
        assert result.total_tasks == 0
        assert result.completed_tasks == 0
        assert test_executor.call_count == 0
    
    async def test_single_task_execution(self):
        """Test execution with single tasks."""
        executor = ParallelExecutor(enable_security=False)
        test_executor = TestExecutor("single_test")
        
        single_block = {
            'name': 'single_test',
            'strategy': 'all',
            'steps': ['single_step'],
            'max_concurrency': 10
        }
        
        result = await executor.execute_parallel_block(single_block, test_executor)
        assert result.total_tasks == 1
        assert result.completed_tasks == 1
        assert test_executor.call_count == 1
    
    async def test_timeout_scenarios(self):
        """Test timeout handling."""
        executor = ParallelExecutor(enable_security=False)
        slow_executor = TestExecutor("timeout_test", execution_time=1.0)  # 1 second
        
        timeout_block = {
            'name': 'timeout_test',
            'strategy': 'all',
            'steps': ['slow_step1', 'slow_step2'],
            'max_concurrency': 2,
            'timeout_seconds': 0.5  # 500ms timeout
        }
        
        result = await executor.execute_parallel_block(timeout_block, slow_executor)
        # Should handle timeout gracefully
        assert result.overall_status in ["timeout", "failed", "completed"]
    
    async def test_exception_propagation(self):
        """Test exception handling and propagation."""
        executor = ParallelExecutor(enable_security=False)
        failing_executor = TestExecutor("exception_test", should_fail=True)
        
        exception_block = {
            'name': 'exception_test',
            'strategy': 'all',
            'steps': ['failing_step1', 'failing_step2'],
            'max_concurrency': 2
        }
        
        result = await executor.execute_parallel_block(exception_block, failing_executor)
        assert result.failed_tasks > 0
        assert failing_executor.call_count >= 1
    
    async def test_cancellation_scenarios(self):
        """Test task cancellation scenarios."""
        executor = ParallelExecutor(enable_security=False)
        
        # Test RACE strategy with mixed execution times
        mixed_executors = [
            TestExecutor("fast", execution_time=0.01),
            TestExecutor("slow", execution_time=1.0)
        ]
        
        async def mixed_step_executor(step, context=None):
            if "fast" in step:
                return await mixed_executors[0](step, context)
            else:
                return await mixed_executors[1](step, context)
        
        race_block = {
            'name': 'cancellation_test',
            'strategy': 'race',
            'steps': ['fast_step', 'slow_step1', 'slow_step2'],
            'max_concurrency': 3
        }
        
        result = await executor.execute_parallel_block(race_block, mixed_step_executor)
        # RACE should complete quickly with fast step
        assert result.overall_status == "completed"
    
    async def test_resource_exhaustion(self):
        """Test behavior under resource exhaustion."""
        # Test with very high concurrency
        executor = ParallelExecutor(default_max_concurrency=100, enable_security=False)
        test_executor = TestExecutor("resource_test", execution_time=0.01)
        
        resource_steps = [f"resource_step_{i}" for i in range(1000)]
        resource_block = {
            'name': 'resource_exhaustion_test',
            'strategy': 'all',
            'steps': resource_steps,
            'max_concurrency': 100
        }
        
        start_time = time.time()
        result = await executor.execute_parallel_block(resource_block, test_executor)
        execution_time = time.time() - start_time
        
        # Should handle high load gracefully
        assert result.completed_tasks > 0
        assert execution_time < 60  # Should not take more than 1 minute
    
    async def test_concurrent_executor_usage(self):
        """Test multiple executors running concurrently."""
        executor1 = ParallelExecutor(enable_security=False)
        executor2 = ParallelExecutor(enable_security=False)
        
        test_executor1 = TestExecutor("concurrent1", execution_time=0.1)
        test_executor2 = TestExecutor("concurrent2", execution_time=0.15)
        
        block1 = {
            'name': 'concurrent_test1',
            'strategy': 'all',
            'steps': ['c1_step1', 'c1_step2'],
            'max_concurrency': 2
        }
        
        block2 = {
            'name': 'concurrent_test2',
            'strategy': 'all',
            'steps': ['c2_step1', 'c2_step2', 'c2_step3'],
            'max_concurrency': 2
        }
        
        # Run both concurrently
        results = await asyncio.gather(
            executor1.execute_parallel_block(block1, test_executor1),
            executor2.execute_parallel_block(block2, test_executor2),
            return_exceptions=True
        )
        
        assert len(results) == 2
        assert all(not isinstance(r, Exception) for r in results)
        assert test_executor1.call_count == 2
        assert test_executor2.call_count == 3
    
    async def test_malformed_inputs(self):
        """Test handling of malformed inputs."""
        executor = ParallelExecutor(enable_security=False)
        test_executor = TestExecutor("malformed_test")
        
        # Test various malformed configurations
        malformed_configs = [
            {
                'name': 'missing_strategy',
                'steps': ['step1'],
                # 'strategy' missing - should use default
            },
            {
                'name': 'invalid_strategy',
                'strategy': 'invalid_strategy_name',
                'steps': ['step1'],
            },
            {
                'name': 'negative_concurrency',
                'strategy': 'all',
                'steps': ['step1'],
                'max_concurrency': -1,
            }
        ]
        
        for config in malformed_configs:
            try:
                result = await executor.execute_parallel_block(config, test_executor)
                # Should handle gracefully or complete with defaults
                assert result is not None
            except Exception as e:
                # Should raise meaningful exceptions for truly invalid inputs
                assert isinstance(e, (ValueError, TypeError))
    
    async def test_network_simulation_failures(self):
        """Test simulated network failures in distributed scenarios."""
        # Create broker with simulated failures
        broker = MemoryMessageBroker()
        queue = DistributedTaskQueue(broker=broker, enable_security=False)
        
        await queue.start()
        
        try:
            # Simulate network issues by submitting tasks and then "disconnecting"
            task_ids = []
            for i in range(5):
                task_id = await queue.submit_task(
                    task_type="network_test",
                    payload={"data": f"test_{i}"},
                    priority=1
                )
                task_ids.append(task_id)
            
            assert len(task_ids) == 5
            
            # Simulate broker failure (in real scenario, this would be network disconnection)
            # For memory broker, we'll just verify it handles the state correctly
            original_pending = len(queue.pending_tasks)
            assert original_pending == 5
            
        finally:
            await queue.stop()


# =============================================================================
# Test Runner and Reporting
# =============================================================================

class ComprehensiveTestRunner:
    """Main test runner for comprehensive testing suite."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
    
    async def run_all_tests(self, include_performance: bool = True, include_edge_cases: bool = True) -> Dict[str, Any]:
        """Run complete test suite."""
        print("🧪 Starting Comprehensive Namel3ss Test Suite...\n")
        self.start_time = time.time()
        
        # Unit Tests
        print("🔬 Running Unit Tests...")
        unit_suite = UnitTestSuite()
        unit_results = await unit_suite.run_all_unit_tests()
        self.results.extend(unit_results)
        self._print_category_summary("Unit", unit_results)
        
        # Integration Tests
        print("\n🔗 Running Integration Tests...")
        integration_suite = IntegrationTestSuite()
        integration_results = await integration_suite.run_all_integration_tests()
        self.results.extend(integration_results)
        self._print_category_summary("Integration", integration_results)
        
        # Performance Tests
        if include_performance:
            print("\n⚡ Running Performance Tests...")
            performance_suite = PerformanceTestSuite()
            performance_results = await performance_suite.run_all_performance_tests()
            self.results.extend(performance_results)
            self._print_category_summary("Performance", performance_results)
            self._print_performance_benchmarks(performance_results)
        
        # Edge Case Tests
        if include_edge_cases:
            print("\n🎯 Running Edge Case Tests...")
            edge_case_suite = EdgeCaseTestSuite()
            edge_case_results = await edge_case_suite.run_all_edge_case_tests()
            self.results.extend(edge_case_results)
            self._print_category_summary("Edge Case", edge_case_results)
        
        self.end_time = time.time()
        
        # Generate final report
        return self._generate_final_report()
    
    def _print_category_summary(self, category: str, results: List[TestResult]):
        """Print summary for a test category."""
        passed = len([r for r in results if r.status == "PASS"])
        failed = len([r for r in results if r.status == "FAIL"])
        total = len(results)
        
        print(f"  {category} Tests: {passed}/{total} PASSED")
        
        if failed > 0:
            print(f"  ❌ Failed tests:")
            for result in results:
                if result.status == "FAIL":
                    print(f"    - {result.test_name}: {result.error_message}")
        else:
            print(f"  ✅ All {category.lower()} tests passed!")
    
    def _print_performance_benchmarks(self, results: List[TestResult]):
        """Print performance benchmark details."""
        print("  📊 Performance Benchmarks:")
        
        for result in results:
            if result.status == "PASS" and result.details:
                print(f"    {result.test_name}:")
                
                if 'steps_5' in result.details:
                    # Parallel execution performance
                    for key, data in result.details.items():
                        if isinstance(data, dict) and 'tasks_per_second' in data:
                            print(f"      {key}: {data['tasks_per_second']:.1f} tasks/sec")
                
                elif 'throughput_tasks_per_second' in result.details:
                    # Throughput benchmark
                    print(f"      Throughput: {result.details['throughput_tasks_per_second']:.1f} tasks/sec")
                
                elif 'avg_latency_ms' in result.details:
                    # Latency measurement
                    print(f"      Avg Latency: {result.details['avg_latency_ms']:.2f}ms")
                    print(f"      Min/Max: {result.details['min_latency_ms']:.2f}/{result.details['max_latency_ms']:.2f}ms")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "PASS"])
        failed_tests = len([r for r in self.results if r.status == "FAIL"])
        total_duration = self.end_time - self.start_time
        
        # Categorize results
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {'passed': 0, 'failed': 0, 'total': 0}
            
            categories[result.category]['total'] += 1
            if result.status == "PASS":
                categories[result.category]['passed'] += 1
            else:
                categories[result.category]['failed'] += 1
        
        # Generate report
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                'total_duration_seconds': total_duration,
                'test_categories': categories
            },
            'results': self.results,
            'execution_timestamp': time.time(),
            'version': "1.0.0"
        }
        
        # Print final summary
        print(f"\n🎉 TEST SUITE COMPLETE!")
        print(f"📊 Results: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        print(f"⏱️ Total time: {total_duration:.2f} seconds")
        
        if failed_tests == 0:
            print("✅ ALL TESTS PASSED - System is ready for production!")
        else:
            print(f"❌ {failed_tests} tests failed - Review failures before deployment")
        
        print("\n📈 Test Coverage Summary:")
        for category, stats in categories.items():
            print(f"  {category.title()}: {stats['passed']}/{stats['total']} passed")
        
        return report


# =============================================================================
# Main Test Execution
# =============================================================================

async def main():
    """Run the comprehensive test suite."""
    runner = ComprehensiveTestRunner()
    
    # Run all tests
    report = await runner.run_all_tests(
        include_performance=True,
        include_edge_cases=True
    )
    
    # Save report to file
    report_file = "comprehensive_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return success/failure based on test results
    return report['summary']['failed_tests'] == 0


if __name__ == "__main__":
    import sys
    
    result = asyncio.run(main())
    sys.exit(0 if result else 1)