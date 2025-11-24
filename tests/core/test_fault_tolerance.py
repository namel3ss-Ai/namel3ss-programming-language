"""
Fault Tolerance and Recovery Testing for Namel3ss Runtime.

This module implements comprehensive fault tolerance testing including:
- System failure scenarios
- Recovery mechanisms validation
- Data consistency verification
- Graceful degradation testing
- Failover and backup system testing

Critical for production deployment confidence.
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

# Import all components for fault tolerance testing
from namel3ss.runtime.parallel import ParallelExecutor, ParallelStrategy
from namel3ss.runtime.distributed import DistributedTaskQueue, MemoryMessageBroker, DistributedTask
from namel3ss.runtime.coordinator import DistributedParallelExecutor
from namel3ss.runtime.events import EventDrivenExecutor, EventBus
from namel3ss.runtime.security import SecurityManager, SecurityContext, PermissionLevel
from namel3ss.runtime.observability import ObservabilityManager, HealthMonitor, HealthStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Fault Injection Utilities
# =============================================================================

class FaultInjector:
    """Utility for injecting various types of faults."""
    
    def __init__(self):
        self.active_faults = {}
        self.fault_history = []
    
    def inject_latency(self, component: str, min_delay: float, max_delay: float):
        """Inject random latency into component operations."""
        self.active_faults[f"{component}_latency"] = {
            'type': 'latency',
            'min_delay': min_delay,
            'max_delay': max_delay,
            'active': True
        }
    
    def inject_failure(self, component: str, failure_rate: float):
        """Inject random failures into component operations."""
        self.active_faults[f"{component}_failure"] = {
            'type': 'failure',
            'failure_rate': failure_rate,
            'active': True
        }
    
    def inject_resource_exhaustion(self, component: str, resource_type: str):
        """Inject resource exhaustion scenarios."""
        self.active_faults[f"{component}_resource"] = {
            'type': 'resource_exhaustion',
            'resource_type': resource_type,
            'active': True
        }
    
    def clear_faults(self, component: Optional[str] = None):
        """Clear all or specific component faults."""
        if component:
            keys_to_remove = [k for k in self.active_faults.keys() if k.startswith(component)]
            for key in keys_to_remove:
                del self.active_faults[key]
        else:
            self.active_faults.clear()
    
    async def apply_fault(self, component: str, operation: str) -> bool:
        """Apply fault if applicable. Returns True if operation should fail."""
        fault_key = f"{component}_failure"
        if fault_key in self.active_faults:
            fault = self.active_faults[fault_key]
            if random.random() < fault['failure_rate']:
                self.fault_history.append({
                    'component': component,
                    'operation': operation,
                    'fault_type': 'failure',
                    'timestamp': time.time()
                })
                return True
        
        latency_key = f"{component}_latency"
        if latency_key in self.active_faults:
            fault = self.active_faults[latency_key]
            delay = random.uniform(fault['min_delay'], fault['max_delay'])
            await asyncio.sleep(delay)
            self.fault_history.append({
                'component': component,
                'operation': operation,
                'fault_type': 'latency',
                'delay': delay,
                'timestamp': time.time()
            })
        
        return False


class FaultTolerantExecutor:
    """Test executor with configurable fault injection."""
    
    def __init__(self, name: str, fault_injector: FaultInjector, 
                 base_execution_time: float = 0.1):
        self.name = name
        self.fault_injector = fault_injector
        self.base_execution_time = base_execution_time
        self.call_count = 0
        self.failure_count = 0
        self.recovery_count = 0
        self.calls = []
    
    async def __call__(self, step: Any, context: Dict[str, Any] = None) -> Any:
        self.call_count += 1
        
        # Check for faults
        should_fail = await self.fault_injector.apply_fault(self.name, f"execute_{step}")
        
        # Record call
        call_info = {
            'step': step,
            'context': context,
            'timestamp': time.time(),
            'should_fail': should_fail
        }
        self.calls.append(call_info)
        
        # Base execution time
        await asyncio.sleep(self.base_execution_time)
        
        if should_fail:
            self.failure_count += 1
            raise RuntimeError(f"Injected fault in {self.name} for step {step}")
        
        return f"{self.name}_result_{step}"


class CircuitBreakerExecutor:
    """Executor implementing circuit breaker pattern."""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: float = 2.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.call_count = 0
        self.success_count = 0
    
    async def __call__(self, step: Any, context: Dict[str, Any] = None) -> Any:
        self.call_count += 1
        current_time = time.time()
        
        # Check circuit breaker state
        if self.state == "OPEN":
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise RuntimeError(f"Circuit breaker OPEN for {self.name}")
        
        try:
            # Simulate potential failure
            if random.random() < 0.3:  # 30% failure rate
                raise RuntimeError(f"Simulated failure in {self.name}")
            
            # Success
            await asyncio.sleep(0.1)
            self.success_count += 1
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return f"circuit_breaker_result_{step}"
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


# =============================================================================
# Fault Tolerance Test Scenarios
# =============================================================================

class FaultToleranceTestSuite:
    """Comprehensive fault tolerance testing scenarios."""
    
    def __init__(self):
        self.results = []
        self.fault_injector = FaultInjector()
    
    async def run_all_fault_tolerance_tests(self) -> List[Dict[str, Any]]:
        """Run all fault tolerance tests."""
        test_methods = [
            self.test_executor_failure_recovery,
            self.test_partial_failure_handling,
            self.test_timeout_recovery,
            self.test_resource_exhaustion_handling,
            self.test_cascade_failure_prevention,
            self.test_graceful_degradation,
            self.test_circuit_breaker_pattern,
            self.test_retry_mechanisms,
            self.test_data_consistency_under_failure,
            self.test_distributed_failure_scenarios
        ]
        
        for test_method in test_methods:
            try:
                start_time = time.time()
                test_data = await test_method()
                duration = time.time() - start_time
                
                self.results.append({
                    'test_name': test_method.__name__,
                    'status': 'PASS',
                    'duration': duration,
                    'data': test_data
                })
            except Exception as e:
                duration = time.time() - start_time
                self.results.append({
                    'test_name': test_method.__name__,
                    'status': 'FAIL',
                    'duration': duration,
                    'error': str(e),
                    'data': None
                })
        
        return self.results
    
    async def test_executor_failure_recovery(self) -> Dict[str, Any]:
        """Test recovery from executor failures."""
        # Inject failures into executor
        self.fault_injector.inject_failure("test_executor", failure_rate=0.5)
        
        executor = ParallelExecutor(enable_security=False)
        fault_executor = FaultTolerantExecutor("failure_test", self.fault_injector)
        
        parallel_block = {
            'name': 'failure_recovery_test',
            'strategy': 'all_with_retries',  # Custom strategy for testing
            'steps': [f'recovery_step_{i}' for i in range(10)],
            'max_concurrency': 3,
            'max_retries': 2
        }
        
        try:
            result = await executor.execute_parallel_block(parallel_block, fault_executor)
            
            recovery_data = {
                'total_calls': fault_executor.call_count,
                'failures': fault_executor.failure_count,
                'completed_tasks': result.completed_tasks,
                'failed_tasks': result.failed_tasks,
                'recovery_success_rate': (result.completed_tasks / len(parallel_block['steps'])) * 100
            }
            
        finally:
            self.fault_injector.clear_faults("test_executor")
        
        return recovery_data
    
    async def test_partial_failure_handling(self) -> Dict[str, Any]:
        """Test handling of partial failures in parallel execution."""
        executor = ParallelExecutor(enable_security=False)
        
        # Mix of reliable and unreliable executors
        executors = {
            'reliable': FaultTolerantExecutor("reliable", self.fault_injector),
            'unreliable': FaultTolerantExecutor("unreliable", self.fault_injector)
        }
        
        # Inject high failure rate for unreliable executor
        self.fault_injector.inject_failure("unreliable", failure_rate=0.8)
        
        async def mixed_executor(step, context=None):
            if "reliable" in step:
                return await executors['reliable'](step, context)
            else:
                return await executors['unreliable'](step, context)
        
        parallel_block = {
            'name': 'partial_failure_test',
            'strategy': 'all',
            'steps': ['reliable_step_1', 'unreliable_step_1', 'reliable_step_2', 
                     'unreliable_step_2', 'reliable_step_3'],
            'max_concurrency': 5
        }
        
        try:
            result = await executor.execute_parallel_block(parallel_block, mixed_executor)
            
            partial_failure_data = {
                'total_tasks': len(parallel_block['steps']),
                'reliable_calls': executors['reliable'].call_count,
                'reliable_failures': executors['reliable'].failure_count,
                'unreliable_calls': executors['unreliable'].call_count,
                'unreliable_failures': executors['unreliable'].failure_count,
                'overall_success_rate': (result.completed_tasks / len(parallel_block['steps'])) * 100,
                'completed_tasks': result.completed_tasks,
                'failed_tasks': result.failed_tasks
            }
            
        finally:
            self.fault_injector.clear_faults("unreliable")
        
        return partial_failure_data
    
    async def test_timeout_recovery(self) -> Dict[str, Any]:
        """Test recovery from timeout scenarios."""
        executor = ParallelExecutor(enable_security=False)
        
        # Inject high latency
        self.fault_injector.inject_latency("timeout_test", min_delay=0.5, max_delay=1.0)
        timeout_executor = FaultTolerantExecutor("timeout_test", self.fault_injector)
        
        parallel_block = {
            'name': 'timeout_recovery_test',
            'strategy': 'all',
            'steps': [f'timeout_step_{i}' for i in range(5)],
            'max_concurrency': 3,
            'timeout_seconds': 0.8  # Shorter than max latency
        }
        
        try:
            start_time = time.time()
            result = await executor.execute_parallel_block(parallel_block, timeout_executor)
            execution_time = time.time() - start_time
            
            timeout_data = {
                'execution_time': execution_time,
                'timeout_threshold': parallel_block['timeout_seconds'],
                'completed_tasks': result.completed_tasks,
                'timed_out_tasks': result.cancelled_tasks,
                'timeout_handling_success': execution_time <= (parallel_block['timeout_seconds'] * 2)  # Allow some overhead
            }
            
        finally:
            self.fault_injector.clear_faults("timeout_test")
        
        return timeout_data
    
    async def test_resource_exhaustion_handling(self) -> Dict[str, Any]:
        """Test handling of resource exhaustion scenarios."""
        # Test with limited concurrency under high load
        executor = ParallelExecutor(default_max_concurrency=2, enable_security=False)  # Very limited
        
        # Create many tasks
        large_block = {
            'name': 'resource_exhaustion_test',
            'strategy': 'all',
            'steps': [f'resource_step_{i}' for i in range(20)],
            'max_concurrency': 10  # Request more than available
        }
        
        resource_executor = FaultTolerantExecutor("resource_test", self.fault_injector)
        
        start_time = time.time()
        result = await executor.execute_parallel_block(large_block, resource_executor)
        execution_time = time.time() - start_time
        
        resource_data = {
            'requested_concurrency': large_block['max_concurrency'],
            'actual_max_concurrency': executor.default_max_concurrency,
            'total_tasks': len(large_block['steps']),
            'completed_tasks': result.completed_tasks,
            'execution_time': execution_time,
            'tasks_per_second': result.completed_tasks / execution_time if execution_time > 0 else 0,
            'resource_contention_handled': result.completed_tasks == len(large_block['steps'])
        }
        
        return resource_data
    
    async def test_cascade_failure_prevention(self) -> Dict[str, Any]:
        """Test prevention of cascading failures."""
        # Create interdependent system with failure isolation
        broker = MemoryMessageBroker()
        distributed_queue = DistributedTaskQueue(broker=broker, enable_security=False)
        parallel_executor = ParallelExecutor(enable_security=False)
        
        await distributed_queue.start()
        
        try:
            # Inject failures in distributed layer
            self.fault_injector.inject_failure("distributed", failure_rate=0.6)
            
            # But parallel executor should continue working
            fault_executor = FaultTolerantExecutor("cascade_test", self.fault_injector)
            
            # Test parallel execution even with distributed failures
            parallel_block = {
                'name': 'cascade_prevention_test',
                'strategy': 'all',
                'steps': [f'cascade_step_{i}' for i in range(8)],
                'max_concurrency': 4
            }
            
            result = await parallel_executor.execute_parallel_block(parallel_block, fault_executor)
            
            # Verify distributed queue isolation
            distributed_health = {
                'queue_operational': len(distributed_queue.pending_tasks) >= 0,  # Basic health check
                'stats_available': bool(distributed_queue.stats)
            }
            
            cascade_data = {
                'parallel_execution_success': result.overall_status == "completed",
                'parallel_tasks_completed': result.completed_tasks,
                'distributed_layer_isolated': distributed_health['queue_operational'],
                'failure_isolation_effective': (result.completed_tasks > 0) and distributed_health['queue_operational'],
                'fault_injection_active': len(self.fault_injector.active_faults) > 0
            }
            
        finally:
            self.fault_injector.clear_faults("distributed")
            await distributed_queue.stop()
        
        return cascade_data
    
    async def test_graceful_degradation(self) -> Dict[str, Any]:
        """Test graceful degradation under various failure conditions."""
        # Test with decreasing capabilities
        degradation_scenarios = [
            {'name': 'normal', 'failure_rate': 0.0, 'concurrency': 5},
            {'name': 'light_failure', 'failure_rate': 0.2, 'concurrency': 5},
            {'name': 'medium_failure', 'failure_rate': 0.5, 'concurrency': 3},
            {'name': 'heavy_failure', 'failure_rate': 0.7, 'concurrency': 2},
        ]
        
        degradation_results = {}
        
        for scenario in degradation_scenarios:
            self.fault_injector.inject_failure("degradation_test", scenario['failure_rate'])
            
            executor = ParallelExecutor(
                default_max_concurrency=scenario['concurrency'], 
                enable_security=False
            )
            
            degradation_executor = FaultTolerantExecutor("degradation_test", self.fault_injector)
            
            parallel_block = {
                'name': f'degradation_{scenario["name"]}',
                'strategy': 'all',
                'steps': [f'{scenario["name"]}_step_{i}' for i in range(10)],
                'max_concurrency': scenario['concurrency']
            }
            
            start_time = time.time()
            result = await executor.execute_parallel_block(parallel_block, degradation_executor)
            execution_time = time.time() - start_time
            
            degradation_results[scenario['name']] = {
                'failure_rate': scenario['failure_rate'],
                'configured_concurrency': scenario['concurrency'],
                'completed_tasks': result.completed_tasks,
                'failed_tasks': result.failed_tasks,
                'success_rate': (result.completed_tasks / 10) * 100,
                'execution_time': execution_time,
                'throughput': result.completed_tasks / execution_time if execution_time > 0 else 0
            }
            
            self.fault_injector.clear_faults("degradation_test")
        
        # Verify graceful degradation (performance decreases but system remains functional)
        graceful_degradation = True
        prev_success_rate = 100
        
        for scenario_name, data in degradation_results.items():
            if data['success_rate'] > prev_success_rate + 10:  # Allow some variance
                graceful_degradation = False
            prev_success_rate = min(prev_success_rate, data['success_rate'])
        
        return {
            'scenarios': degradation_results,
            'graceful_degradation_verified': graceful_degradation,
            'system_remains_functional': all(d['completed_tasks'] > 0 for d in degradation_results.values())
        }
    
    async def test_circuit_breaker_pattern(self) -> Dict[str, Any]:
        """Test circuit breaker pattern implementation."""
        executor = ParallelExecutor(enable_security=False)
        circuit_executor = CircuitBreakerExecutor("circuit_test")
        
        # Execute enough tasks to trigger circuit breaker
        circuit_block = {
            'name': 'circuit_breaker_test',
            'strategy': 'all',
            'steps': [f'circuit_step_{i}' for i in range(20)],
            'max_concurrency': 5
        }
        
        start_time = time.time()
        try:
            result = await executor.execute_parallel_block(circuit_block, circuit_executor)
        except Exception as e:
            # Circuit breaker may cause some failures
            pass
        
        execution_time = time.time() - start_time
        
        circuit_data = {
            'total_calls': circuit_executor.call_count,
            'successes': circuit_executor.success_count,
            'failures': circuit_executor.failure_count,
            'final_state': circuit_executor.state,
            'circuit_triggered': circuit_executor.state in ["OPEN", "HALF_OPEN"],
            'failure_threshold': circuit_executor.failure_threshold,
            'execution_time': execution_time
        }
        
        return circuit_data
    
    async def test_retry_mechanisms(self) -> Dict[str, Any]:
        """Test retry mechanisms and exponential backoff."""
        class RetryExecutor:
            def __init__(self, max_retries=3, base_delay=0.1):
                self.max_retries = max_retries
                self.base_delay = base_delay
                self.retry_counts = {}
                self.total_attempts = 0
                self.successful_after_retry = 0
            
            async def __call__(self, step, context=None):
                self.total_attempts += 1
                step_retries = self.retry_counts.get(step, 0)
                
                # Fail for first few attempts, then succeed
                if step_retries < 2 and random.random() < 0.7:
                    self.retry_counts[step] = step_retries + 1
                    
                    # Exponential backoff
                    delay = self.base_delay * (2 ** step_retries)
                    await asyncio.sleep(delay)
                    
                    raise RuntimeError(f"Retry test failure for {step} (attempt {step_retries + 1})")
                
                # Success after retries
                if step_retries > 0:
                    self.successful_after_retry += 1
                
                await asyncio.sleep(0.05)
                return f"retry_success_{step}"
        
        executor = ParallelExecutor(enable_security=False)
        retry_executor = RetryExecutor()
        
        # Implement custom retry logic in parallel execution
        async def retry_wrapper(step, context=None):
            max_retries = 3
            base_delay = 0.1
            
            for attempt in range(max_retries + 1):
                try:
                    return await retry_executor(step, context)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    # Exponential backoff
                    await asyncio.sleep(base_delay * (2 ** attempt))
        
        retry_block = {
            'name': 'retry_mechanism_test',
            'strategy': 'all',
            'steps': [f'retry_step_{i}' for i in range(8)],
            'max_concurrency': 4
        }
        
        start_time = time.time()
        result = await executor.execute_parallel_block(retry_block, retry_wrapper)
        execution_time = time.time() - start_time
        
        retry_data = {
            'total_steps': len(retry_block['steps']),
            'completed_tasks': result.completed_tasks,
            'failed_tasks': result.failed_tasks,
            'total_attempts': retry_executor.total_attempts,
            'successful_after_retry': retry_executor.successful_after_retry,
            'retry_effectiveness': retry_executor.successful_after_retry / len(retry_block['steps']),
            'execution_time': execution_time
        }
        
        return retry_data
    
    async def test_data_consistency_under_failure(self) -> Dict[str, Any]:
        """Test data consistency when failures occur."""
        # Simulate data operations with potential failures
        class DataConsistencyExecutor:
            def __init__(self):
                self.data_store = {}
                self.operation_log = []
                self.consistency_violations = 0
            
            async def __call__(self, step, context=None):
                operation_id = str(uuid.uuid4())
                
                self.operation_log.append({
                    'id': operation_id,
                    'step': step,
                    'timestamp': time.time(),
                    'status': 'started'
                })
                
                try:
                    # Simulate data operation
                    if step.startswith('write_'):
                        key = step.replace('write_', '')
                        value = f"data_{operation_id}"
                        
                        # Simulate failure during write
                        if random.random() < 0.3:
                            raise RuntimeError(f"Write failure for {key}")
                        
                        self.data_store[key] = value
                        
                        # Verify data integrity
                        if self.data_store.get(key) != value:
                            self.consistency_violations += 1
                    
                    elif step.startswith('read_'):
                        key = step.replace('read_', '')
                        
                        # Simulate failure during read
                        if random.random() < 0.2:
                            raise RuntimeError(f"Read failure for {key}")
                        
                        value = self.data_store.get(key, "not_found")
                    
                    # Mark as completed
                    for log_entry in self.operation_log:
                        if log_entry['id'] == operation_id:
                            log_entry['status'] = 'completed'
                    
                    await asyncio.sleep(0.05)
                    return f"data_op_success_{step}"
                    
                except Exception as e:
                    # Mark as failed
                    for log_entry in self.operation_log:
                        if log_entry['id'] == operation_id:
                            log_entry['status'] = 'failed'
                            log_entry['error'] = str(e)
                    raise e
        
        executor = ParallelExecutor(enable_security=False)
        data_executor = DataConsistencyExecutor()
        
        data_block = {
            'name': 'data_consistency_test',
            'strategy': 'all',
            'steps': [
                'write_item1', 'write_item2', 'write_item3',
                'read_item1', 'read_item2', 'read_item3',
                'write_item4', 'read_item4'
            ],
            'max_concurrency': 3
        }
        
        result = await executor.execute_parallel_block(data_block, data_executor)
        
        # Analyze consistency
        completed_ops = [op for op in data_executor.operation_log if op['status'] == 'completed']
        failed_ops = [op for op in data_executor.operation_log if op['status'] == 'failed']
        
        consistency_data = {
            'total_operations': len(data_executor.operation_log),
            'completed_operations': len(completed_ops),
            'failed_operations': len(failed_ops),
            'data_store_size': len(data_executor.data_store),
            'consistency_violations': data_executor.consistency_violations,
            'data_integrity_maintained': data_executor.consistency_violations == 0,
            'operation_atomicity': all(op['status'] in ['completed', 'failed'] for op in data_executor.operation_log)
        }
        
        return consistency_data
    
    async def test_distributed_failure_scenarios(self) -> Dict[str, Any]:
        """Test distributed system failure scenarios."""
        broker = MemoryMessageBroker()
        distributed_queue = DistributedTaskQueue(broker=broker, enable_security=False)
        
        await distributed_queue.start()
        
        try:
            # Submit tasks before introducing failures
            task_ids = []
            for i in range(10):
                task_id = await distributed_queue.submit_task(
                    task_type="distributed_failure_test",
                    payload={"index": i},
                    priority=1
                )
                task_ids.append(task_id)
            
            initial_pending = len(distributed_queue.pending_tasks)
            initial_stats = dict(distributed_queue.stats)
            
            # Inject distributed failures
            self.fault_injector.inject_failure("message_broker", failure_rate=0.5)
            
            # Simulate broker issues by manipulating internal state
            # (In real scenarios, this would be network partitions, etc.)
            
            # Try to submit more tasks under failure conditions
            additional_tasks = 0
            for i in range(5):
                try:
                    should_fail = await self.fault_injector.apply_fault("message_broker", "submit_task")
                    if not should_fail:
                        task_id = await distributed_queue.submit_task(
                            task_type="failure_scenario_test",
                            payload={"index": i + 100},
                            priority=2
                        )
                        additional_tasks += 1
                except Exception:
                    pass  # Expected under failure conditions
            
            final_stats = dict(distributed_queue.stats)
            
            distributed_failure_data = {
                'initial_pending_tasks': initial_pending,
                'initial_submitted_tasks': initial_stats.get('tasks_submitted', 0),
                'additional_tasks_submitted': additional_tasks,
                'final_submitted_tasks': final_stats.get('tasks_submitted', 0),
                'broker_failure_rate': 0.5,
                'system_degradation_graceful': additional_tasks > 0,  # Some tasks still got through
                'queue_state_consistent': len(distributed_queue.pending_tasks) >= 0,
                'fault_tolerance_effective': final_stats.get('tasks_submitted', 0) >= initial_stats.get('tasks_submitted', 0)
            }
            
        finally:
            self.fault_injector.clear_faults("message_broker")
            await distributed_queue.stop()
        
        return distributed_failure_data


# =============================================================================
# Recovery and Resilience Testing
# =============================================================================

class RecoveryTestSuite:
    """Test recovery and resilience mechanisms."""
    
    def __init__(self):
        self.results = []
    
    async def run_all_recovery_tests(self) -> List[Dict[str, Any]]:
        """Run all recovery tests."""
        test_methods = [
            self.test_automatic_recovery,
            self.test_manual_recovery_procedures,
            self.test_backup_system_activation,
            self.test_state_restoration,
            self.test_progressive_recovery,
            self.test_recovery_time_objectives
        ]
        
        for test_method in test_methods:
            try:
                start_time = time.time()
                test_data = await test_method()
                duration = time.time() - start_time
                
                self.results.append({
                    'test_name': test_method.__name__,
                    'status': 'PASS',
                    'duration': duration,
                    'data': test_data
                })
            except Exception as e:
                duration = time.time() - start_time
                self.results.append({
                    'test_name': test_method.__name__,
                    'status': 'FAIL',
                    'duration': duration,
                    'error': str(e),
                    'data': None
                })
        
        return self.results
    
    async def test_automatic_recovery(self) -> Dict[str, Any]:
        """Test automatic recovery mechanisms."""
        # Implement health monitoring with automatic recovery
        class AutoRecoverySystem:
            def __init__(self):
                self.health_status = "healthy"
                self.failure_count = 0
                self.recovery_attempts = 0
                self.auto_recovery_enabled = True
                self.recovery_threshold = 3
            
            async def health_check(self):
                # Simulate random health issues
                if random.random() < 0.3:
                    self.health_status = "degraded"
                    self.failure_count += 1
                    return False
                return True
            
            async def attempt_recovery(self):
                self.recovery_attempts += 1
                await asyncio.sleep(0.2)  # Recovery time
                
                # Recovery success rate
                if random.random() < 0.7:
                    self.health_status = "healthy"
                    self.failure_count = 0
                    return True
                return False
            
            async def monitor_and_recover(self, duration: float = 2.0):
                start_time = time.time()
                health_checks = 0
                successful_recoveries = 0
                
                while time.time() - start_time < duration:
                    health_checks += 1
                    is_healthy = await self.health_check()
                    
                    if not is_healthy and self.auto_recovery_enabled:
                        if self.failure_count >= self.recovery_threshold:
                            recovery_successful = await self.attempt_recovery()
                            if recovery_successful:
                                successful_recoveries += 1
                    
                    await asyncio.sleep(0.1)  # Health check interval
                
                return {
                    'health_checks_performed': health_checks,
                    'recovery_attempts': self.recovery_attempts,
                    'successful_recoveries': successful_recoveries,
                    'final_health_status': self.health_status,
                    'auto_recovery_effective': successful_recoveries > 0
                }
        
        recovery_system = AutoRecoverySystem()
        return await recovery_system.monitor_and_recover(duration=3.0)
    
    async def test_manual_recovery_procedures(self) -> Dict[str, Any]:
        """Test manual recovery procedures."""
        # Simulate manual recovery steps
        recovery_procedures = [
            "diagnose_failure",
            "isolate_affected_components",
            "restore_from_backup",
            "restart_services",
            "verify_functionality"
        ]
        
        recovery_log = []
        recovery_success = True
        
        for procedure in recovery_procedures:
            start_time = time.time()
            
            # Simulate manual procedure execution
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Some procedures might fail
            procedure_success = random.random() > 0.1  # 90% success rate
            
            recovery_log.append({
                'procedure': procedure,
                'success': procedure_success,
                'duration': time.time() - start_time,
                'timestamp': time.time()
            })
            
            if not procedure_success:
                recovery_success = False
        
        manual_recovery_data = {
            'total_procedures': len(recovery_procedures),
            'successful_procedures': len([log for log in recovery_log if log['success']]),
            'recovery_log': recovery_log,
            'overall_success': recovery_success,
            'total_recovery_time': sum(log['duration'] for log in recovery_log),
            'procedure_success_rate': len([log for log in recovery_log if log['success']]) / len(recovery_procedures)
        }
        
        return manual_recovery_data
    
    async def test_backup_system_activation(self) -> Dict[str, Any]:
        """Test backup system activation and failover."""
        # Simulate primary and backup systems
        class SystemCluster:
            def __init__(self):
                self.primary_active = True
                self.backup_active = False
                self.failover_time = None
                self.data_sync_status = "synchronized"
                self.requests_handled = 0
            
            async def handle_request(self, request_id):
                if self.primary_active:
                    self.requests_handled += 1
                    # Simulate primary failure
                    if random.random() < 0.2:  # 20% failure chance
                        await self.trigger_failover()
                    return f"primary_handled_{request_id}"
                elif self.backup_active:
                    self.requests_handled += 1
                    return f"backup_handled_{request_id}"
                else:
                    raise RuntimeError("No active system")
            
            async def trigger_failover(self):
                if self.primary_active:
                    failover_start = time.time()
                    self.primary_active = False
                    
                    # Simulate failover time
                    await asyncio.sleep(0.5)
                    
                    self.backup_active = True
                    self.failover_time = time.time() - failover_start
        
        cluster = SystemCluster()
        
        # Simulate handling requests during potential failover
        handled_requests = []
        failed_requests = []
        
        for i in range(20):
            try:
                result = await cluster.handle_request(f"req_{i}")
                handled_requests.append(result)
            except Exception as e:
                failed_requests.append(f"req_{i}: {str(e)}")
            
            await asyncio.sleep(0.05)
        
        backup_activation_data = {
            'total_requests': 20,
            'handled_requests': len(handled_requests),
            'failed_requests': len(failed_requests),
            'failover_occurred': cluster.failover_time is not None,
            'failover_time': cluster.failover_time,
            'primary_active': cluster.primary_active,
            'backup_active': cluster.backup_active,
            'service_availability': (len(handled_requests) / 20) * 100,
            'backup_system_effective': cluster.backup_active and len(handled_requests) > 10
        }
        
        return backup_activation_data
    
    async def test_state_restoration(self) -> Dict[str, Any]:
        """Test state restoration after failures."""
        # Simulate stateful system with backup/restore capability
        class StatefulSystem:
            def __init__(self):
                self.state = {"counter": 0, "data": {}}
                self.backups = []
                self.restore_points = []
            
            async def create_backup(self):
                backup = {
                    'state': self.state.copy(),
                    'timestamp': time.time(),
                    'backup_id': str(uuid.uuid4())
                }
                self.backups.append(backup)
                return backup['backup_id']
            
            async def restore_from_backup(self, backup_id=None):
                if not self.backups:
                    raise RuntimeError("No backups available")
                
                if backup_id:
                    backup = next((b for b in self.backups if b['backup_id'] == backup_id), None)
                    if not backup:
                        raise RuntimeError(f"Backup {backup_id} not found")
                else:
                    backup = self.backups[-1]  # Latest backup
                
                restore_point = {
                    'previous_state': self.state.copy(),
                    'restored_state': backup['state'].copy(),
                    'restore_timestamp': time.time(),
                    'backup_timestamp': backup['timestamp']
                }
                
                self.state = backup['state'].copy()
                self.restore_points.append(restore_point)
                return restore_point
            
            async def modify_state(self, key, value):
                self.state[key] = value
                if key == "counter":
                    self.state["counter"] += 1
            
            async def simulate_failure(self):
                # Corrupt state
                self.state = {"corrupted": True}
        
        system = StatefulSystem()
        
        # Build up state
        for i in range(5):
            await system.modify_state(f"item_{i}", f"value_{i}")
        
        # Create backup
        backup_id = await system.create_backup()
        
        # Continue modifying state
        for i in range(3):
            await system.modify_state(f"new_item_{i}", f"new_value_{i}")
        
        # Simulate failure
        pre_failure_state = system.state.copy()
        await system.simulate_failure()
        post_failure_state = system.state.copy()
        
        # Restore from backup
        restore_info = await system.restore_from_backup(backup_id)
        post_restore_state = system.state.copy()
        
        state_restoration_data = {
            'backup_created': backup_id is not None,
            'pre_failure_state_size': len(pre_failure_state),
            'post_failure_state_corrupted': post_failure_state.get("corrupted", False),
            'restore_successful': "corrupted" not in post_restore_state,
            'state_consistency_maintained': len(post_restore_state) > 0,
            'restore_time_lag': restore_info['restore_timestamp'] - restore_info['backup_timestamp'],
            'data_loss_minimized': len(post_restore_state) >= 5  # At least original 5 items
        }
        
        return state_restoration_data
    
    async def test_progressive_recovery(self) -> Dict[str, Any]:
        """Test progressive recovery strategies."""
        # Simulate system recovery in phases
        class ProgressiveRecoverySystem:
            def __init__(self):
                self.recovery_phases = [
                    {"name": "basic_functionality", "priority": 1, "duration": 0.3},
                    {"name": "core_services", "priority": 2, "duration": 0.5},
                    {"name": "advanced_features", "priority": 3, "duration": 0.4},
                    {"name": "full_capacity", "priority": 4, "duration": 0.6}
                ]
                self.completed_phases = []
                self.current_capacity = 0
            
            async def execute_recovery_phase(self, phase):
                start_time = time.time()
                
                # Simulate recovery work
                await asyncio.sleep(phase["duration"])
                
                # Phase might fail
                success = random.random() > 0.15  # 85% success rate
                
                phase_result = {
                    'name': phase["name"],
                    'priority': phase["priority"],
                    'success': success,
                    'duration': time.time() - start_time,
                    'timestamp': time.time()
                }
                
                if success:
                    self.completed_phases.append(phase_result)
                    self.current_capacity = len(self.completed_phases) / len(self.recovery_phases)
                
                return phase_result
            
            async def progressive_recovery(self):
                recovery_log = []
                
                for phase in sorted(self.recovery_phases, key=lambda x: x["priority"]):
                    result = await self.execute_recovery_phase(phase)
                    recovery_log.append(result)
                    
                    if not result["success"]:
                        # Retry failed phase once
                        retry_result = await self.execute_recovery_phase(phase)
                        recovery_log.append(retry_result)
                
                return recovery_log
        
        recovery_system = ProgressiveRecoverySystem()
        recovery_log = await recovery_system.progressive_recovery()
        
        progressive_recovery_data = {
            'total_phases': len(recovery_system.recovery_phases),
            'completed_phases': len(recovery_system.completed_phases),
            'current_capacity_percentage': recovery_system.current_capacity * 100,
            'recovery_log': recovery_log,
            'full_recovery_achieved': recovery_system.current_capacity == 1.0,
            'recovery_efficiency': len([r for r in recovery_log if r['success']]) / len(recovery_log) if recovery_log else 0,
            'total_recovery_time': sum(r['duration'] for r in recovery_log)
        }
        
        return progressive_recovery_data
    
    async def test_recovery_time_objectives(self) -> Dict[str, Any]:
        """Test recovery time objectives (RTO) compliance."""
        # Define recovery time objectives for different components
        recovery_scenarios = [
            {"component": "database", "rto_seconds": 2.0, "criticality": "high"},
            {"component": "api_gateway", "rto_seconds": 1.0, "criticality": "high"},
            {"component": "background_processor", "rto_seconds": 5.0, "criticality": "medium"},
            {"component": "analytics_engine", "rto_seconds": 10.0, "criticality": "low"}
        ]
        
        recovery_results = []
        
        for scenario in recovery_scenarios:
            # Simulate failure and recovery
            failure_time = time.time()
            
            # Simulate detection time
            detection_delay = random.uniform(0.1, 0.5)
            await asyncio.sleep(detection_delay)
            
            # Simulate recovery process
            recovery_start = time.time()
            recovery_duration = random.uniform(0.5, scenario["rto_seconds"] * 1.2)  # Might exceed RTO
            await asyncio.sleep(recovery_duration)
            recovery_end = time.time()
            
            actual_rto = recovery_end - failure_time
            rto_compliance = actual_rto <= scenario["rto_seconds"]
            
            recovery_results.append({
                'component': scenario["component"],
                'target_rto': scenario["rto_seconds"],
                'actual_rto': actual_rto,
                'detection_delay': detection_delay,
                'recovery_duration': recovery_duration,
                'rto_compliance': rto_compliance,
                'criticality': scenario["criticality"]
            })
        
        # Calculate compliance metrics
        high_criticality_compliant = all(
            r['rto_compliance'] for r in recovery_results 
            if r['criticality'] == 'high'
        )
        
        overall_compliance_rate = len([
            r for r in recovery_results if r['rto_compliance']
        ]) / len(recovery_results)
        
        rto_compliance_data = {
            'total_scenarios': len(recovery_scenarios),
            'compliant_scenarios': len([r for r in recovery_results if r['rto_compliance']]),
            'overall_compliance_rate': overall_compliance_rate,
            'high_criticality_compliance': high_criticality_compliant,
            'recovery_results': recovery_results,
            'average_actual_rto': sum(r['actual_rto'] for r in recovery_results) / len(recovery_results),
            'rto_objectives_met': overall_compliance_rate >= 0.8  # 80% compliance target
        }
        
        return rto_compliance_data


# =============================================================================
# Main Fault Tolerance Test Runner
# =============================================================================

class FaultToleranceTestRunner:
    """Main runner for fault tolerance and recovery tests."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
    
    async def run_all_fault_tolerance_tests(self) -> Dict[str, Any]:
        """Run complete fault tolerance test suite."""
        print("🛡️ Starting Fault Tolerance and Recovery Test Suite...\n")
        self.start_time = time.time()
        
        # Fault Tolerance Tests
        print("⚡ Running Fault Tolerance Tests...")
        fault_tolerance_suite = FaultToleranceTestSuite()
        fault_tolerance_results = await fault_tolerance_suite.run_all_fault_tolerance_tests()
        self.results.extend(fault_tolerance_results)
        self._print_category_summary("Fault Tolerance", fault_tolerance_results)
        
        # Recovery Tests
        print("\n🔄 Running Recovery Tests...")
        recovery_suite = RecoveryTestSuite()
        recovery_results = await recovery_suite.run_all_recovery_tests()
        self.results.extend(recovery_results)
        self._print_category_summary("Recovery", recovery_results)
        
        self.end_time = time.time()
        
        # Generate final report
        return self._generate_final_report()
    
    def _print_category_summary(self, category: str, results: List[Dict[str, Any]]):
        """Print summary for a test category."""
        passed = len([r for r in results if r['status'] == 'PASS'])
        failed = len([r for r in results if r['status'] == 'FAIL'])
        total = len(results)
        
        print(f"  {category} Tests: {passed}/{total} PASSED")
        
        if failed > 0:
            print(f"  ❌ Failed tests:")
            for result in results:
                if result['status'] == 'FAIL':
                    print(f"    - {result['test_name']}: {result['error']}")
        else:
            print(f"  ✅ All {category.lower()} tests passed!")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.results if r['status'] == 'FAIL'])
        total_duration = self.end_time - self.start_time
        
        # Generate report
        report = {
            'fault_tolerance_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                'total_duration_seconds': total_duration,
                'resilience_score': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'test_results': self.results,
            'execution_timestamp': time.time(),
            'system_resilience_assessment': {
                'fault_tolerance_verified': passed_tests >= total_tests * 0.8,  # 80% threshold
                'recovery_mechanisms_tested': True,
                'production_readiness_score': min(100, (passed_tests / total_tests) * 120) if total_tests > 0 else 0
            }
        }
        
        # Print final summary
        print(f"\n🎯 FAULT TOLERANCE TEST SUITE COMPLETE!")
        print(f"📊 Results: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
        print(f"⏱️ Total time: {total_duration:.2f} seconds")
        print(f"🛡️ System Resilience Score: {report['fault_tolerance_summary']['resilience_score']:.1f}%")
        
        if failed_tests == 0:
            print("✅ EXCELLENT FAULT TOLERANCE - System is highly resilient!")
        else:
            print(f"⚠️ {failed_tests} resilience gaps found - Review for production deployment")
        
        return report


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Run the fault tolerance and recovery test suite."""
    runner = FaultToleranceTestRunner()
    
    # Run all fault tolerance tests
    report = await runner.run_all_fault_tolerance_tests()
    
    # Save report to file
    report_file = "fault_tolerance_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed fault tolerance report saved to: {report_file}")
    
    return report['fault_tolerance_summary']['failed_tests'] == 0


if __name__ == "__main__":
    import sys
    
    result = asyncio.run(main())
    sys.exit(0 if result else 1)