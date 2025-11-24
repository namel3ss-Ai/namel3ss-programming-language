"""
Standalone Security Integration Test for Namel3ss.

Validates the complete security framework without external dependencies.
"""

import asyncio
import time
import uuid
import sys
import os
from typing import Dict, List, Any

# Add the project root to the path so we can import namel3ss modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import security framework
from namel3ss.runtime.security import (
    SecurityManager, SecurityContext, CapabilityBasedValidator,
    PermissionLevel, SecurityAction, ResourceType, Capability,
    WorkerSecurityPolicy, InsufficientPermissionsError,
    create_execution_context, DEFAULT_CAPABILITIES
)

# Import execution engines
from namel3ss.runtime.parallel import ParallelExecutor
from namel3ss.runtime.distributed import DistributedTaskQueue, MemoryMessageBroker


class MockStepExecutor:
    """Mock step executor for testing."""
    
    def __init__(self, should_fail: bool = False, execution_time: float = 0.1):
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.call_count = 0
        self.calls = []
    
    async def __call__(self, step: Any, context: Dict[str, Any] = None) -> Any:
        """Execute a step."""
        self.call_count += 1
        self.calls.append({'step': step, 'context': context, 'timestamp': time.time()})
        
        await asyncio.sleep(self.execution_time)
        
        if self.should_fail:
            raise RuntimeError(f"Mock step failed: {step}")
        
        return f"result_for_{step}"


async def test_security_context_creation():
    """Test security context creation and validation."""
    print("🔐 Testing security context creation...")
    
    security_manager = SecurityManager(
        encryption_enabled=False,  # Disable encryption for testing
        audit_enabled=True,
        strict_mode=True,
    )
    
    context = await security_manager.create_security_context(
        user_id="test_user",
        permission_level=PermissionLevel.READ_WRITE,
        capabilities=['execute_basic', 'access_data'],
        resource_quotas={'memory': 1000},
    )
    
    assert context.user_id == "test_user"
    assert context.permission_level == PermissionLevel.READ_WRITE
    assert len(context.capabilities) == 2
    assert context.has_capability('execute_basic')
    assert context.can_perform_action(SecurityAction.EXECUTE_TASK, ResourceType.COMPUTE)
    assert not context.can_perform_action(SecurityAction.ADMIN_OPERATION, ResourceType.COMPUTE)
    
    print("  ✅ Security context created successfully")
    print(f"  ✅ User: {context.user_id}, Permission: {context.permission_level.value}")
    print(f"  ✅ Capabilities: {[cap.name for cap in context.capabilities]}")
    
    return security_manager, context


async def test_capability_validation(security_manager):
    """Test capability-based validation."""
    print("\n🔒 Testing capability validation...")
    
    validator = security_manager.validator
    
    # Test basic execution validation
    context = await security_manager.create_security_context(
        user_id="user1",
        permission_level=PermissionLevel.READ_WRITE,
        capabilities=['execute_basic'],
    )
    
    execution_request = {
        'name': 'test_execution',
        'action_type': SecurityAction.EXECUTE_TASK,
        'resource_type': ResourceType.COMPUTE,
    }
    
    is_valid = await validator.validate_execution_context(context, execution_request)
    assert is_valid
    print("  ✅ Basic execution validation passed")
    
    # Test insufficient capabilities
    admin_request = {
        'name': 'admin_task',
        'action_type': SecurityAction.ADMIN_OPERATION,
        'resource_type': ResourceType.COMPUTE,
    }
    
    is_valid = await validator.validate_execution_context(context, admin_request)
    assert not is_valid
    print("  ✅ Admin operation correctly denied for basic user")


async def test_worker_security_policy(security_manager):
    """Test worker security policy validation."""
    print("\n👷 Testing worker security policies...")
    
    # Create worker policy
    policy = await security_manager.register_worker_security_policy(
        worker_id="worker_1",
        allowed_task_types={'compute', 'data_process'},
        required_capabilities={'execute_basic'},
        resource_limits={'memory': 2000, 'cpu': 2},
        created_by="admin",
    )
    
    # Test valid task assignment
    task_request = {
        'task_type': 'compute',
        'resource_requirements': {'memory': 1000, 'cpu': 1},
    }
    
    is_valid = await security_manager.validate_worker_task_assignment(
        "worker_1", task_request
    )
    assert is_valid
    print("  ✅ Valid task assignment accepted")
    
    # Test invalid task type
    invalid_task = {
        'task_type': 'network_scan',
        'resource_requirements': {'memory': 1000, 'cpu': 1},
    }
    
    is_valid = await security_manager.validate_worker_task_assignment(
        "worker_1", invalid_task
    )
    assert not is_valid
    print("  ✅ Invalid task type correctly rejected")
    
    # Test resource limit exceeded
    resource_heavy_task = {
        'task_type': 'compute',
        'resource_requirements': {'memory': 5000, 'cpu': 1},
    }
    
    is_valid = await security_manager.validate_worker_task_assignment(
        "worker_1", resource_heavy_task
    )
    assert not is_valid
    print("  ✅ Resource limit violation correctly detected")


async def test_parallel_execution_security():
    """Test security integration with parallel execution."""
    print("\n⚡ Testing parallel execution security...")
    
    security_manager = SecurityManager(
        encryption_enabled=False,
        audit_enabled=True,
    )
    
    # Create security context with appropriate permissions
    context = await security_manager.create_security_context(
        user_id="parallel_user",
        permission_level=PermissionLevel.READ_WRITE,
        capabilities=['execute_basic', 'access_data'],
        resource_quotas={'memory': 4000, 'cpu': 8},
    )
    
    # Create parallel executor with security
    executor = ParallelExecutor(
        default_max_concurrency=2,
        enable_tracing=False,
        enable_security=True,
        security_manager=security_manager,
    )
    
    step_executor = MockStepExecutor()
    
    # Test successful execution with valid security context
    parallel_block = {
        'name': 'secure_parallel_test',
        'strategy': 'all',
        'steps': ['step1', 'step2', 'step3'],
        'max_concurrency': 2,
    }
    
    result = await executor.execute_parallel_block(
        parallel_block,
        step_executor,
        security_context=context,
    )
    
    assert result.overall_status == "completed"
    assert result.completed_tasks == 3
    assert step_executor.call_count == 3
    
    print("  ✅ Parallel execution completed with security validation")
    print(f"  ✅ Status: {result.overall_status}")
    print(f"  ✅ Completed tasks: {result.completed_tasks}/{result.total_tasks}")


async def test_insufficient_permissions():
    """Test execution with insufficient permissions."""
    print("\n🚫 Testing insufficient permissions handling...")
    
    security_manager = SecurityManager()
    
    # Create limited security context
    limited_context = await security_manager.create_security_context(
        user_id="limited_user",
        permission_level=PermissionLevel.READ_ONLY,
        capabilities=[],  # No capabilities
    )
    
    executor = ParallelExecutor(
        enable_security=True,
        security_manager=security_manager,
    )
    
    step_executor = MockStepExecutor()
    
    parallel_block = {
        'name': 'restricted_test',
        'strategy': 'all',
        'steps': ['step1'],
    }
    
    # This should fail due to insufficient permissions
    try:
        await executor.execute_parallel_block(
            parallel_block,
            step_executor,
            security_context=limited_context,
        )
        assert False, "Should have raised InsufficientPermissionsError"
    except InsufficientPermissionsError as e:
        print(f"  ✅ Insufficient permissions correctly detected: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        raise


async def test_audit_trail():
    """Test security audit logging."""
    print("\n📋 Testing audit trail...")
    
    security_manager = SecurityManager(audit_enabled=True)
    
    context = await security_manager.create_security_context(
        user_id="audit_test_user",
        permission_level=PermissionLevel.READ_WRITE,
        capabilities=['execute_basic'],
    )
    
    # Perform some operations that generate audit entries
    context.add_audit_entry(
        action="test_action",
        resource="test_resource", 
        result="success"
    )
    
    # Check audit trail
    audit_trail = security_manager.get_audit_trail(user_id="audit_test_user")
    assert len(audit_trail) > 0
    print(f"  ✅ Audit trail contains {len(audit_trail)} entries")
    
    context_events = [event for event in audit_trail if event.get('event_type') == 'context_created']
    assert len(context_events) > 0
    print(f"  ✅ Context creation events: {len(context_events)}")
    
    user_events = [event for event in audit_trail if event.get('user_id') == 'audit_test_user']
    assert len(user_events) > 0
    print(f"  ✅ User-specific events: {len(user_events)}")


async def test_context_propagation():
    """Test security context propagation across nodes."""
    print("\n🌐 Testing context propagation...")
    
    security_manager = SecurityManager(encryption_enabled=False)  # Disable encryption for testing
    
    context = await security_manager.create_security_context(
        user_id="distributed_user",
        permission_level=PermissionLevel.READ_WRITE,
        capabilities=['execute_basic', 'distribute_execution'],
    )
    
    # Propagate context
    payload = await security_manager.propagate_security_context(
        context, "worker_node_1"
    )
    
    assert 'user_id' in payload or 'encrypted' in payload
    print("  ✅ Context propagated successfully")
    
    # Receive context on another node
    received_context = await security_manager.receive_security_context(
        payload, "master_node"
    )
    
    assert received_context.user_id == context.user_id
    assert received_context.permission_level == context.permission_level
    assert len(received_context.capabilities) == len(context.capabilities)
    print("  ✅ Context received and reconstructed correctly")


async def test_resource_quotas():
    """Test resource quota enforcement."""
    print("\n💾 Testing resource quota enforcement...")
    
    security_manager = SecurityManager()
    
    context = await security_manager.create_security_context(
        user_id="quota_test_user",
        permission_level=PermissionLevel.READ_WRITE,
        capabilities=['execute_basic'],
        resource_quotas={'memory': 1000, 'cpu': 2},
    )
    
    # Test within quota
    low_resource_request = {
        'name': 'low_resource_task',
        'action_type': SecurityAction.EXECUTE_TASK,
        'resource_type': ResourceType.COMPUTE,
        'resource_requirements': {'memory': 500, 'cpu': 1},
    }
    
    is_valid = await security_manager.validator.validate_execution_context(
        context, low_resource_request
    )
    assert is_valid
    print("  ✅ Low resource request approved")
    
    # Test exceeding quota
    high_resource_request = {
        'name': 'high_resource_task',
        'action_type': SecurityAction.EXECUTE_TASK,
        'resource_type': ResourceType.COMPUTE,
        'resource_requirements': {'memory': 2000, 'cpu': 1},
    }
    
    is_valid = await security_manager.validator.validate_execution_context(
        context, high_resource_request
    )
    assert not is_valid
    print("  ✅ High resource request correctly denied")


async def run_all_tests():
    """Run all security integration tests."""
    print("🚀 Starting Namel3ss Security Integration Tests...\n")
    
    try:
        # Test 1: Security context creation
        security_manager, context = await test_security_context_creation()
        
        # Test 2: Capability validation
        await test_capability_validation(security_manager)
        
        # Test 3: Worker security policies
        await test_worker_security_policy(security_manager)
        
        # Test 4: Parallel execution security
        await test_parallel_execution_security()
        
        # Test 5: Insufficient permissions
        await test_insufficient_permissions()
        
        # Test 6: Audit trail
        await test_audit_trail()
        
        # Test 7: Context propagation
        await test_context_propagation()
        
        # Test 8: Resource quotas
        await test_resource_quotas()
        
        print("\n🎉 ALL SECURITY INTEGRATION TESTS PASSED! 🎉")
        print("\n📊 Summary of validated security features:")
        print("  ✅ Capability-based access control")
        print("  ✅ Worker security policies")
        print("  ✅ Parallel execution authorization")  
        print("  ✅ Insufficient permission detection")
        print("  ✅ Security audit logging")
        print("  ✅ Context propagation across nodes")
        print("  ✅ Resource quota enforcement")
        print("  ✅ Security integration with execution engines")
        
        print("\n🔐 Security framework is PRODUCTION READY!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)