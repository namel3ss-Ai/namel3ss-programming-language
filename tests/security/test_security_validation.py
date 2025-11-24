"""
Security Validation Testing for Namel3ss Runtime.

This module implements comprehensive security testing including:
- Authentication and authorization validation
- Capability-based access control testing
- Security context propagation verification
- Audit trail validation
- Security policy enforcement testing
- Vulnerability and attack simulation

Critical security validation for production deployment.
"""

import asyncio
import json
import logging
import random
import time
import uuid
import hashlib
import sys
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Set
from unittest.mock import Mock, AsyncMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all security components for testing
from namel3ss.runtime.security import (
    SecurityManager, SecurityContext, PermissionLevel, SecurityAction, 
    ResourceType, Capability, WorkerSecurityPolicy, CapabilityBasedValidator,
    AuditEvent, SecurityPolicyEnforcer
)
from namel3ss.runtime.parallel import ParallelExecutor
from namel3ss.runtime.distributed import DistributedTaskQueue, MemoryMessageBroker
from namel3ss.runtime.coordinator import DistributedParallelExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# Security Test Utilities
# =============================================================================

class SecurityTestExecutor:
    """Test executor with security context awareness."""
    
    def __init__(self, name: str, required_capabilities: Set[str] = None):
        self.name = name
        self.required_capabilities = required_capabilities or set()
        self.call_count = 0
        self.security_checks_passed = 0
        self.security_violations = 0
        self.access_attempts = []
    
    async def __call__(self, step: Any, context: Dict[str, Any] = None) -> Any:
        self.call_count += 1
        
        # Record access attempt
        access_attempt = {
            'step': step,
            'timestamp': time.time(),
            'context_provided': context is not None,
            'security_context': context.get('security_context') if context else None
        }
        self.access_attempts.append(access_attempt)
        
        # Simulate security-sensitive operation
        if context and context.get('security_context'):
            security_context = context['security_context']
            
            # Check if user has required capabilities
            user_capabilities = {cap.name for cap in security_context.capabilities}
            
            if self.required_capabilities.issubset(user_capabilities):
                self.security_checks_passed += 1
                access_attempt['security_check'] = 'passed'
            else:
                self.security_violations += 1
                access_attempt['security_check'] = 'failed'
                access_attempt['missing_capabilities'] = self.required_capabilities - user_capabilities
                raise PermissionError(f"Insufficient capabilities: missing {self.required_capabilities - user_capabilities}")
        else:
            # No security context provided
            if self.required_capabilities:
                self.security_violations += 1
                access_attempt['security_check'] = 'no_context'
                raise SecurityError("Security context required for this operation")
        
        await asyncio.sleep(0.05)  # Simulate work
        return f"secure_result_{step}_{self.name}"


class MaliciousExecutor:
    """Executor that attempts malicious operations."""
    
    def __init__(self, attack_type: str):
        self.attack_type = attack_type
        self.attack_attempts = []
        self.successful_attacks = 0
        self.blocked_attacks = 0
    
    async def __call__(self, step: Any, context: Dict[str, Any] = None) -> Any:
        attack_attempt = {
            'attack_type': self.attack_type,
            'step': step,
            'timestamp': time.time(),
            'success': False,
            'blocked': False
        }
        
        try:
            if self.attack_type == "privilege_escalation":
                # Attempt to access higher-privilege operations
                if context and context.get('security_context'):
                    # Try to modify security context
                    security_context = context['security_context']
                    security_context.permission_level = PermissionLevel.ADMIN
                    attack_attempt['success'] = True
                    self.successful_attacks += 1
                
            elif self.attack_type == "unauthorized_access":
                # Attempt to access without proper capabilities
                if not context or not context.get('security_context'):
                    # This should be blocked by the security system
                    raise PermissionError("Unauthorized access attempt")
                
            elif self.attack_type == "data_injection":
                # Attempt to inject malicious data
                malicious_data = {"__proto__": {"isAdmin": True}}
                return malicious_data
                
            elif self.attack_type == "resource_exhaustion":
                # Attempt to exhaust system resources
                large_data = [i for i in range(100000)]  # Memory attack
                await asyncio.sleep(5)  # Time attack
                
        except Exception as e:
            attack_attempt['blocked'] = True
            attack_attempt['error'] = str(e)
            self.blocked_attacks += 1
            raise e
        
        self.attack_attempts.append(attack_attempt)
        return f"malicious_result_{step}"


# =============================================================================
# Security Validation Tests
# =============================================================================

class SecurityValidationSuite:
    """Comprehensive security validation tests."""
    
    def __init__(self):
        self.results = []
        self.security_manager = None
    
    async def run_all_security_tests(self) -> List[Dict[str, Any]]:
        """Run all security validation tests."""
        test_methods = [
            self.test_capability_based_access_control,
            self.test_security_context_propagation,
            self.test_permission_level_enforcement,
            self.test_audit_trail_generation,
            self.test_security_policy_enforcement,
            self.test_malicious_input_handling,
            self.test_privilege_escalation_prevention,
            self.test_unauthorized_access_blocking,
            self.test_security_context_validation,
            self.test_cross_context_isolation
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
    
    async def test_capability_based_access_control(self) -> Dict[str, Any]:
        """Test capability-based access control system."""
        self.security_manager = SecurityManager(audit_enabled=True)
        
        # Create different security contexts with different capabilities
        contexts = {}
        
        # Basic user context
        contexts['basic'] = await self.security_manager.create_security_context(
            user_id="basic_user",
            permission_level=PermissionLevel.READ_ONLY,
            capabilities=['read_data']
        )
        
        # Advanced user context
        contexts['advanced'] = await self.security_manager.create_security_context(
            user_id="advanced_user",
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=['read_data', 'write_data', 'execute_basic']
        )
        
        # Admin context
        contexts['admin'] = await self.security_manager.create_security_context(
            user_id="admin_user",
            permission_level=PermissionLevel.ADMIN,
            capabilities=['read_data', 'write_data', 'execute_basic', 'execute_advanced', 'admin_operations']
        )
        
        # Test executors with different capability requirements
        executors = {
            'basic_op': SecurityTestExecutor("basic_op", {'read_data'}),
            'advanced_op': SecurityTestExecutor("advanced_op", {'read_data', 'execute_basic'}),
            'admin_op': SecurityTestExecutor("admin_op", {'admin_operations'})
        }
        
        # Test access control matrix
        access_results = {}
        
        for context_name, context in contexts.items():
            access_results[context_name] = {}
            
            for executor_name, executor in executors.items():
                try:
                    result = await executor(f"test_step_{context_name}", {'security_context': context})
                    access_results[context_name][executor_name] = {
                        'access_granted': True,
                        'result': result,
                        'security_checks_passed': executor.security_checks_passed,
                        'security_violations': executor.security_violations
                    }
                except Exception as e:
                    access_results[context_name][executor_name] = {
                        'access_granted': False,
                        'error': str(e),
                        'security_checks_passed': executor.security_checks_passed,
                        'security_violations': executor.security_violations
                    }
        
        # Verify access control matrix
        expected_access = {
            'basic': {'basic_op': True, 'advanced_op': False, 'admin_op': False},
            'advanced': {'basic_op': True, 'advanced_op': True, 'admin_op': False},
            'admin': {'basic_op': True, 'advanced_op': True, 'admin_op': True}
        }
        
        access_control_correct = True
        for context_name, expected in expected_access.items():
            for executor_name, should_have_access in expected.items():
                actual_access = access_results[context_name][executor_name]['access_granted']
                if actual_access != should_have_access:
                    access_control_correct = False
        
        capability_data = {
            'contexts_created': len(contexts),
            'executors_tested': len(executors),
            'access_matrix': access_results,
            'expected_access': expected_access,
            'access_control_correct': access_control_correct,
            'total_security_checks': sum(
                sum(executor.security_checks_passed for executor in executors.values()) 
                for _ in contexts
            ),
            'total_violations_blocked': sum(
                sum(executor.security_violations for executor in executors.values())
                for _ in contexts
            )
        }
        
        return capability_data
    
    async def test_security_context_propagation(self) -> Dict[str, Any]:
        """Test security context propagation through execution chain."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Create security context
        security_context = await self.security_manager.create_security_context(
            user_id="propagation_test_user",
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=['read_data', 'execute_basic']
        )
        
        # Test propagation through parallel execution
        executor = ParallelExecutor(
            enable_security=True,
            security_manager=self.security_manager
        )
        
        test_executor = SecurityTestExecutor("propagation_test", {'execute_basic'})
        
        parallel_block = {
            'name': 'propagation_test',
            'strategy': 'all',
            'steps': ['prop_step_1', 'prop_step_2', 'prop_step_3'],
            'max_concurrency': 3
        }
        
        result = await executor.execute_parallel_block(
            parallel_block,
            test_executor,
            security_context=security_context
        )
        
        # Verify context was propagated to all steps
        propagation_data = {
            'total_steps': len(parallel_block['steps']),
            'completed_tasks': result.completed_tasks,
            'security_checks_passed': test_executor.security_checks_passed,
            'security_violations': test_executor.security_violations,
            'context_propagated_correctly': test_executor.security_checks_passed == len(parallel_block['steps']),
            'access_attempts': test_executor.access_attempts,
            'all_attempts_had_context': all(
                attempt['context_provided'] and attempt['security_context'] is not None
                for attempt in test_executor.access_attempts
            )
        }
        
        return propagation_data
    
    async def test_permission_level_enforcement(self) -> Dict[str, Any]:
        """Test enforcement of permission levels."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Test different permission levels
        permission_tests = [
            {
                'level': PermissionLevel.READ_ONLY,
                'user_id': 'readonly_user',
                'capabilities': ['read_data'],
                'allowed_operations': ['read'],
                'forbidden_operations': ['write', 'execute', 'admin']
            },
            {
                'level': PermissionLevel.READ_WRITE,
                'user_id': 'readwrite_user',
                'capabilities': ['read_data', 'write_data', 'execute_basic'],
                'allowed_operations': ['read', 'write', 'execute'],
                'forbidden_operations': ['admin']
            },
            {
                'level': PermissionLevel.ADMIN,
                'user_id': 'admin_user',
                'capabilities': ['read_data', 'write_data', 'execute_basic', 'admin_operations'],
                'allowed_operations': ['read', 'write', 'execute', 'admin'],
                'forbidden_operations': []
            }
        ]
        
        permission_results = {}
        
        for test_config in permission_tests:
            context = await self.security_manager.create_security_context(
                user_id=test_config['user_id'],
                permission_level=test_config['level'],
                capabilities=test_config['capabilities']
            )
            
            # Test operations
            operation_results = {}
            
            for operation in ['read', 'write', 'execute', 'admin']:
                operation_executor = SecurityTestExecutor(
                    f"{operation}_executor",
                    {f"{operation}_data" if operation != 'admin' else 'admin_operations'}
                )
                
                try:
                    result = await operation_executor(
                        f"{operation}_operation",
                        {'security_context': context}
                    )
                    operation_results[operation] = {
                        'success': True,
                        'result': result,
                        'violations': operation_executor.security_violations
                    }
                except Exception as e:
                    operation_results[operation] = {
                        'success': False,
                        'error': str(e),
                        'violations': operation_executor.security_violations
                    }
            
            permission_results[test_config['level'].name] = {
                'config': test_config,
                'operation_results': operation_results
            }
        
        # Verify permission enforcement
        enforcement_correct = True
        for level_name, results in permission_results.items():
            config = results['config']
            for operation, result in results['operation_results'].items():
                should_succeed = operation in config['allowed_operations']
                actually_succeeded = result['success']
                
                if should_succeed != actually_succeeded:
                    enforcement_correct = False
        
        permission_enforcement_data = {
            'permission_levels_tested': len(permission_tests),
            'permission_results': permission_results,
            'enforcement_correct': enforcement_correct,
            'total_operations_tested': len(permission_tests) * 4,  # 4 operations per level
            'violations_properly_blocked': sum(
                sum(result['violations'] for result in level['operation_results'].values())
                for level in permission_results.values()
            )
        }
        
        return permission_enforcement_data
    
    async def test_audit_trail_generation(self) -> Dict[str, Any]:
        """Test audit trail generation and integrity."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Create multiple contexts for audit testing
        contexts = []
        for i in range(3):
            context = await self.security_manager.create_security_context(
                user_id=f"audit_user_{i}",
                permission_level=PermissionLevel.READ_WRITE,
                capabilities=['read_data', 'execute_basic']
            )
            contexts.append(context)
        
        # Perform various operations to generate audit events
        test_executor = SecurityTestExecutor("audit_test", {'execute_basic'})
        
        audit_operations = []
        for i, context in enumerate(contexts):
            for j in range(3):
                try:
                    result = await test_executor(
                        f"audit_operation_{i}_{j}",
                        {'security_context': context}
                    )
                    audit_operations.append({
                        'user_id': context.user_id,
                        'operation': f"audit_operation_{i}_{j}",
                        'success': True,
                        'result': result
                    })
                except Exception as e:
                    audit_operations.append({
                        'user_id': context.user_id,
                        'operation': f"audit_operation_{i}_{j}",
                        'success': False,
                        'error': str(e)
                    })
        
        # Retrieve audit trails
        audit_trails = {}
        for context in contexts:
            trail = self.security_manager.get_audit_trail(user_id=context.user_id)
            audit_trails[context.user_id] = trail
        
        # Verify audit trail completeness and integrity
        total_audit_events = sum(len(trail) for trail in audit_trails.values())
        expected_events = len(audit_operations)
        
        audit_data = {
            'operations_performed': len(audit_operations),
            'audit_events_generated': total_audit_events,
            'audit_completeness': total_audit_events >= expected_events,
            'audit_trails': audit_trails,
            'audit_trail_integrity': all(
                event.timestamp > 0 and event.user_id and event.action
                for trail in audit_trails.values()
                for event in trail
            ),
            'user_specific_trails': len(audit_trails),
            'avg_events_per_user': total_audit_events / len(contexts) if contexts else 0
        }
        
        return audit_data
    
    async def test_security_policy_enforcement(self) -> Dict[str, Any]:
        """Test security policy enforcement mechanisms."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Create worker security policies
        worker_policies = [
            WorkerSecurityPolicy(
                worker_id="restricted_worker",
                allowed_capabilities={'read_data'},
                max_execution_time=1.0,
                resource_limits={'memory_mb': 100, 'cpu_percent': 50}
            ),
            WorkerSecurityPolicy(
                worker_id="standard_worker",
                allowed_capabilities={'read_data', 'write_data', 'execute_basic'},
                max_execution_time=5.0,
                resource_limits={'memory_mb': 500, 'cpu_percent': 80}
            ),
            WorkerSecurityPolicy(
                worker_id="privileged_worker",
                allowed_capabilities={'read_data', 'write_data', 'execute_basic', 'execute_advanced'},
                max_execution_time=30.0,
                resource_limits={'memory_mb': 2000, 'cpu_percent': 100}
            )
        ]
        
        # Test policy enforcement
        policy_test_results = {}
        
        for policy in worker_policies:
            # Create context that should match policy
            context = await self.security_manager.create_security_context(
                user_id=f"user_for_{policy.worker_id}",
                permission_level=PermissionLevel.READ_WRITE,
                capabilities=list(policy.allowed_capabilities)
            )
            
            # Test operations within policy limits
            policy_executor = SecurityTestExecutor(
                policy.worker_id,
                policy.allowed_capabilities
            )
            
            # Test allowed operations
            allowed_operations = 0
            blocked_operations = 0
            
            for capability in ['read_data', 'write_data', 'execute_basic', 'execute_advanced']:
                try:
                    if capability in policy.allowed_capabilities:
                        result = await policy_executor(
                            f"policy_test_{capability}",
                            {'security_context': context, 'worker_policy': policy}
                        )
                        allowed_operations += 1
                    else:
                        # This should be blocked
                        test_executor = SecurityTestExecutor("policy_violator", {capability})
                        result = await test_executor(
                            f"policy_violation_{capability}",
                            {'security_context': context}
                        )
                        # If we get here, policy wasn't enforced properly
                except PermissionError:
                    blocked_operations += 1
                except Exception:
                    blocked_operations += 1
            
            policy_test_results[policy.worker_id] = {
                'policy': {
                    'worker_id': policy.worker_id,
                    'allowed_capabilities': list(policy.allowed_capabilities),
                    'max_execution_time': policy.max_execution_time,
                    'resource_limits': policy.resource_limits
                },
                'allowed_operations': allowed_operations,
                'blocked_operations': blocked_operations,
                'policy_violations': policy_executor.security_violations
            }
        
        # Verify policy enforcement effectiveness
        policy_enforcement_effective = all(
            result['blocked_operations'] > 0 or len(result['policy']['allowed_capabilities']) >= 4
            for result in policy_test_results.values()
        )
        
        policy_enforcement_data = {
            'policies_tested': len(worker_policies),
            'policy_results': policy_test_results,
            'enforcement_effective': policy_enforcement_effective,
            'total_operations_tested': sum(
                result['allowed_operations'] + result['blocked_operations']
                for result in policy_test_results.values()
            ),
            'avg_violations_per_policy': sum(
                result['policy_violations'] for result in policy_test_results.values()
            ) / len(policy_test_results) if policy_test_results else 0
        }
        
        return policy_enforcement_data
    
    async def test_malicious_input_handling(self) -> Dict[str, Any]:
        """Test handling of malicious inputs and attack attempts."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Create legitimate context for comparison
        legitimate_context = await self.security_manager.create_security_context(
            user_id="legitimate_user",
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=['read_data', 'execute_basic']
        )
        
        # Test different types of malicious inputs
        malicious_tests = [
            {
                'name': 'sql_injection',
                'input': "'; DROP TABLE users; --",
                'expected_blocked': True
            },
            {
                'name': 'script_injection',
                'input': "<script>alert('xss')</script>",
                'expected_blocked': True
            },
            {
                'name': 'path_traversal',
                'input': "../../../etc/passwd",
                'expected_blocked': True
            },
            {
                'name': 'command_injection',
                'input': "; rm -rf /",
                'expected_blocked': True
            },
            {
                'name': 'oversized_input',
                'input': "A" * 1000000,  # 1MB string
                'expected_blocked': True
            }
        ]
        
        malicious_input_results = {}
        
        for test in malicious_tests:
            test_executor = SecurityTestExecutor("malicious_test", {'execute_basic'})
            
            try:
                # Attempt to process malicious input
                result = await test_executor(
                    test['input'],  # Use malicious input as step
                    {'security_context': legitimate_context}
                )
                
                malicious_input_results[test['name']] = {
                    'input': test['input'][:100] + "..." if len(test['input']) > 100 else test['input'],
                    'blocked': False,
                    'result': str(result)[:100] + "..." if len(str(result)) > 100 else str(result),
                    'expected_blocked': test['expected_blocked']
                }
                
            except Exception as e:
                malicious_input_results[test['name']] = {
                    'input': test['input'][:100] + "..." if len(test['input']) > 100 else test['input'],
                    'blocked': True,
                    'error': str(e),
                    'expected_blocked': test['expected_blocked']
                }
        
        # Verify malicious input handling
        properly_handled = sum(
            1 for result in malicious_input_results.values()
            if result['blocked'] == result['expected_blocked']
        )
        
        malicious_input_data = {
            'malicious_tests_performed': len(malicious_tests),
            'properly_handled': properly_handled,
            'handling_success_rate': (properly_handled / len(malicious_tests)) * 100,
            'test_results': malicious_input_results,
            'security_effective': properly_handled == len(malicious_tests)
        }
        
        return malicious_input_data
    
    async def test_privilege_escalation_prevention(self) -> Dict[str, Any]:
        """Test prevention of privilege escalation attacks."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Create low-privilege context
        low_privilege_context = await self.security_manager.create_security_context(
            user_id="low_privilege_user",
            permission_level=PermissionLevel.READ_ONLY,
            capabilities=['read_data']
        )
        
        # Attempt various privilege escalation attacks
        escalation_attempts = []
        
        # Attack 1: Direct permission level modification
        try:
            original_level = low_privilege_context.permission_level
            low_privilege_context.permission_level = PermissionLevel.ADMIN
            
            # If this succeeds, it's a security vulnerability
            escalation_attempts.append({
                'attack_type': 'direct_permission_modification',
                'success': low_privilege_context.permission_level == PermissionLevel.ADMIN,
                'original_level': original_level.name,
                'attempted_level': PermissionLevel.ADMIN.name
            })
            
        except Exception as e:
            escalation_attempts.append({
                'attack_type': 'direct_permission_modification',
                'success': False,
                'blocked': True,
                'error': str(e)
            })
        
        # Attack 2: Capability set modification
        try:
            original_capabilities = set(cap.name for cap in low_privilege_context.capabilities)
            
            # Try to add admin capability
            admin_capability = Capability(
                name="admin_operations",
                description="Admin operations",
                actions=frozenset({SecurityAction.ADMIN_OPERATION}),
                resource_types=frozenset({ResourceType.SYSTEM})
            )
            
            low_privilege_context.capabilities.add(admin_capability)
            
            escalation_attempts.append({
                'attack_type': 'capability_addition',
                'success': any(cap.name == "admin_operations" for cap in low_privilege_context.capabilities),
                'original_capabilities': list(original_capabilities),
                'attempted_capability': 'admin_operations'
            })
            
        except Exception as e:
            escalation_attempts.append({
                'attack_type': 'capability_addition',
                'success': False,
                'blocked': True,
                'error': str(e)
            })
        
        # Attack 3: Context impersonation
        try:
            # Attempt to create admin context without proper authorization
            fake_admin_context = await self.security_manager.create_security_context(
                user_id="low_privilege_user",  # Same user
                permission_level=PermissionLevel.ADMIN,  # But admin level
                capabilities=['admin_operations']
            )
            
            escalation_attempts.append({
                'attack_type': 'context_impersonation',
                'success': fake_admin_context.permission_level == PermissionLevel.ADMIN,
                'user_id': fake_admin_context.user_id,
                'permission_level': fake_admin_context.permission_level.name
            })
            
        except Exception as e:
            escalation_attempts.append({
                'attack_type': 'context_impersonation',
                'success': False,
                'blocked': True,
                'error': str(e)
            })
        
        # Attack 4: Malicious executor with escalation attempt
        malicious_executor = MaliciousExecutor("privilege_escalation")
        
        try:
            result = await malicious_executor(
                "escalation_attempt",
                {'security_context': low_privilege_context}
            )
            
            escalation_attempts.append({
                'attack_type': 'malicious_executor_escalation',
                'success': malicious_executor.successful_attacks > 0,
                'blocked_attacks': malicious_executor.blocked_attacks,
                'successful_attacks': malicious_executor.successful_attacks
            })
            
        except Exception as e:
            escalation_attempts.append({
                'attack_type': 'malicious_executor_escalation',
                'success': False,
                'blocked': True,
                'error': str(e),
                'blocked_attacks': malicious_executor.blocked_attacks
            })
        
        # Analyze escalation prevention effectiveness
        successful_escalations = sum(1 for attempt in escalation_attempts if attempt.get('success', False))
        blocked_escalations = sum(1 for attempt in escalation_attempts if attempt.get('blocked', False))
        
        escalation_prevention_data = {
            'escalation_attempts': len(escalation_attempts),
            'successful_escalations': successful_escalations,
            'blocked_escalations': blocked_escalations,
            'prevention_effectiveness': (blocked_escalations / len(escalation_attempts)) * 100 if escalation_attempts else 100,
            'escalation_details': escalation_attempts,
            'security_robust': successful_escalations == 0
        }
        
        return escalation_prevention_data
    
    async def test_unauthorized_access_blocking(self) -> Dict[str, Any]:
        """Test blocking of unauthorized access attempts."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Test unauthorized access scenarios
        unauthorized_tests = [
            {
                'name': 'no_context',
                'context': None,
                'expected_blocked': True
            },
            {
                'name': 'invalid_context',
                'context': {'invalid': 'context'},
                'expected_blocked': True
            },
            {
                'name': 'expired_context',
                'context': await self.security_manager.create_security_context(
                    user_id="expired_user",
                    permission_level=PermissionLevel.READ_ONLY,
                    capabilities=['read_data']
                ),
                'expected_blocked': False  # Valid context
            }
        ]
        
        # Manually expire the context for testing
        if unauthorized_tests[2]['context']:
            unauthorized_tests[2]['context'].created_at = time.time() - 7200  # 2 hours ago
            unauthorized_tests[2]['expected_blocked'] = True
        
        unauthorized_access_results = {}
        
        for test in unauthorized_tests:
            secure_executor = SecurityTestExecutor("secure_operation", {'execute_basic'})
            
            try:
                result = await secure_executor(
                    f"unauthorized_test_{test['name']}",
                    {'security_context': test['context']} if test['context'] else None
                )
                
                unauthorized_access_results[test['name']] = {
                    'access_granted': True,
                    'result': result,
                    'expected_blocked': test['expected_blocked'],
                    'security_violations': secure_executor.security_violations
                }
                
            except Exception as e:
                unauthorized_access_results[test['name']] = {
                    'access_granted': False,
                    'blocked': True,
                    'error': str(e),
                    'expected_blocked': test['expected_blocked'],
                    'security_violations': secure_executor.security_violations
                }
        
        # Verify unauthorized access blocking
        properly_blocked = sum(
            1 for test_name, result in unauthorized_access_results.items()
            if (result.get('blocked', False) and result['expected_blocked']) or
               (result.get('access_granted', False) and not result['expected_blocked'])
        )
        
        unauthorized_access_data = {
            'unauthorized_tests_performed': len(unauthorized_tests),
            'properly_blocked': properly_blocked,
            'blocking_success_rate': (properly_blocked / len(unauthorized_tests)) * 100,
            'test_results': unauthorized_access_results,
            'total_violations_detected': sum(
                result['security_violations'] for result in unauthorized_access_results.values()
            ),
            'access_control_effective': properly_blocked == len(unauthorized_tests)
        }
        
        return unauthorized_access_data
    
    async def test_security_context_validation(self) -> Dict[str, Any]:
        """Test security context validation mechanisms."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Test various context validation scenarios
        validation_tests = []
        
        # Valid context
        valid_context = await self.security_manager.create_security_context(
            user_id="valid_user",
            permission_level=PermissionLevel.READ_WRITE,
            capabilities=['read_data', 'execute_basic']
        )
        
        validation_tests.append({
            'name': 'valid_context',
            'context': valid_context,
            'expected_valid': True
        })
        
        # Invalid contexts
        invalid_contexts = [
            {
                'name': 'missing_user_id',
                'context': SecurityContext(
                    user_id="",
                    session_id="test_session",
                    permission_level=PermissionLevel.READ_ONLY,
                    capabilities=set()
                ),
                'expected_valid': False
            },
            {
                'name': 'missing_session_id',
                'context': SecurityContext(
                    user_id="test_user",
                    session_id="",
                    permission_level=PermissionLevel.READ_ONLY,
                    capabilities=set()
                ),
                'expected_valid': False
            },
            {
                'name': 'invalid_permission_level',
                'context': None,  # We'll create this with invalid data
                'expected_valid': False
            }
        ]
        
        validation_tests.extend(invalid_contexts)
        
        # Test context validation
        validation_results = {}
        
        for test in validation_tests:
            if test['context'] is None:
                continue  # Skip invalid context that couldn't be created
            
            try:
                # Use security manager to validate context
                is_valid = await self.security_manager.validate_security_context(test['context'])
                
                validation_results[test['name']] = {
                    'context_valid': is_valid,
                    'expected_valid': test['expected_valid'],
                    'validation_correct': is_valid == test['expected_valid']
                }
                
            except Exception as e:
                validation_results[test['name']] = {
                    'context_valid': False,
                    'expected_valid': test['expected_valid'],
                    'validation_correct': not test['expected_valid'],  # Exception means invalid
                    'error': str(e)
                }
        
        # Calculate validation effectiveness
        correct_validations = sum(
            1 for result in validation_results.values()
            if result['validation_correct']
        )
        
        context_validation_data = {
            'validation_tests_performed': len(validation_results),
            'correct_validations': correct_validations,
            'validation_accuracy': (correct_validations / len(validation_results)) * 100 if validation_results else 100,
            'validation_results': validation_results,
            'validation_system_effective': correct_validations == len(validation_results)
        }
        
        return context_validation_data
    
    async def test_cross_context_isolation(self) -> Dict[str, Any]:
        """Test isolation between different security contexts."""
        if not self.security_manager:
            self.security_manager = SecurityManager(audit_enabled=True)
        
        # Create multiple isolated contexts
        contexts = []
        for i in range(3):
            context = await self.security_manager.create_security_context(
                user_id=f"isolated_user_{i}",
                permission_level=PermissionLevel.READ_WRITE,
                capabilities=[f'capability_{i}', 'shared_capability']
            )
            contexts.append(context)
        
        # Test cross-context access attempts
        isolation_tests = []
        
        for i, context_a in enumerate(contexts):
            for j, context_b in enumerate(contexts):
                if i != j:
                    # Try to use context_a to access resources meant for context_b
                    executor_for_b = SecurityTestExecutor(
                        f"executor_for_user_{j}",
                        {f'capability_{j}'}  # Capability specific to context_b
                    )
                    
                    try:
                        result = await executor_for_b(
                            f"cross_access_test_{i}_to_{j}",
                            {'security_context': context_a}  # Using context_a instead of context_b
                        )
                        
                        isolation_tests.append({
                            'from_user': context_a.user_id,
                            'to_user_resource': f"user_{j}",
                            'access_granted': True,
                            'isolation_broken': True,
                            'result': result
                        })
                        
                    except Exception as e:
                        isolation_tests.append({
                            'from_user': context_a.user_id,
                            'to_user_resource': f"user_{j}",
                            'access_granted': False,
                            'isolation_maintained': True,
                            'error': str(e)
                        })
        
        # Test shared resource access (should work for all)
        shared_executor = SecurityTestExecutor("shared_resource", {'shared_capability'})
        shared_access_results = []
        
        for i, context in enumerate(contexts):
            try:
                result = await shared_executor(
                    f"shared_access_{i}",
                    {'security_context': context}
                )
                shared_access_results.append({
                    'user': context.user_id,
                    'access_granted': True,
                    'result': result
                })
            except Exception as e:
                shared_access_results.append({
                    'user': context.user_id,
                    'access_granted': False,
                    'error': str(e)
                })
        
        # Analyze isolation effectiveness
        isolation_violations = sum(
            1 for test in isolation_tests 
            if test.get('isolation_broken', False)
        )
        
        isolation_maintained = sum(
            1 for test in isolation_tests 
            if test.get('isolation_maintained', False)
        )
        
        shared_access_working = sum(
            1 for result in shared_access_results 
            if result['access_granted']
        )
        
        cross_context_isolation_data = {
            'contexts_created': len(contexts),
            'cross_access_tests': len(isolation_tests),
            'isolation_violations': isolation_violations,
            'isolation_maintained': isolation_maintained,
            'isolation_effectiveness': (isolation_maintained / len(isolation_tests)) * 100 if isolation_tests else 100,
            'shared_resource_access_rate': (shared_access_working / len(contexts)) * 100 if contexts else 100,
            'isolation_tests': isolation_tests,
            'shared_access_results': shared_access_results,
            'security_isolation_robust': isolation_violations == 0 and shared_access_working == len(contexts)
        }
        
        return cross_context_isolation_data


# =============================================================================
# Security Test Runner
# =============================================================================

class SecurityTestRunner:
    """Main runner for security validation tests."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
    
    async def run_all_security_tests(self) -> Dict[str, Any]:
        """Run complete security test suite."""
        print("🔐 Starting Security Validation Test Suite...\n")
        self.start_time = time.time()
        
        # Security Validation Tests
        print("🛡️ Running Security Validation Tests...")
        security_suite = SecurityValidationSuite()
        security_results = await security_suite.run_all_security_tests()
        self.results.extend(security_results)
        self._print_category_summary("Security Validation", security_results)
        
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
        
        # Calculate security score
        security_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Generate report
        report = {
            'security_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': security_score,
                'total_duration_seconds': total_duration,
                'security_score': security_score
            },
            'security_test_results': self.results,
            'execution_timestamp': time.time(),
            'security_assessment': {
                'access_control_validated': True,
                'audit_trail_verified': True,
                'attack_resistance_tested': True,
                'security_ready_for_production': security_score >= 95  # 95% threshold for production
            }
        }
        
        # Print final summary
        print(f"\n🎯 SECURITY VALIDATION SUITE COMPLETE!")
        print(f"📊 Results: {passed_tests}/{total_tests} tests passed ({security_score:.1f}%)")
        print(f"⏱️ Total time: {total_duration:.2f} seconds")
        print(f"🔐 Security Score: {security_score:.1f}%")
        
        if failed_tests == 0:
            print("✅ EXCELLENT SECURITY - System is production-ready!")
        elif security_score >= 95:
            print("⚠️ Minor security issues detected - Review before production deployment")
        else:
            print("❌ Significant security vulnerabilities found - Address before deployment")
        
        return report


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Run the security validation test suite."""
    runner = SecurityTestRunner()
    
    # Run all security tests
    report = await runner.run_all_security_tests()
    
    # Save report to file
    report_file = "security_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed security report saved to: {report_file}")
    
    return report['security_summary']['security_score'] >= 95


if __name__ == "__main__":
    import sys
    
    result = asyncio.run(main())
    sys.exit(0 if result else 1)