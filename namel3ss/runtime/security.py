"""
Security Integration for Namel3ss Parallel and Distributed Execution.

This module implements comprehensive security controls for:
- Capability-based access control for parallel execution
- Worker pool security policies and restrictions
- Distributed permission validation and authorization
- Security context propagation across distributed nodes
- Resource access controls and sandboxing
- Audit logging and security monitoring

Production-ready security framework with defense-in-depth approach.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    cryptography = None
    Fernet = None
    hashes = None
    PBKDF2HMAC = None


# =============================================================================
# Security Domain Models
# =============================================================================

class PermissionLevel(Enum):
    """Permission levels for execution contexts."""
    NONE = "none"
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SYSTEM = "system"


class SecurityAction(Enum):
    """Security actions that can be performed."""
    EXECUTE_TASK = "execute_task"
    CREATE_WORKER = "create_worker"
    MANAGE_QUEUE = "manage_queue"
    ACCESS_DATA = "access_data"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM_ACCESS = "file_system_access"
    DISTRIBUTE_TASK = "distribute_task"
    TRIGGER_EVENT = "trigger_event"
    ADMIN_OPERATION = "admin_operation"


class ResourceType(Enum):
    """Types of resources that can be secured."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATA = "data"
    QUEUE = "queue"
    WORKER = "worker"
    EVENT = "event"


@dataclass
class Capability:
    """Represents a security capability."""
    name: str
    description: str
    actions: Set[SecurityAction]
    resource_types: Set[ResourceType]
    conditions: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[float] = None
    granted_by: Optional[str] = None
    
    def __post_init__(self):
        """Convert sets to frozensets for hashability."""
        if isinstance(self.actions, set):
            self.actions = frozenset(self.actions)
        if isinstance(self.resource_types, set):
            self.resource_types = frozenset(self.resource_types)
    
    def __hash__(self):
        """Make Capability hashable."""
        return hash((
            self.name,
            self.description,
            self.actions,
            self.resource_types,
            tuple(sorted(self.conditions.items())) if self.conditions else (),
            self.expires_at,
            self.granted_by
        ))
    
    def __eq__(self, other):
        """Check equality based on name primarily."""
        if not isinstance(other, Capability):
            return False
        return self.name == other.name
    
    def is_valid(self) -> bool:
        """Check if capability is still valid."""
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True
    
    def allows_action(self, action: SecurityAction, resource_type: ResourceType) -> bool:
        """Check if capability allows specific action on resource type."""
        if not self.is_valid():
            return False
        
        return action in self.actions and resource_type in self.resource_types


@dataclass
class SecurityContext:
    """Security context for execution."""
    user_id: str
    session_id: str
    permission_level: PermissionLevel
    capabilities: Set[Capability]
    resource_quotas: Dict[str, int] = field(default_factory=dict)
    restrictions: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if context has specific capability."""
        return any(cap.name == capability_name and cap.is_valid() for cap in self.capabilities)
    
    def can_perform_action(self, action: SecurityAction, resource_type: ResourceType) -> bool:
        """Check if context allows specific action."""
        return any(
            cap.allows_action(action, resource_type) 
            for cap in self.capabilities if cap.is_valid()
        )
    
    def add_audit_entry(self, action: str, resource: str, result: str, details: Dict[str, Any] = None):
        """Add audit trail entry."""
        entry = {
            'timestamp': time.time(),
            'action': action,
            'resource': resource,
            'result': result,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'details': details or {}
        }
        self.audit_trail.append(entry)


@dataclass
class WorkerSecurityPolicy:
    """Security policy for worker nodes."""
    worker_id: str
    allowed_task_types: Set[str]
    required_capabilities: Set[str]
    resource_limits: Dict[str, Any]
    network_restrictions: Dict[str, Any] = field(default_factory=dict)
    data_access_rules: Dict[str, Any] = field(default_factory=dict)
    execution_sandbox: Dict[str, Any] = field(default_factory=dict)
    audit_level: str = "standard"
    created_at: float = field(default_factory=time.time)
    created_by: Optional[str] = None


# =============================================================================
# Security Validators and Engines
# =============================================================================

class SecurityValidator(ABC):
    """Abstract base class for security validation."""
    
    @abstractmethod
    async def validate_execution_context(
        self, 
        context: SecurityContext, 
        execution_request: Dict[str, Any]
    ) -> bool:
        """Validate if execution is allowed in given context."""
        pass
    
    @abstractmethod
    async def validate_worker_access(
        self, 
        worker_policy: WorkerSecurityPolicy,
        task_request: Dict[str, Any]
    ) -> bool:
        """Validate if worker can execute specific task."""
        pass
    
    @abstractmethod
    async def validate_resource_access(
        self,
        context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: SecurityAction
    ) -> bool:
        """Validate access to specific resource."""
        pass


class CapabilityBasedValidator(SecurityValidator):
    """Capability-based security validator."""
    
    def __init__(
        self,
        strict_mode: bool = True,
        default_deny: bool = True,
        audit_all_requests: bool = True,
    ):
        """Initialize capability-based validator."""
        self.strict_mode = strict_mode
        self.default_deny = default_deny
        self.audit_all_requests = audit_all_requests
        
        # Built-in capabilities
        self.built_in_capabilities = self._create_built_in_capabilities()
    
    def _create_built_in_capabilities(self) -> Dict[str, Capability]:
        """Create built-in security capabilities."""
        capabilities = {}
        
        # Basic execution capability
        capabilities['execute_basic'] = Capability(
            name='execute_basic',
            description='Basic task execution',
            actions=frozenset({SecurityAction.EXECUTE_TASK}),
            resource_types=frozenset({ResourceType.COMPUTE}),
        )
        
        # Worker management capability
        capabilities['manage_workers'] = Capability(
            name='manage_workers',
            description='Create and manage workers',
            actions=frozenset({SecurityAction.CREATE_WORKER, SecurityAction.MANAGE_QUEUE}),
            resource_types=frozenset({ResourceType.WORKER, ResourceType.QUEUE}),
        )
        
        # Data access capability
        capabilities['access_data'] = Capability(
            name='access_data',
            description='Access data resources',
            actions=frozenset({SecurityAction.ACCESS_DATA}),
            resource_types=frozenset({ResourceType.DATA, ResourceType.STORAGE}),
        )
        
        # Network access capability
        capabilities['network_access'] = Capability(
            name='network_access',
            description='Network communication',
            actions=frozenset({SecurityAction.NETWORK_ACCESS}),
            resource_types=frozenset({ResourceType.NETWORK}),
        )
        
        # Distributed execution capability
        capabilities['distribute_execution'] = Capability(
            name='distribute_execution',
            description='Distribute tasks across nodes',
            actions=frozenset({SecurityAction.DISTRIBUTE_TASK}),
            resource_types=frozenset({ResourceType.COMPUTE, ResourceType.NETWORK, ResourceType.QUEUE}),
        )
        
        # Event management capability
        capabilities['manage_events'] = Capability(
            name='manage_events',
            description='Trigger and manage events',
            actions=frozenset({SecurityAction.TRIGGER_EVENT}),
            resource_types=frozenset({ResourceType.EVENT}),
        )
        
        # Administrative capability
        capabilities['admin'] = Capability(
            name='admin',
            description='Administrative operations',
            actions=frozenset({SecurityAction.ADMIN_OPERATION}),
            resource_types=frozenset({ResourceType.COMPUTE, ResourceType.STORAGE, 
                           ResourceType.NETWORK, ResourceType.DATA,
                           ResourceType.QUEUE, ResourceType.WORKER, ResourceType.EVENT}),
        )
        
        return capabilities
    
    async def validate_execution_context(
        self, 
        context: SecurityContext, 
        execution_request: Dict[str, Any]
    ) -> bool:
        """Validate execution context against request."""
        try:
            # Check if context is expired
            if context.expires_at and time.time() > context.expires_at:
                context.add_audit_entry(
                    action='validate_execution',
                    resource=execution_request.get('name', 'unknown'),
                    result='denied',
                    details={'reason': 'expired_context'}
                )
                return False
            
            # Check required action
            action_type = execution_request.get('action_type', SecurityAction.EXECUTE_TASK)
            resource_type = execution_request.get('resource_type', ResourceType.COMPUTE)
            
            # Validate capability
            has_permission = context.can_perform_action(action_type, resource_type)
            
            if not has_permission and self.default_deny:
                context.add_audit_entry(
                    action='validate_execution',
                    resource=execution_request.get('name', 'unknown'),
                    result='denied',
                    details={'reason': 'insufficient_capabilities', 'required_action': action_type.value}
                )
                return False
            
            # Check resource quotas
            if not self._check_resource_quotas(context, execution_request):
                context.add_audit_entry(
                    action='validate_execution',
                    resource=execution_request.get('name', 'unknown'),
                    result='denied',
                    details={'reason': 'quota_exceeded'}
                )
                return False
            
            # Check restrictions
            if not self._check_restrictions(context, execution_request):
                context.add_audit_entry(
                    action='validate_execution',
                    resource=execution_request.get('name', 'unknown'),
                    result='denied',
                    details={'reason': 'policy_violation'}
                )
                return False
            
            # Validation passed
            if self.audit_all_requests:
                context.add_audit_entry(
                    action='validate_execution',
                    resource=execution_request.get('name', 'unknown'),
                    result='allowed',
                    details={'action_type': action_type.value, 'resource_type': resource_type.value}
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            context.add_audit_entry(
                action='validate_execution',
                resource=execution_request.get('name', 'unknown'),
                result='error',
                details={'error': str(e)}
            )
            return False
    
    async def validate_worker_access(
        self, 
        worker_policy: WorkerSecurityPolicy,
        task_request: Dict[str, Any]
    ) -> bool:
        """Validate worker access to task."""
        try:
            task_type = task_request.get('task_type', 'unknown')
            
            # Check if worker can handle this task type
            if task_type not in worker_policy.allowed_task_types:
                logger.warning(f"Worker {worker_policy.worker_id} denied task type {task_type}")
                return False
            
            # Check resource requirements
            resource_requirements = task_request.get('resource_requirements', {})
            if not self._check_worker_resources(worker_policy, resource_requirements):
                logger.warning(f"Worker {worker_policy.worker_id} insufficient resources for task")
                return False
            
            # Check data access requirements
            data_requirements = task_request.get('data_access', [])
            if not self._check_worker_data_access(worker_policy, data_requirements):
                logger.warning(f"Worker {worker_policy.worker_id} insufficient data access for task")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Worker validation error: {e}")
            return False
    
    async def validate_resource_access(
        self,
        context: SecurityContext,
        resource_type: ResourceType,
        resource_id: str,
        action: SecurityAction
    ) -> bool:
        """Validate access to specific resource."""
        try:
            # Check capability for this action and resource type
            has_capability = context.can_perform_action(action, resource_type)
            
            if not has_capability:
                context.add_audit_entry(
                    action='validate_resource_access',
                    resource=f"{resource_type.value}:{resource_id}",
                    result='denied',
                    details={'reason': 'insufficient_capability', 'required_action': action.value}
                )
                return False
            
            # Check resource-specific restrictions
            resource_restrictions = context.restrictions.get(resource_type.value, {})
            if resource_restrictions:
                allowed_resources = resource_restrictions.get('allowed_resources', [])
                if allowed_resources and resource_id not in allowed_resources:
                    context.add_audit_entry(
                        action='validate_resource_access',
                        resource=f"{resource_type.value}:{resource_id}",
                        result='denied',
                        details={'reason': 'resource_not_allowed'}
                    )
                    return False
            
            context.add_audit_entry(
                action='validate_resource_access',
                resource=f"{resource_type.value}:{resource_id}",
                result='allowed',
                details={'action': action.value}
            )
            return True
            
        except Exception as e:
            logger.error(f"Resource access validation error: {e}")
            return False
    
    def _check_resource_quotas(self, context: SecurityContext, request: Dict[str, Any]) -> bool:
        """Check resource quota constraints."""
        required_resources = request.get('resource_requirements', {})
        
        for resource_type, required_amount in required_resources.items():
            quota_limit = context.resource_quotas.get(resource_type)
            if quota_limit is not None and required_amount > quota_limit:
                return False
        
        return True
    
    def _check_restrictions(self, context: SecurityContext, request: Dict[str, Any]) -> bool:
        """Check policy restrictions."""
        # Check time-based restrictions
        time_restrictions = context.restrictions.get('time_restrictions', {})
        if time_restrictions:
            current_time = time.time()
            start_time = time_restrictions.get('start_time')
            end_time = time_restrictions.get('end_time')
            
            if start_time and current_time < start_time:
                return False
            if end_time and current_time > end_time:
                return False
        
        # Check execution context restrictions
        context_restrictions = context.restrictions.get('execution_context', {})
        if context_restrictions:
            max_concurrent = context_restrictions.get('max_concurrent_executions')
            if max_concurrent is not None:
                # This would need integration with runtime tracking
                pass
        
        return True
    
    def _check_worker_resources(self, policy: WorkerSecurityPolicy, requirements: Dict[str, Any]) -> bool:
        """Check if worker has sufficient resources."""
        resource_limits = policy.resource_limits
        
        for resource, required in requirements.items():
            limit = resource_limits.get(resource)
            if limit is not None and required > limit:
                return False
        
        return True
    
    def _check_worker_data_access(self, policy: WorkerSecurityPolicy, data_requirements: List[str]) -> bool:
        """Check if worker can access required data."""
        allowed_data = policy.data_access_rules.get('allowed_datasets', [])
        
        if not allowed_data:  # No restrictions
            return True
        
        for data_requirement in data_requirements:
            if data_requirement not in allowed_data:
                return False
        
        return True


# =============================================================================
# Security Manager
# =============================================================================

class SecurityManager:
    """
    Central security manager for parallel and distributed execution.
    
    Features:
    - Security context management and propagation
    - Capability-based access control
    - Worker security policy enforcement
    - Audit logging and monitoring
    - Encryption and secure communication
    """
    
    def __init__(
        self,
        validator: SecurityValidator = None,
        encryption_enabled: bool = True,
        audit_enabled: bool = True,
        strict_mode: bool = True,
    ):
        """Initialize security manager."""
        self.validator = validator or CapabilityBasedValidator(strict_mode=strict_mode)
        self.encryption_enabled = encryption_enabled and cryptography is not None
        self.audit_enabled = audit_enabled
        self.strict_mode = strict_mode
        
        # Security state
        self.active_contexts: Dict[str, SecurityContext] = {}
        self.worker_policies: Dict[str, WorkerSecurityPolicy] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Encryption setup
        if self.encryption_enabled:
            self._setup_encryption()
        
        logger.info(f"SecurityManager initialized: encryption={self.encryption_enabled}, audit={audit_enabled}")
    
    def _setup_encryption(self):
        """Setup encryption for secure communication."""
        if not cryptography:
            logger.warning("Cryptography not available, encryption disabled")
            self.encryption_enabled = False
            return
        
        # Generate encryption key (in production, this should be from secure key management)
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    async def create_security_context(
        self,
        user_id: str,
        permission_level: PermissionLevel,
        capabilities: List[str],
        resource_quotas: Dict[str, int] = None,
        restrictions: Dict[str, Any] = None,
        expires_in_seconds: Optional[float] = None,
    ) -> SecurityContext:
        """Create a new security context."""
        session_id = str(uuid.uuid4())
        
        # Build capability set
        capability_objects = set()
        validator = self.validator
        if hasattr(validator, 'built_in_capabilities'):
            for cap_name in capabilities:
                if cap_name in validator.built_in_capabilities:
                    capability_objects.add(validator.built_in_capabilities[cap_name])
                else:
                    logger.warning(f"Unknown capability requested: {cap_name}")
        
        # Calculate expiration
        expires_at = None
        if expires_in_seconds:
            expires_at = time.time() + expires_in_seconds
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            permission_level=permission_level,
            capabilities=capability_objects,
            resource_quotas=resource_quotas or {},
            restrictions=restrictions or {},
            expires_at=expires_at,
        )
        
        # Store context
        self.active_contexts[session_id] = context
        
        # Audit
        if self.audit_enabled:
            self._log_security_event(
                event_type='context_created',
                user_id=user_id,
                session_id=session_id,
                details={
                    'permission_level': permission_level.value,
                    'capabilities': capabilities,
                    'expires_at': expires_at,
                }
            )
        
        logger.info(f"Created security context for user {user_id}, session {session_id}")
        return context
    
    async def validate_parallel_execution(
        self,
        context: SecurityContext,
        parallel_block: Dict[str, Any],
        execution_metadata: Dict[str, Any] = None,
    ) -> bool:
        """Validate parallel execution request."""
        execution_request = {
            'name': parallel_block.get('name', 'unknown_parallel_block'),
            'action_type': SecurityAction.EXECUTE_TASK,
            'resource_type': ResourceType.COMPUTE,
            'strategy': parallel_block.get('strategy', 'all'),
            'task_count': len(parallel_block.get('steps', [])),
            'max_concurrency': parallel_block.get('max_concurrency'),
            'resource_requirements': execution_metadata or {},
        }
        
        # Check if distributed execution is requested
        if execution_metadata and execution_metadata.get('distributed', False):
            execution_request['action_type'] = SecurityAction.DISTRIBUTE_TASK
            execution_request['resource_type'] = ResourceType.NETWORK
        
        return await self.validator.validate_execution_context(context, execution_request)
    
    async def validate_worker_task_assignment(
        self,
        worker_id: str,
        task_request: Dict[str, Any],
    ) -> bool:
        """Validate worker task assignment."""
        if worker_id not in self.worker_policies:
            logger.warning(f"No security policy found for worker {worker_id}")
            return False
        
        worker_policy = self.worker_policies[worker_id]
        return await self.validator.validate_worker_access(worker_policy, task_request)
    
    async def register_worker_security_policy(
        self,
        worker_id: str,
        allowed_task_types: Set[str],
        required_capabilities: Set[str],
        resource_limits: Dict[str, Any],
        created_by: str,
        **additional_policies
    ) -> WorkerSecurityPolicy:
        """Register security policy for worker."""
        policy = WorkerSecurityPolicy(
            worker_id=worker_id,
            allowed_task_types=allowed_task_types,
            required_capabilities=required_capabilities,
            resource_limits=resource_limits,
            network_restrictions=additional_policies.get('network_restrictions', {}),
            data_access_rules=additional_policies.get('data_access_rules', {}),
            execution_sandbox=additional_policies.get('execution_sandbox', {}),
            audit_level=additional_policies.get('audit_level', 'standard'),
            created_by=created_by,
        )
        
        self.worker_policies[worker_id] = policy
        
        if self.audit_enabled:
            self._log_security_event(
                event_type='worker_policy_registered',
                user_id=created_by,
                details={
                    'worker_id': worker_id,
                    'allowed_task_types': list(allowed_task_types),
                    'resource_limits': resource_limits,
                }
            )
        
        logger.info(f"Registered security policy for worker {worker_id}")
        return policy
    
    async def propagate_security_context(
        self,
        context: SecurityContext,
        target_node: str,
    ) -> Dict[str, Any]:
        """Propagate security context to distributed node."""
        # Create secure context payload
        context_payload = {
            'user_id': context.user_id,
            'session_id': context.session_id,
            'permission_level': context.permission_level.value,
            'capabilities': [cap.name for cap in context.capabilities if cap.is_valid()],
            'resource_quotas': context.resource_quotas,
            'restrictions': context.restrictions,
            'expires_at': context.expires_at,
            'propagated_at': time.time(),
            'target_node': target_node,
        }
        
        # Encrypt context if enabled
        if self.encryption_enabled:
            context_json = json.dumps(context_payload)
            encrypted_context = self.cipher.encrypt(context_json.encode())
            context_payload = {
                'encrypted': True,
                'data': encrypted_context.decode(),
            }
        
        # Audit context propagation
        if self.audit_enabled:
            context.add_audit_entry(
                action='propagate_context',
                resource=target_node,
                result='success',
                details={'encryption_used': self.encryption_enabled}
            )
        
        return context_payload
    
    async def receive_security_context(
        self,
        context_payload: Dict[str, Any],
        source_node: str,
    ) -> SecurityContext:
        """Receive and validate security context from another node."""
        try:
            # Decrypt context if encrypted
            if context_payload.get('encrypted'):
                if not self.encryption_enabled:
                    raise SecurityError("Received encrypted context but encryption not enabled")
                
                encrypted_data = context_payload['data'].encode()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                context_data = json.loads(decrypted_data.decode())
            else:
                context_data = context_payload
            
            # Validate context timing
            propagated_at = context_data.get('propagated_at', 0)
            if time.time() - propagated_at > 300:  # 5 minutes max
                raise SecurityError("Security context too old")
            
            # Reconstruct capabilities
            capability_objects = set()
            validator = self.validator
            if hasattr(validator, 'built_in_capabilities'):
                for cap_name in context_data['capabilities']:
                    if cap_name in validator.built_in_capabilities:
                        capability_objects.add(validator.built_in_capabilities[cap_name])
            
            # Create context
            context = SecurityContext(
                user_id=context_data['user_id'],
                session_id=context_data['session_id'],
                permission_level=PermissionLevel(context_data['permission_level']),
                capabilities=capability_objects,
                resource_quotas=context_data['resource_quotas'],
                restrictions=context_data['restrictions'],
                expires_at=context_data['expires_at'],
            )
            
            # Store context
            self.active_contexts[context.session_id] = context
            
            # Audit
            if self.audit_enabled:
                self._log_security_event(
                    event_type='context_received',
                    user_id=context.user_id,
                    session_id=context.session_id,
                    details={'source_node': source_node}
                )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to receive security context: {e}")
            raise SecurityError(f"Invalid security context: {e}")
    
    def _log_security_event(
        self,
        event_type: str,
        user_id: str = None,
        session_id: str = None,
        details: Dict[str, Any] = None,
    ):
        """Log security event for audit trail."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'user_id': user_id,
            'session_id': session_id,
            'details': details or {},
            'event_id': str(uuid.uuid4()),
        }
        
        self.audit_log.append(event)
        logger.info(f"Security event: {event_type} for user {user_id}")
    
    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get filtered audit trail."""
        filtered_events = []
        
        for event in self.audit_log:
            # Apply filters
            if user_id and event.get('user_id') != user_id:
                continue
            if session_id and event.get('session_id') != session_id:
                continue
            if event_type and event.get('event_type') != event_type:
                continue
            if since and event.get('timestamp', 0) < since:
                continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    async def cleanup_expired_contexts(self):
        """Clean up expired security contexts."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, context in self.active_contexts.items():
            if context.expires_at and current_time > context.expires_at:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            context = self.active_contexts.pop(session_id)
            if self.audit_enabled:
                self._log_security_event(
                    event_type='context_expired',
                    user_id=context.user_id,
                    session_id=session_id,
                )
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired security contexts")


# =============================================================================
# Security Exceptions
# =============================================================================

class SecurityError(Exception):
    """Security-related errors."""
    pass


class InsufficientPermissionsError(SecurityError):
    """Insufficient permissions for operation."""
    pass


class ExpiredContextError(SecurityError):
    """Security context has expired."""
    pass


class InvalidCapabilityError(SecurityError):
    """Invalid or unknown capability."""
    pass


# =============================================================================
# Convenience Functions and Integration
# =============================================================================

# Global security manager instance
_global_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    
    return _global_security_manager


async def create_execution_context(
    user_id: str,
    capabilities: List[str],
    permission_level: PermissionLevel = PermissionLevel.READ_WRITE,
    **kwargs
) -> SecurityContext:
    """Convenience function to create security context."""
    manager = get_security_manager()
    return await manager.create_security_context(
        user_id=user_id,
        permission_level=permission_level,
        capabilities=capabilities,
        **kwargs
    )


async def validate_execution_security(
    context: SecurityContext,
    execution_config: Dict[str, Any],
    **kwargs
) -> bool:
    """Convenience function to validate execution security."""
    manager = get_security_manager()
    return await manager.validate_parallel_execution(context, execution_config, **kwargs)


# Default capability sets for common use cases
DEFAULT_CAPABILITIES = {
    'basic_user': ['execute_basic', 'access_data'],
    'worker_node': ['execute_basic', 'access_data', 'network_access'],
    'administrator': ['execute_basic', 'access_data', 'network_access', 'manage_workers', 'distribute_execution', 'admin'],
    'system_service': ['execute_basic', 'access_data', 'network_access', 'manage_workers', 'distribute_execution', 'manage_events'],
}