"""
Security and capability model AST nodes for Namel3ss.

This module defines the security model that enforces safe, controlled
execution of agents, tools, and LLM calls. Security is a first-class
language concern, not an afterthought.

Core Concepts:
--------------

1. **Capabilities**: Explicit grants that allow agents to perform actions
   - Tool access: Permission to invoke specific tools
   - Permission levels: READ_ONLY, READ_WRITE, NETWORK, etc.
   - Environment-based: Different policies for dev/staging/prod

2. **Security Policies**: Quantitative constraints on execution
   - Rate limits: Requests per minute/hour
   - Timeouts: Maximum execution time for tools/LLMs
   - Token limits: Maximum tokens per request/agent
   - Cost controls: Budget constraints for LLM calls

3. **Environment Profiles**: Deployment-specific security configurations
   - Development: Loose restrictions, verbose warnings
   - Staging: Moderate restrictions, realistic testing
   - Production: Strict restrictions, minimal attack surface
   - Sandbox: Fully isolated, no external access

Example Usage:
--------------

    # Tool with security metadata
    tool web_search {
        description: "Search the web"
        permission_level: NETWORK
        required_capabilities: [NETWORK, HTTP_READ]
        
        parameters: { query: string }
        
        security: {
            timeout_seconds: 10.0
            rate_limit_per_minute: 30
        }
    }
    
    # Agent with capabilities
    agent researcher {
        llm: gpt_4o
        tools: [web_search, calculator]  # Explicit grants
        
        capabilities: [NETWORK, HTTP_READ]
        permission_level: READ_ONLY
        
        security_policy: {
            max_tokens_per_request: 2000
            rate_limit_per_minute: 20
        }
    }
    
    # Environment profile
    environment production {
        allowed_permission_levels: [READ_ONLY, NETWORK]
        deny_filesystem_access: true
        enforce_rate_limits: true
        require_https: true
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .source_location import SourceLocation


# =============================================================================
# Enumerations
# =============================================================================


class CapabilityType(Enum):
    """Types of capabilities that can be granted to agents."""
    
    TOOL_ACCESS = "tool_access"          # Can invoke specific tool
    TOOL_CATEGORY = "tool_category"      # Can invoke tools in category (http, db, etc.)
    MEMORY_READ = "memory_read"          # Can read from memory
    MEMORY_WRITE = "memory_write"        # Can write to memory
    NETWORK = "network"                  # Can make network calls
    FILESYSTEM = "filesystem"            # Can access filesystem
    DATABASE = "database"                # Can query databases
    HTTP_READ = "http_read"              # HTTP GET/read-only
    HTTP_WRITE = "http_write"            # HTTP POST/PUT/DELETE
    ADMIN = "admin"                      # Administrative operations


class PermissionLevel(Enum):
    """
    Permission levels for tools and agents.
    
    Hierarchy: READ_ONLY < READ_WRITE < NETWORK/FILESYSTEM < ADMIN < UNRESTRICTED
    """
    
    READ_ONLY = "read_only"           # Read-only operations, safest
    READ_WRITE = "read_write"         # Can modify data
    NETWORK = "network"               # Makes external network calls
    FILESYSTEM = "filesystem"         # Accesses local filesystem
    ADMIN = "admin"                   # Administrative operations
    UNRESTRICTED = "unrestricted"     # No restrictions (dangerous)
    
    def __lt__(self, other: PermissionLevel) -> bool:
        """Define permission level hierarchy."""
        order = {
            PermissionLevel.READ_ONLY: 0,
            PermissionLevel.READ_WRITE: 1,
            PermissionLevel.NETWORK: 2,
            PermissionLevel.FILESYSTEM: 2,
            PermissionLevel.ADMIN: 3,
            PermissionLevel.UNRESTRICTED: 4,
        }
        return order[self] < order[other]
    
    def __le__(self, other: PermissionLevel) -> bool:
        return self == other or self < other


class Environment(Enum):
    """Deployment environments with different security profiles."""
    
    DEVELOPMENT = "development"       # Loose restrictions, verbose warnings
    STAGING = "staging"               # Moderate restrictions, realistic testing
    PRODUCTION = "production"         # Strict restrictions, minimal attack surface
    SANDBOX = "sandbox"               # Fully isolated, no external access


class AuditLogLevel(Enum):
    """Audit logging verbosity levels."""
    
    NONE = "none"        # No audit logging
    MINIMAL = "minimal"  # Only security violations and errors
    INFO = "info"        # All security checks and tool/LLM calls
    DEBUG = "debug"      # Detailed debugging information
    FULL = "full"        # Everything including request/response payloads


# =============================================================================
# Core Security Data Structures
# =============================================================================


@dataclass
class SecurityCapability:
    """
    A capability grant that allows an agent to perform specific actions.
    
    Capabilities are explicit permissions that must be granted for agents
    to access tools, memory, or other resources. They follow the principle
    of least privilege: agents start with no capabilities and must request
    each one explicitly.
    
    Example:
        # Simple tool access
        SecurityCapability(
            capability_type=CapabilityType.TOOL_ACCESS,
            resource_name="web_search"
        )
        
        # Category-based access
        SecurityCapability(
            capability_type=CapabilityType.TOOL_CATEGORY,
            resource_name="http",
            constraints={"methods": ["GET"]}
        )
    """
    capability_type: CapabilityType
    resource_name: Optional[str] = None  # Tool name, memory scope, etc.
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional['SourceLocation'] = None


@dataclass
class ToolSecurity:
    """
    Security metadata for tool definitions.
    
    Defines what capabilities and permission levels are required to invoke
    a tool, plus tool-specific security constraints like timeouts and rate limits.
    
    Example:
        tool web_search {
            security: {
                permission_level: NETWORK
                required_capabilities: [NETWORK, HTTP_READ]
                timeout_seconds: 10.0
                rate_limit_per_minute: 30
                allowed_environments: [development, staging, production]
            }
        }
    """
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    required_capabilities: List[CapabilityType] = field(default_factory=list)
    timeout_seconds: float = 30.0
    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    allowed_environments: List[Environment] = field(default_factory=list)
    deny_environments: List[Environment] = field(default_factory=list)
    require_https: bool = False
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional['SourceLocation'] = None


@dataclass
class AgentSecurity:
    """
    Security configuration for agent definitions.
    
    Defines an agent's capabilities, permission level, and security policies.
    Agents can only invoke tools that match their granted capabilities and
    permission levels.
    
    Example:
        agent researcher {
            security: {
                capabilities: [NETWORK, HTTP_READ, MEMORY_READ]
                permission_level: READ_ONLY
                max_tokens_per_request: 2000
                rate_limit_per_minute: 20
                allowed_tools: [web_search, calculator]
            }
        }
    """
    capabilities: List[SecurityCapability] = field(default_factory=list)
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    allowed_tools: List[str] = field(default_factory=list)  # Explicit tool grants
    max_tokens_per_request: Optional[int] = None
    max_tokens_total: Optional[int] = None
    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    timeout_seconds: Optional[float] = None
    max_cost_per_request: Optional[float] = None  # In USD
    max_cost_total: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional['SourceLocation'] = None


@dataclass
class SecurityPolicy:
    """
    Configurable security constraints and policies.
    
    Security policies define quantitative limits on execution:
    - Rate limits (requests per minute/hour)
    - Timeouts (tool, LLM, total execution)
    - Token limits (per request, per agent, total)
    - Cost controls (budget limits)
    - Concurrency limits
    
    Policies stack in order of specificity:
    1. Global policy (applied to all)
    2. Environment policy (overrides global for env)
    3. Agent policy (overrides environment for agent)
    4. Tool policy (overrides agent for tool)
    
    Example:
        policy default_limits {
            rate_limit_requests_per_minute: 60
            tool_timeout_seconds: 30.0
            llm_timeout_seconds: 60.0
            max_tokens_per_request: 4000
            max_concurrent_tool_calls: 10
            max_concurrent_llm_calls: 5
        }
    """
    name: str
    
    # Rate limiting
    rate_limit_requests_per_minute: Optional[int] = None
    rate_limit_requests_per_hour: Optional[int] = None
    rate_limit_scope: str = "agent"  # agent, tool, global
    
    # Timeouts
    tool_timeout_seconds: float = 30.0
    llm_timeout_seconds: float = 60.0
    total_execution_timeout_seconds: Optional[float] = None
    
    # Token limits
    max_tokens_per_request: Optional[int] = None
    max_tokens_per_agent: Optional[int] = None
    max_total_tokens: Optional[int] = None
    
    # Cost controls
    max_cost_per_request: Optional[float] = None  # USD
    max_cost_per_agent: Optional[float] = None
    max_total_cost: Optional[float] = None
    
    # Concurrency
    max_concurrent_tool_calls: int = 10
    max_concurrent_llm_calls: int = 5
    max_concurrent_agents: Optional[int] = None
    
    # Validation
    enforce_strict: bool = False  # Fail on any violation vs. warn
    fail_mode: str = "closed"  # closed = deny on error, open = allow on error
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional['SourceLocation'] = None


@dataclass
class EnvironmentProfile:
    """
    Deployment environment security profile.
    
    Defines environment-specific security policies and restrictions:
    - Allowed permission levels and tool categories
    - Enforcement strictness (warnings vs. errors)
    - Audit logging configuration
    - Transport security requirements
    
    Example:
        environment production {
            allowed_permission_levels: [READ_ONLY, NETWORK]
            allowed_tool_categories: [http, db, vector]
            deny_filesystem_access: true
            enforce_rate_limits: true
            enforce_strict_timeouts: true
            require_https: true
            audit_log_level: INFO
        }
    """
    name: str
    environment: Environment
    
    # Permission restrictions
    allowed_permission_levels: List[PermissionLevel] = field(default_factory=list)
    denied_permission_levels: List[PermissionLevel] = field(default_factory=list)
    
    # Tool restrictions
    allowed_tool_categories: List[str] = field(default_factory=list)  # http, db, fs, etc.
    denied_tool_categories: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)  # Specific tools
    denied_tools: List[str] = field(default_factory=list)
    
    # Access restrictions
    deny_filesystem_access: bool = False
    deny_network_access: bool = False
    deny_database_access: bool = False
    
    # Enforcement
    enforce_rate_limits: bool = True
    enforce_strict_timeouts: bool = False
    enforce_token_limits: bool = True
    enforce_cost_limits: bool = False
    
    # Transport security
    require_https: bool = False
    require_tls: bool = False
    allowed_hosts: List[str] = field(default_factory=list)  # Whitelist
    denied_hosts: List[str] = field(default_factory=list)   # Blacklist
    
    # Warnings and logging
    warn_on_elevated_permissions: bool = True
    warn_on_policy_violations: bool = True
    audit_log_level: AuditLogLevel = AuditLogLevel.INFO
    
    # Default policies
    default_policy: Optional[SecurityPolicy] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional['SourceLocation'] = None


@dataclass
class SecurityAuditEvent:
    """
    Security audit log entry.
    
    Records security-relevant events for compliance, monitoring, and debugging:
    - Tool and LLM invocations
    - Capability checks
    - Policy violations
    - Authentication/authorization events
    """
    timestamp: str
    event_type: str  # tool_call, llm_call, capability_check, policy_violation, etc.
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    resource_name: Optional[str] = None
    action: Optional[str] = None
    result: str = "unknown"  # allowed, denied, warning, error
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityCheckResult:
    """
    Result of a security validation check.
    
    Returned by validation functions and runtime guards to indicate
    whether an operation is allowed and why.
    """
    allowed: bool
    reason: Optional[str] = None
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_event: Optional[SecurityAuditEvent] = None


# =============================================================================
# Exceptions
# =============================================================================


class SecurityViolation(Exception):
    """Base exception for security constraint violations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class CapabilityDenied(SecurityViolation):
    """Agent lacks required capability for operation."""
    pass


class PermissionDenied(SecurityViolation):
    """Permission level insufficient for operation."""
    pass


class RateLimitExceeded(SecurityViolation):
    """Rate limit exceeded."""
    pass


class TimeoutExceeded(SecurityViolation):
    """Operation timeout exceeded."""
    pass


class TokenLimitExceeded(SecurityViolation):
    """Token limit exceeded."""
    pass


class CostLimitExceeded(SecurityViolation):
    """Cost limit exceeded."""
    pass


class EnvironmentRestriction(SecurityViolation):
    """Operation not allowed in current environment."""
    pass


class InvalidSecurityConfig(Exception):
    """Security configuration is invalid."""
    pass


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "CapabilityType",
    "PermissionLevel",
    "Environment",
    "AuditLogLevel",
    
    # Data structures
    "SecurityCapability",
    "ToolSecurity",
    "AgentSecurity",
    "SecurityPolicy",
    "EnvironmentProfile",
    "SecurityAuditEvent",
    "SecurityCheckResult",
    
    # Exceptions
    "SecurityViolation",
    "CapabilityDenied",
    "PermissionDenied",
    "RateLimitExceeded",
    "TimeoutExceeded",
    "TokenLimitExceeded",
    "CostLimitExceeded",
    "EnvironmentRestriction",
    "InvalidSecurityConfig",
]
