"""
Static validation of security constraints for Namel3ss.

This module provides compile-time validation of security constraints:
- Tool access validation (agents can only use declared tools)
- Capability validation (agents have required capabilities)
- Permission level validation (agent permissions >= tool requirements)
- Policy validation (policies are well-formed and valid)

These validators are used by the resolver to catch security issues at
compile time, before any code execution.

Example usage:
--------------

    from namel3ss.security.validation import SecurityValidator
    
    validator = SecurityValidator()
    
    # Validate agent can access tool
    result = validator.validate_tool_access(
        agent=agent_def,
        tool=tool_def,
        app=app
    )
    
    if not result.allowed:
        raise SecurityError(result.reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from namel3ss.ast.agents import AgentDefinition
from namel3ss.ast.ai_tools import ToolDefinition
from namel3ss.ast.application import App
from namel3ss.ast.security import (
    CapabilityType,
    PermissionLevel,
    SecurityPolicy,
    SecurityCheckResult,
    CapabilityDenied,
    PermissionDenied,
    InvalidSecurityConfig,
)
from namel3ss.security.config import SecurityConfig, get_security_config


# =============================================================================
# Validation Functions
# =============================================================================


def validate_tool_access(
    agent: AgentDefinition,
    tool: ToolDefinition,
    app: App,
    config: Optional[SecurityConfig] = None
) -> SecurityCheckResult:
    """
    Validate that an agent can access a tool.
    
    Checks:
    1. Tool is declared in the application
    2. Tool is in agent's tool_names list
    3. Tool is allowed in current environment
    4. Agent has required capabilities
    5. Agent permission level >= tool permission level
    
    Args:
        agent: Agent attempting to access the tool
        tool: Tool being accessed
        app: Application containing all declarations
        config: Security configuration (uses global if not provided)
    
    Returns:
        SecurityCheckResult with allowed=True or violations
    """
    config = config or get_security_config()
    violations = []
    warnings = []
    
    # Check 1: Tool in agent's allowed list
    if tool.name not in agent.tool_names:
        violations.append(
            f"Tool '{tool.name}' not in agent '{agent.name}' tools list. "
            f"Agent tools: {agent.tool_names}"
        )
    
    # Check 2: Tool allowed in environment
    if not config.is_tool_allowed(tool.name):
        violations.append(
            f"Tool '{tool.name}' not allowed in environment '{config.current_environment.value}'"
        )
    
    # Check 3: Capability validation
    cap_result = validate_capability_grant(agent, tool, config)
    if not cap_result.allowed:
        violations.extend(cap_result.violations)
        warnings.extend(cap_result.warnings)
    
    # Check 4: Permission level validation
    perm_result = validate_permission_level(agent, tool, config)
    if not perm_result.allowed:
        violations.extend(perm_result.violations)
        warnings.extend(perm_result.warnings)
    
    return SecurityCheckResult(
        allowed=len(violations) == 0,
        reason=violations[0] if violations else None,
        violations=violations,
        warnings=warnings,
    )


def validate_capability_grant(
    agent: AgentDefinition,
    tool: ToolDefinition,
    config: Optional[SecurityConfig] = None
) -> SecurityCheckResult:
    """
    Validate that agent has required capabilities for tool.
    
    Args:
        agent: Agent with granted capabilities
        tool: Tool with required capabilities
        config: Security configuration
    
    Returns:
        SecurityCheckResult
    """
    config = config or get_security_config()
    violations = []
    warnings = []
    
    # Parse tool's required capabilities
    required_caps = set(tool.required_capabilities or [])
    
    # Parse agent's granted capabilities
    granted_caps = set(agent.capabilities or [])
    
    # Check each required capability
    missing_caps = required_caps - granted_caps
    
    if missing_caps:
        violations.append(
            f"Agent '{agent.name}' lacks required capabilities for tool '{tool.name}'. "
            f"Missing: {sorted(missing_caps)}, "
            f"Has: {sorted(granted_caps)}, "
            f"Needs: {sorted(required_caps)}"
        )
    
    return SecurityCheckResult(
        allowed=len(violations) == 0,
        reason=violations[0] if violations else None,
        violations=violations,
        warnings=warnings,
    )


def validate_permission_level(
    agent: AgentDefinition,
    tool: ToolDefinition,
    config: Optional[SecurityConfig] = None
) -> SecurityCheckResult:
    """
    Validate that agent permission level is sufficient for tool.
    
    Permission hierarchy: READ_ONLY < READ_WRITE < NETWORK/FILESYSTEM < ADMIN < UNRESTRICTED
    
    Args:
        agent: Agent with permission level
        tool: Tool with required permission level
        config: Security configuration
    
    Returns:
        SecurityCheckResult
    """
    config = config or get_security_config()
    violations = []
    warnings = []
    
    # Parse permission levels (handle both string and enum)
    agent_perm_value = agent.permission_level or "read_only"
    tool_perm_value = tool.permission_level or "read_only"
    
    # Convert to PermissionLevel if string
    if isinstance(agent_perm_value, str):
        try:
            agent_perm = PermissionLevel(agent_perm_value.lower())
        except ValueError:
            violations.append(
                f"Invalid permission level for agent '{agent.name}': {agent_perm_value}"
            )
            return SecurityCheckResult(
                allowed=False,
                reason=violations[0],
                violations=violations,
            )
    else:
        agent_perm = agent_perm_value
    
    # Convert to PermissionLevel if string
    if isinstance(tool_perm_value, str):
        try:
            tool_perm = PermissionLevel(tool_perm_value.lower())
        except ValueError:
            violations.append(
                f"Invalid permission level for tool '{tool.name}': {tool_perm_value}"
            )
            return SecurityCheckResult(
                allowed=False,
                reason=violations[0],
                violations=violations,
            )
    else:
        tool_perm = tool_perm_value
    
    # Check permission hierarchy
    if agent_perm < tool_perm:
        violations.append(
            f"Agent '{agent.name}' permission level '{agent_perm.value}' "
            f"insufficient for tool '{tool.name}' (requires '{tool_perm.value}')"
        )
    
    # Check if permission is allowed in environment
    if not config.is_permission_allowed(tool_perm):
        violations.append(
            f"Permission level '{tool_perm.value}' not allowed in "
            f"environment '{config.current_environment.value}'"
        )
    
    # Warn on elevated permissions if configured
    profile = config.get_current_profile()
    if profile.warn_on_elevated_permissions:
        if agent_perm in [PermissionLevel.ADMIN, PermissionLevel.UNRESTRICTED]:
            warnings.append(
                f"Agent '{agent.name}' has elevated permission level: {agent_perm.value}"
            )
    
    return SecurityCheckResult(
        allowed=len(violations) == 0,
        reason=violations[0] if violations else None,
        violations=violations,
        warnings=warnings,
    )


def validate_security_policy(
    policy: SecurityPolicy,
    config: Optional[SecurityConfig] = None
) -> SecurityCheckResult:
    """
    Validate that a security policy is well-formed and valid.
    
    Checks:
    - Timeouts are positive
    - Limits are non-negative
    - Rate limits are reasonable
    - Concurrency limits are positive
    
    Args:
        policy: Policy to validate
        config: Security configuration
    
    Returns:
        SecurityCheckResult
    """
    violations = []
    warnings = []
    
    # Validate timeouts
    if policy.tool_timeout_seconds <= 0:
        violations.append(
            f"Invalid tool_timeout_seconds: {policy.tool_timeout_seconds} (must be positive)"
        )
    
    if policy.llm_timeout_seconds <= 0:
        violations.append(
            f"Invalid llm_timeout_seconds: {policy.llm_timeout_seconds} (must be positive)"
        )
    
    if policy.total_execution_timeout_seconds is not None:
        if policy.total_execution_timeout_seconds <= 0:
            violations.append(
                f"Invalid total_execution_timeout_seconds: "
                f"{policy.total_execution_timeout_seconds} (must be positive)"
            )
    
    # Validate rate limits
    if policy.rate_limit_requests_per_minute is not None:
        if policy.rate_limit_requests_per_minute < 0:
            violations.append(
                f"Invalid rate_limit_requests_per_minute: "
                f"{policy.rate_limit_requests_per_minute} (must be non-negative)"
            )
        elif policy.rate_limit_requests_per_minute == 0:
            warnings.append("Rate limit of 0 will block all requests")
    
    if policy.rate_limit_requests_per_hour is not None:
        if policy.rate_limit_requests_per_hour < 0:
            violations.append(
                f"Invalid rate_limit_requests_per_hour: "
                f"{policy.rate_limit_requests_per_hour} (must be non-negative)"
            )
    
    # Validate token limits
    if policy.max_tokens_per_request is not None:
        if policy.max_tokens_per_request < 0:
            violations.append(
                f"Invalid max_tokens_per_request: "
                f"{policy.max_tokens_per_request} (must be non-negative)"
            )
    
    if policy.max_tokens_per_agent is not None:
        if policy.max_tokens_per_agent < 0:
            violations.append(
                f"Invalid max_tokens_per_agent: "
                f"{policy.max_tokens_per_agent} (must be non-negative)"
            )
    
    if policy.max_total_tokens is not None:
        if policy.max_total_tokens < 0:
            violations.append(
                f"Invalid max_total_tokens: "
                f"{policy.max_total_tokens} (must be non-negative)"
            )
    
    # Validate cost limits
    if policy.max_cost_per_request is not None:
        if policy.max_cost_per_request < 0:
            violations.append(
                f"Invalid max_cost_per_request: "
                f"{policy.max_cost_per_request} (must be non-negative)"
            )
    
    if policy.max_cost_per_agent is not None:
        if policy.max_cost_per_agent < 0:
            violations.append(
                f"Invalid max_cost_per_agent: "
                f"{policy.max_cost_per_agent} (must be non-negative)"
            )
    
    if policy.max_total_cost is not None:
        if policy.max_total_cost < 0:
            violations.append(
                f"Invalid max_total_cost: "
                f"{policy.max_total_cost} (must be non-negative)"
            )
    
    # Validate concurrency limits
    if policy.max_concurrent_tool_calls <= 0:
        violations.append(
            f"Invalid max_concurrent_tool_calls: "
            f"{policy.max_concurrent_tool_calls} (must be positive)"
        )
    
    if policy.max_concurrent_llm_calls <= 0:
        violations.append(
            f"Invalid max_concurrent_llm_calls: "
            f"{policy.max_concurrent_llm_calls} (must be positive)"
        )
    
    if policy.max_concurrent_agents is not None:
        if policy.max_concurrent_agents <= 0:
            violations.append(
                f"Invalid max_concurrent_agents: "
                f"{policy.max_concurrent_agents} (must be positive)"
            )
    
    # Validate fail mode
    if policy.fail_mode not in ["closed", "open"]:
        violations.append(
            f"Invalid fail_mode: {policy.fail_mode} (must be 'closed' or 'open')"
        )
    
    # Validate rate limit scope
    if policy.rate_limit_scope not in ["agent", "tool", "global"]:
        violations.append(
            f"Invalid rate_limit_scope: {policy.rate_limit_scope} "
            f"(must be 'agent', 'tool', or 'global')"
        )
    
    return SecurityCheckResult(
        allowed=len(violations) == 0,
        reason=violations[0] if violations else None,
        violations=violations,
        warnings=warnings,
    )


def validate_application_security(
    app: App,
    config: Optional[SecurityConfig] = None
) -> SecurityCheckResult:
    """
    Validate security constraints across entire application.
    
    Performs comprehensive validation:
    - All agents reference declared tools
    - All tool accesses are valid
    - All policies are valid
    - No security violations exist
    
    Args:
        app: Application to validate
        config: Security configuration
    
    Returns:
        SecurityCheckResult with all violations/warnings
    """
    config = config or get_security_config()
    violations = []
    warnings = []
    
    # Build tool index
    tools_by_name = {tool.name: tool for tool in app.tools}
    
    # Validate each agent
    for agent in app.agents:
        # Check all referenced tools exist
        for tool_name in agent.tool_names:
            if tool_name not in tools_by_name:
                violations.append(
                    f"Agent '{agent.name}' references undeclared tool '{tool_name}'"
                )
                continue
            
            # Validate access to this tool
            tool = tools_by_name[tool_name]
            result = validate_tool_access(agent, tool, app, config)
            violations.extend(result.violations)
            warnings.extend(result.warnings)
    
    # Validate all tools have valid security configs
    for tool in app.tools:
        if tool.permission_level:
            # Handle both string and enum
            if isinstance(tool.permission_level, str):
                try:
                    perm = PermissionLevel(tool.permission_level.lower())
                    if not config.is_permission_allowed(perm):
                        warnings.append(
                            f"Tool '{tool.name}' has permission level '{perm.value}' "
                            f"not allowed in environment '{config.current_environment.value}'"
                        )
                except ValueError:
                    violations.append(
                        f"Tool '{tool.name}' has invalid permission level: {tool.permission_level}"
                    )
            else:
                # Already a PermissionLevel enum
                perm = tool.permission_level
                if not config.is_permission_allowed(perm):
                    warnings.append(
                        f"Tool '{tool.name}' has permission level '{perm.value}' "
                        f"not allowed in environment '{config.current_environment.value}'"
                    )
    
    return SecurityCheckResult(
        allowed=len(violations) == 0,
        reason=violations[0] if violations else None,
        violations=violations,
        warnings=warnings,
    )


# =============================================================================
# SecurityValidator Class
# =============================================================================


class SecurityValidator:
    """
    Comprehensive security validator for static analysis.
    
    Provides methods for validating security constraints during compilation.
    Integrates with the resolver to catch security issues before execution.
    
    Example:
        validator = SecurityValidator(config)
        
        for agent in app.agents:
            for tool_name in agent.tool_names:
                tool = get_tool(tool_name)
                result = validator.validate_tool_access(agent, tool, app)
                if not result.allowed:
                    raise CompileError(result.reason)
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize validator with security configuration.
        
        Args:
            config: Security configuration (uses global if not provided)
        """
        self.config = config or get_security_config()
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_tool_access(
        self,
        agent: AgentDefinition,
        tool: ToolDefinition,
        app: App
    ) -> SecurityCheckResult:
        """Validate agent can access tool."""
        result = validate_tool_access(agent, tool, app, self.config)
        self.errors.extend(result.violations)
        self.warnings.extend(result.warnings)
        return result
    
    def validate_capability_grant(
        self,
        agent: AgentDefinition,
        tool: ToolDefinition
    ) -> SecurityCheckResult:
        """Validate agent has required capabilities."""
        result = validate_capability_grant(agent, tool, self.config)
        self.errors.extend(result.violations)
        self.warnings.extend(result.warnings)
        return result
    
    def validate_permission_level(
        self,
        agent: AgentDefinition,
        tool: ToolDefinition
    ) -> SecurityCheckResult:
        """Validate agent permission level is sufficient."""
        result = validate_permission_level(agent, tool, self.config)
        self.errors.extend(result.violations)
        self.warnings.extend(result.warnings)
        return result
    
    def validate_policy(self, policy: SecurityPolicy) -> SecurityCheckResult:
        """Validate security policy is well-formed."""
        result = validate_security_policy(policy, self.config)
        self.errors.extend(result.violations)
        self.warnings.extend(result.warnings)
        return result
    
    def validate_application(self, app: App) -> SecurityCheckResult:
        """Validate entire application security."""
        result = validate_application_security(app, self.config)
        self.errors.extend(result.violations)
        self.warnings.extend(result.warnings)
        return result
    
    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings were recorded."""
        return len(self.warnings) > 0
    
    def clear(self) -> None:
        """Clear recorded errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
    
    def get_summary(self) -> str:
        """Get formatted summary of errors and warnings."""
        lines = []
        
        if self.errors:
            lines.append(f"Security Errors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"  {i}. {error}")
        
        if self.warnings:
            lines.append(f"\nSecurity Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
        
        return "\n".join(lines) if lines else "No security issues found."


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "validate_tool_access",
    "validate_capability_grant",
    "validate_permission_level",
    "validate_security_policy",
    "validate_application_security",
    "SecurityValidator",
]
