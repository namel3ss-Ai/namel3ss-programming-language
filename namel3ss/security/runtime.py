"""
Runtime security enforcement for Namel3ss.

This module implements runtime guards and enforcement of security policies:
- SecurityGuard: Main enforcement class
- RateLimiter: Request rate limiting
- TokenCounter: Token usage tracking
- CostTracker: Cost/budget tracking
- TimeoutEnforcer: Execution timeout management

These components enforce security policies during execution, providing
defense-in-depth beyond compile-time validation.

Example usage:
--------------

    from namel3ss.security.runtime import get_security_guard
    
    guard = get_security_guard()
    
    # Before tool invocation
    result = guard.check_tool_invocation(
        agent_name="researcher",
        tool_name="web_search",
        context=execution_context
    )
    
    if not result.allowed:
        raise SecurityViolation(result.reason)
    
    # Execute tool...
    
    # Record completion
    guard.record_tool_completion(agent_name, tool_name, success=True)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from namel3ss.ast.security import (
    SecurityCheckResult,
    SecurityAuditEvent,
    RateLimitExceeded,
    TimeoutExceeded,
    TokenLimitExceeded,
    CostLimitExceeded,
    CapabilityDenied,
)
from namel3ss.security.config import SecurityConfig, get_security_config


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class RateLimitState:
    """Tracks rate limit state for a scope (agent, tool, or global)."""
    
    requests: List[float] = field(default_factory=list)  # Timestamps
    
    def add_request(self, timestamp: float) -> None:
        """Record a request at given timestamp."""
        self.requests.append(timestamp)
    
    def clean_old_requests(self, current_time: float, window_seconds: float) -> None:
        """Remove requests older than the time window."""
        cutoff = current_time - window_seconds
        self.requests = [t for t in self.requests if t >= cutoff]
    
    def count_in_window(self, current_time: float, window_seconds: float) -> int:
        """Count requests in the time window."""
        self.clean_old_requests(current_time, window_seconds)
        return len(self.requests)


class RateLimiter:
    """
    Token bucket-based rate limiter with per-agent, per-tool, and global scopes.
    
    Enforces rate limits defined in security policies:
    - Requests per minute
    - Requests per hour
    - Scoped by agent, tool, or global
    """
    
    def __init__(self):
        """Initialize rate limiter."""
        self.state: Dict[str, RateLimitState] = defaultdict(RateLimitState)
    
    def check_rate_limit(
        self,
        scope_key: str,
        limit_per_minute: Optional[int],
        limit_per_hour: Optional[int],
        current_time: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if rate limit is exceeded.
        
        Args:
            scope_key: Unique key for scope (e.g., "agent:researcher", "tool:web_search")
            limit_per_minute: Requests per minute limit
            limit_per_hour: Requests per hour limit
            current_time: Current timestamp (uses time.time() if not provided)
        
        Returns:
            (allowed, reason) tuple
        """
        if limit_per_minute is None and limit_per_hour is None:
            return True, None  # No limits configured
        
        current_time = current_time or time.time()
        state = self.state[scope_key]
        
        # Check per-minute limit
        if limit_per_minute is not None:
            count_minute = state.count_in_window(current_time, 60.0)
            if count_minute >= limit_per_minute:
                return False, f"Rate limit exceeded: {count_minute}/{limit_per_minute} requests per minute"
        
        # Check per-hour limit
        if limit_per_hour is not None:
            count_hour = state.count_in_window(current_time, 3600.0)
            if count_hour >= limit_per_hour:
                return False, f"Rate limit exceeded: {count_hour}/{limit_per_hour} requests per hour"
        
        return True, None
    
    def record_request(
        self,
        scope_key: str,
        current_time: Optional[float] = None
    ) -> None:
        """
        Record a request for rate limiting.
        
        Args:
            scope_key: Unique key for scope
            current_time: Current timestamp
        """
        current_time = current_time or time.time()
        self.state[scope_key].add_request(current_time)
    
    def reset(self, scope_key: Optional[str] = None) -> None:
        """
        Reset rate limit state.
        
        Args:
            scope_key: Specific scope to reset, or None to reset all
        """
        if scope_key:
            if scope_key in self.state:
                del self.state[scope_key]
        else:
            self.state.clear()


# =============================================================================
# Token Counter
# =============================================================================


class TokenCounter:
    """
    Tracks token usage across agents and LLM calls.
    
    Enforces token limits defined in security policies:
    - Max tokens per request
    - Max tokens per agent
    - Max total tokens
    """
    
    def __init__(self):
        """Initialize token counter."""
        self.agent_tokens: Dict[str, int] = defaultdict(int)
        self.total_tokens: int = 0
    
    def check_token_limit(
        self,
        agent_name: str,
        requested_tokens: int,
        max_per_request: Optional[int],
        max_per_agent: Optional[int],
        max_total: Optional[int]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if token limits would be exceeded.
        
        Args:
            agent_name: Agent making the request
            requested_tokens: Tokens requested for this call
            max_per_request: Max tokens per request
            max_per_agent: Max tokens per agent
            max_total: Max total tokens
        
        Returns:
            (allowed, reason) tuple
        """
        # Check per-request limit
        if max_per_request is not None and requested_tokens > max_per_request:
            return False, (
                f"Token limit exceeded: requested {requested_tokens}, "
                f"max per request is {max_per_request}"
            )
        
        # Check per-agent limit
        if max_per_agent is not None:
            agent_total = self.agent_tokens[agent_name] + requested_tokens
            if agent_total > max_per_agent:
                return False, (
                    f"Agent token limit exceeded: {agent_total}/{max_per_agent} "
                    f"(current: {self.agent_tokens[agent_name]}, requested: {requested_tokens})"
                )
        
        # Check global limit
        if max_total is not None:
            new_total = self.total_tokens + requested_tokens
            if new_total > max_total:
                return False, (
                    f"Global token limit exceeded: {new_total}/{max_total} "
                    f"(current: {self.total_tokens}, requested: {requested_tokens})"
                )
        
        return True, None
    
    def record_tokens(self, agent_name: str, tokens_used: int) -> None:
        """
        Record token usage.
        
        Args:
            agent_name: Agent that used the tokens
            tokens_used: Number of tokens used
        """
        self.agent_tokens[agent_name] += tokens_used
        self.total_tokens += tokens_used
    
    def get_agent_usage(self, agent_name: str) -> int:
        """Get total tokens used by agent."""
        return self.agent_tokens[agent_name]
    
    def get_total_usage(self) -> int:
        """Get total tokens used globally."""
        return self.total_tokens
    
    def reset(self, agent_name: Optional[str] = None) -> None:
        """
        Reset token counters.
        
        Args:
            agent_name: Specific agent to reset, or None to reset all
        """
        if agent_name:
            if agent_name in self.agent_tokens:
                self.total_tokens -= self.agent_tokens[agent_name]
                del self.agent_tokens[agent_name]
        else:
            self.agent_tokens.clear()
            self.total_tokens = 0


# =============================================================================
# Cost Tracker
# =============================================================================


class CostTracker:
    """
    Tracks monetary cost of LLM calls.
    
    Enforces cost limits defined in security policies:
    - Max cost per request
    - Max cost per agent
    - Max total cost
    """
    
    def __init__(self):
        """Initialize cost tracker."""
        self.agent_costs: Dict[str, float] = defaultdict(float)
        self.total_cost: float = 0.0
    
    def check_cost_limit(
        self,
        agent_name: str,
        estimated_cost: float,
        max_per_request: Optional[float],
        max_per_agent: Optional[float],
        max_total: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if cost limits would be exceeded.
        
        Args:
            agent_name: Agent making the request
            estimated_cost: Estimated cost for this call
            max_per_request: Max cost per request
            max_per_agent: Max cost per agent
            max_total: Max total cost
        
        Returns:
            (allowed, reason) tuple
        """
        # Check per-request limit
        if max_per_request is not None and estimated_cost > max_per_request:
            return False, (
                f"Cost limit exceeded: estimated ${estimated_cost:.4f}, "
                f"max per request is ${max_per_request:.4f}"
            )
        
        # Check per-agent limit
        if max_per_agent is not None:
            agent_total = self.agent_costs[agent_name] + estimated_cost
            if agent_total > max_per_agent:
                return False, (
                    f"Agent cost limit exceeded: ${agent_total:.4f}/${max_per_agent:.4f} "
                    f"(current: ${self.agent_costs[agent_name]:.4f}, "
                    f"estimated: ${estimated_cost:.4f})"
                )
        
        # Check global limit
        if max_total is not None:
            new_total = self.total_cost + estimated_cost
            if new_total > max_total:
                return False, (
                    f"Global cost limit exceeded: ${new_total:.4f}/${max_total:.4f} "
                    f"(current: ${self.total_cost:.4f}, estimated: ${estimated_cost:.4f})"
                )
        
        return True, None
    
    def record_cost(self, agent_name: str, cost: float) -> None:
        """
        Record cost.
        
        Args:
            agent_name: Agent that incurred the cost
            cost: Cost in USD
        """
        self.agent_costs[agent_name] += cost
        self.total_cost += cost
    
    def get_agent_cost(self, agent_name: str) -> float:
        """Get total cost for agent."""
        return self.agent_costs[agent_name]
    
    def get_total_cost(self) -> float:
        """Get total cost globally."""
        return self.total_cost
    
    def reset(self, agent_name: Optional[str] = None) -> None:
        """
        Reset cost tracking.
        
        Args:
            agent_name: Specific agent to reset, or None to reset all
        """
        if agent_name:
            if agent_name in self.agent_costs:
                self.total_cost -= self.agent_costs[agent_name]
                del self.agent_costs[agent_name]
        else:
            self.agent_costs.clear()
            self.total_cost = 0.0


# =============================================================================
# Security Guard
# =============================================================================


@dataclass
class ExecutionContext:
    """Context information for security checks."""
    
    agent_name: str
    tool_name: Optional[str] = None
    model_name: Optional[str] = None
    requested_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityGuard:
    """
    Runtime security enforcement layer.
    
    The SecurityGuard enforces security policies at runtime:
    - Tool invocation checks
    - LLM call checks
    - Rate limiting
    - Token limits
    - Cost limits
    - Audit logging
    
    Example:
        guard = SecurityGuard(config)
        
        # Before tool call
        result = guard.check_tool_invocation(
            agent_name="researcher",
            tool_name="web_search",
            context=ctx
        )
        
        if not result.allowed:
            raise SecurityViolation(result.reason)
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize security guard.
        
        Args:
            config: Security configuration (uses global if not provided)
        """
        self.config = config or get_security_config()
        self.rate_limiter = RateLimiter()
        self.token_counter = TokenCounter()
        self.cost_tracker = CostTracker()
        self.audit_log: List[SecurityAuditEvent] = []
    
    def check_tool_invocation(
        self,
        agent_name: str,
        tool_name: str,
        context: Optional[ExecutionContext] = None
    ) -> SecurityCheckResult:
        """
        Check if tool invocation is allowed.
        
        Validates:
        - Tool is allowed in environment
        - Rate limits not exceeded
        - Agent has permission
        
        Args:
            agent_name: Agent invoking the tool
            tool_name: Tool being invoked
            context: Execution context
        
        Returns:
            SecurityCheckResult
        """
        violations = []
        warnings = []
        
        # Check tool allowed in environment
        if not self.config.is_tool_allowed(tool_name):
            violations.append(
                f"Tool '{tool_name}' not allowed in environment "
                f"'{self.config.current_environment.value}'"
            )
        
        # Get effective policy
        policy = self.config.get_effective_policy(agent_name=agent_name, tool_name=tool_name)
        
        # Check rate limits
        if self.config.get_current_profile().enforce_rate_limits:
            if policy.rate_limit_scope == "agent":
                scope_key = f"agent:{agent_name}"
            elif policy.rate_limit_scope == "tool":
                scope_key = f"tool:{tool_name}"
            else:  # global
                scope_key = "global"
            
            allowed, reason = self.rate_limiter.check_rate_limit(
                scope_key,
                policy.rate_limit_requests_per_minute,
                policy.rate_limit_requests_per_hour
            )
            
            if not allowed:
                violations.append(f"Rate limit exceeded for {scope_key}: {reason}")
        
        # Create audit event
        audit_event = SecurityAuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type="tool_call",
            agent_name=agent_name,
            tool_name=tool_name,
            result="allowed" if not violations else "denied",
            reason=violations[0] if violations else None,
            details={"policy": policy.name} if policy else {},
        )
        
        if self.config.audit_enabled:
            self.audit_log.append(audit_event)
        
        return SecurityCheckResult(
            allowed=len(violations) == 0,
            reason=violations[0] if violations else None,
            violations=violations,
            warnings=warnings,
            audit_event=audit_event,
        )
    
    def check_llm_invocation(
        self,
        agent_name: str,
        model_name: str,
        requested_tokens: int,
        estimated_cost: Optional[float] = None,
        context: Optional[ExecutionContext] = None
    ) -> SecurityCheckResult:
        """
        Check if LLM call is allowed.
        
        Validates:
        - Token limits not exceeded
        - Cost limits not exceeded (if applicable)
        - Rate limits not exceeded
        
        Args:
            agent_name: Agent making the call
            model_name: LLM model being called
            requested_tokens: Tokens requested
            estimated_cost: Estimated cost in USD
            context: Execution context
        
        Returns:
            SecurityCheckResult
        """
        violations = []
        warnings = []
        
        # Get effective policy
        policy = self.config.get_effective_policy(agent_name=agent_name)
        profile = self.config.get_current_profile()
        
        # Check token limits
        if profile.enforce_token_limits:
            allowed, reason = self.token_counter.check_token_limit(
                agent_name,
                requested_tokens,
                policy.max_tokens_per_request,
                policy.max_tokens_per_agent,
                policy.max_total_tokens
            )
            
            if not allowed:
                violations.append(reason)
        
        # Check cost limits
        if estimated_cost is not None and profile.enforce_cost_limits:
            allowed, reason = self.cost_tracker.check_cost_limit(
                agent_name,
                estimated_cost,
                policy.max_cost_per_request,
                policy.max_cost_per_agent,
                policy.max_total_cost
            )
            
            if not allowed:
                violations.append(reason)
        
        # Check rate limits
        if profile.enforce_rate_limits:
            scope_key = f"agent:{agent_name}" if policy.rate_limit_scope == "agent" else "global"
            
            allowed, reason = self.rate_limiter.check_rate_limit(
                scope_key,
                policy.rate_limit_requests_per_minute,
                policy.rate_limit_requests_per_hour
            )
            
            if not allowed:
                violations.append(f"Rate limit exceeded: {reason}")
        
        # Create audit event
        audit_event = SecurityAuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type="llm_call",
            agent_name=agent_name,
            resource_name=model_name,
            result="allowed" if not violations else "denied",
            reason=violations[0] if violations else None,
            details={
                "requested_tokens": requested_tokens,
                "estimated_cost": estimated_cost,
                "policy": policy.name,
            },
        )
        
        if self.config.audit_enabled:
            self.audit_log.append(audit_event)
        
        return SecurityCheckResult(
            allowed=len(violations) == 0,
            reason=violations[0] if violations else None,
            violations=violations,
            warnings=warnings,
            audit_event=audit_event,
        )
    
    def record_tool_completion(
        self,
        agent_name: str,
        tool_name: str,
        success: bool = True,
        duration_seconds: Optional[float] = None
    ) -> None:
        """
        Record tool invocation completion.
        
        Updates rate limiters and audit log.
        
        Args:
            agent_name: Agent that invoked the tool
            tool_name: Tool that was invoked
            success: Whether the invocation succeeded
            duration_seconds: Execution duration
        """
        policy = self.config.get_effective_policy(agent_name=agent_name, tool_name=tool_name)
        
        # Record for rate limiting
        if policy.rate_limit_scope == "agent":
            scope_key = f"agent:{agent_name}"
        elif policy.rate_limit_scope == "tool":
            scope_key = f"tool:{tool_name}"
        else:
            scope_key = "global"
        
        self.rate_limiter.record_request(scope_key)
        
        # Audit log
        if self.config.audit_enabled:
            self.audit_log.append(SecurityAuditEvent(
                timestamp=datetime.now().isoformat(),
                event_type="tool_completion",
                agent_name=agent_name,
                tool_name=tool_name,
                result="success" if success else "error",
                details={"duration_seconds": duration_seconds},
            ))
    
    def record_llm_completion(
        self,
        agent_name: str,
        model_name: str,
        tokens_used: int,
        actual_cost: Optional[float] = None,
        duration_seconds: Optional[float] = None
    ) -> None:
        """
        Record LLM call completion.
        
        Updates token counters, cost trackers, rate limiters, and audit log.
        
        Args:
            agent_name: Agent that made the call
            model_name: Model that was called
            tokens_used: Actual tokens used
            actual_cost: Actual cost in USD
            duration_seconds: Execution duration
        """
        # Record tokens and cost
        self.token_counter.record_tokens(agent_name, tokens_used)
        
        if actual_cost is not None:
            self.cost_tracker.record_cost(agent_name, actual_cost)
        
        # Record for rate limiting
        policy = self.config.get_effective_policy(agent_name=agent_name)
        scope_key = f"agent:{agent_name}" if policy.rate_limit_scope == "agent" else "global"
        self.rate_limiter.record_request(scope_key)
        
        # Audit log
        if self.config.audit_enabled:
            self.audit_log.append(SecurityAuditEvent(
                timestamp=datetime.now().isoformat(),
                event_type="llm_completion",
                agent_name=agent_name,
                resource_name=model_name,
                result="success",
                details={
                    "tokens_used": tokens_used,
                    "actual_cost": actual_cost,
                    "duration_seconds": duration_seconds,
                },
            ))
    
    def get_audit_log(self) -> List[SecurityAuditEvent]:
        """Get audit log entries."""
        return self.audit_log.copy()
    
    def clear_audit_log(self) -> None:
        """Clear audit log (for testing)."""
        self.audit_log.clear()
    
    def reset(self) -> None:
        """Reset all runtime state (for testing)."""
        self.rate_limiter.reset()
        self.token_counter.reset()
        self.cost_tracker.reset()
        self.audit_log.clear()


# =============================================================================
# Global Instance
# =============================================================================


_global_guard: Optional[SecurityGuard] = None


def get_security_guard() -> SecurityGuard:
    """Get the global security guard instance."""
    global _global_guard
    if _global_guard is None:
        _global_guard = SecurityGuard()
    return _global_guard


def reset_security_guard() -> None:
    """Reset global security guard (for testing)."""
    global _global_guard
    _global_guard = None


__all__ = [
    "RateLimiter",
    "TokenCounter",
    "CostTracker",
    "ExecutionContext",
    "SecurityGuard",
    "get_security_guard",
    "reset_security_guard",
]
