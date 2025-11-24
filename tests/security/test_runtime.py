"""
Tests for Namel3ss security model - runtime enforcement.

Tests runtime enforcement of security policies:
- Rate limiting
- Token counting
- Cost tracking
- Security guard checks
"""

import pytest
import time

from namel3ss.ast.security import (
    SecurityPolicy,
    Environment,
    RateLimitExceeded,
    TokenLimitExceeded,
    CostLimitExceeded,
)
from namel3ss.security.runtime import (
    RateLimiter,
    TokenCounter,
    CostTracker,
    SecurityGuard,
    ExecutionContext,
    reset_security_guard,
)
from namel3ss.security.config import SecurityConfig, reset_security_config


@pytest.fixture(autouse=True)
def reset_all():
    """Reset all security components before each test."""
    reset_security_config()
    reset_security_guard()
    yield
    reset_security_config()
    reset_security_guard()


class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    def test_no_limit_always_allows(self):
        """No rate limit means all requests allowed."""
        limiter = RateLimiter()
        
        allowed, reason = limiter.check_rate_limit("test_scope", None, None)
        
        assert allowed
        assert reason is None
    
    def test_within_rate_limit(self):
        """Requests within limit are allowed."""
        limiter = RateLimiter()
        current_time = time.time()
        
        # Set limit: 5 per minute
        for i in range(5):
            allowed, _ = limiter.check_rate_limit("test", 5, None, current_time)
            assert allowed
            limiter.record_request("test", current_time)
            current_time += 1  # 1 second apart
    
    def test_exceeds_per_minute_limit(self):
        """Exceeding per-minute limit is denied."""
        limiter = RateLimiter()
        current_time = time.time()
        
        # Hit limit
        for i in range(5):
            limiter.record_request("test", current_time)
            current_time += 1
        
        # Next request should be denied
        allowed, reason = limiter.check_rate_limit("test", 5, None, current_time)
        
        assert not allowed
        assert "per minute" in reason.lower()
    
    def test_exceeds_per_hour_limit(self):
        """Exceeding per-hour limit is denied."""
        limiter = RateLimiter()
        current_time = time.time()
        
        # Hit limit
        for i in range(100):
            limiter.record_request("test", current_time)
            current_time += 10  # 10 seconds apart
        
        # Next request should be denied
        allowed, reason = limiter.check_rate_limit("test", None, 100, current_time)
        
        assert not allowed
        assert "per hour" in reason.lower()
    
    def test_requests_expire_after_window(self):
        """Old requests outside window don't count."""
        limiter = RateLimiter()
        start_time = time.time()
        
        # Make 5 requests
        for i in range(5):
            limiter.record_request("test", start_time + i)
        
        # Wait 61 seconds (outside 1-minute window)
        current_time = start_time + 61
        
        # Should be allowed again
        allowed, _ = limiter.check_rate_limit("test", 5, None, current_time)
        assert allowed
    
    def test_scope_isolation(self):
        """Different scopes have independent limits."""
        limiter = RateLimiter()
        current_time = time.time()
        
        # Hit limit for scope1
        for i in range(5):
            limiter.record_request("scope1", current_time)
        
        allowed1, _ = limiter.check_rate_limit("scope1", 5, None, current_time)
        assert not allowed1
        
        # scope2 should still be allowed
        allowed2, _ = limiter.check_rate_limit("scope2", 5, None, current_time)
        assert allowed2
    
    def test_reset_specific_scope(self):
        """Can reset specific scope."""
        limiter = RateLimiter()
        current_time = time.time()
        
        for i in range(5):
            limiter.record_request("test", current_time)
        
        # Exceeded
        allowed, _ = limiter.check_rate_limit("test", 5, None, current_time)
        assert not allowed
        
        # Reset
        limiter.reset("test")
        
        # Now allowed
        allowed, _ = limiter.check_rate_limit("test", 5, None, current_time)
        assert allowed


class TestTokenCounter:
    """Tests for TokenCounter class."""
    
    def test_no_limit_always_allows(self):
        """No token limit means all requests allowed."""
        counter = TokenCounter()
        
        allowed, reason = counter.check_token_limit("agent1", 1000, None, None, None)
        
        assert allowed
        assert reason is None
    
    def test_within_per_request_limit(self):
        """Request within per-request limit is allowed."""
        counter = TokenCounter()
        
        allowed, _ = counter.check_token_limit("agent1", 500, 1000, None, None)
        
        assert allowed
    
    def test_exceeds_per_request_limit(self):
        """Request exceeding per-request limit is denied."""
        counter = TokenCounter()
        
        allowed, reason = counter.check_token_limit("agent1", 2000, 1000, None, None)
        
        assert not allowed
        assert "per request" in reason.lower()
    
    def test_within_per_agent_limit(self):
        """Tokens within per-agent limit are allowed."""
        counter = TokenCounter()
        
        # Use some tokens
        counter.record_tokens("agent1", 500)
        
        # Request more (total 1000, under 5000 limit)
        allowed, _ = counter.check_token_limit("agent1", 500, None, 5000, None)
        
        assert allowed
    
    def test_exceeds_per_agent_limit(self):
        """Exceeding per-agent limit is denied."""
        counter = TokenCounter()
        
        # Use tokens up to limit
        counter.record_tokens("agent1", 900)
        
        # Request more (would exceed 1000 limit)
        allowed, reason = counter.check_token_limit("agent1", 200, None, 1000, None)
        
        assert not allowed
        assert "agent token limit" in reason.lower()
    
    def test_within_global_limit(self):
        """Tokens within global limit are allowed."""
        counter = TokenCounter()
        
        counter.record_tokens("agent1", 3000)
        counter.record_tokens("agent2", 2000)
        
        # Request more (total would be 6000, under 10000 limit)
        allowed, _ = counter.check_token_limit("agent3", 1000, None, None, 10000)
        
        assert allowed
    
    def test_exceeds_global_limit(self):
        """Exceeding global limit is denied."""
        counter = TokenCounter()
        
        counter.record_tokens("agent1", 5000)
        counter.record_tokens("agent2", 4000)
        
        # Request more (would exceed 10000 limit)
        allowed, reason = counter.check_token_limit("agent3", 2000, None, None, 10000)
        
        assert not allowed
        assert "global token limit" in reason.lower()
    
    def test_agent_isolation(self):
        """Each agent has independent per-agent limit."""
        counter = TokenCounter()
        
        # Agent1 uses all its tokens
        counter.record_tokens("agent1", 1000)
        
        # Agent2 should still have its full limit
        allowed, _ = counter.check_token_limit("agent2", 500, None, 1000, None)
        assert allowed
    
    def test_get_usage(self):
        """Can retrieve token usage."""
        counter = TokenCounter()
        
        counter.record_tokens("agent1", 100)
        counter.record_tokens("agent1", 200)
        counter.record_tokens("agent2", 150)
        
        assert counter.get_agent_usage("agent1") == 300
        assert counter.get_agent_usage("agent2") == 150
        assert counter.get_total_usage() == 450


class TestCostTracker:
    """Tests for CostTracker class."""
    
    def test_no_limit_always_allows(self):
        """No cost limit means all requests allowed."""
        tracker = CostTracker()
        
        allowed, reason = tracker.check_cost_limit("agent1", 0.50, None, None, None)
        
        assert allowed
        assert reason is None
    
    def test_within_per_request_limit(self):
        """Cost within per-request limit is allowed."""
        tracker = CostTracker()
        
        allowed, _ = tracker.check_cost_limit("agent1", 0.25, 0.50, None, None)
        
        assert allowed
    
    def test_exceeds_per_request_limit(self):
        """Cost exceeding per-request limit is denied."""
        tracker = CostTracker()
        
        allowed, reason = tracker.check_cost_limit("agent1", 0.75, 0.50, None, None)
        
        assert not allowed
        assert "$" in reason
        assert "per request" in reason.lower()
    
    def test_within_per_agent_limit(self):
        """Cost within per-agent limit is allowed."""
        tracker = CostTracker()
        
        tracker.record_cost("agent1", 1.50)
        
        allowed, _ = tracker.check_cost_limit("agent1", 0.50, None, 5.00, None)
        
        assert allowed
    
    def test_exceeds_per_agent_limit(self):
        """Exceeding per-agent limit is denied."""
        tracker = CostTracker()
        
        tracker.record_cost("agent1", 4.50)
        
        allowed, reason = tracker.check_cost_limit("agent1", 1.00, None, 5.00, None)
        
        assert not allowed
        assert "agent cost limit" in reason.lower()
    
    def test_exceeds_global_limit(self):
        """Exceeding global cost limit is denied."""
        tracker = CostTracker()
        
        tracker.record_cost("agent1", 25.00)
        tracker.record_cost("agent2", 20.00)
        
        allowed, reason = tracker.check_cost_limit("agent3", 10.00, None, None, 50.00)
        
        assert not allowed
        assert "global cost limit" in reason.lower()
    
    def test_get_costs(self):
        """Can retrieve cost tracking."""
        tracker = CostTracker()
        
        tracker.record_cost("agent1", 1.25)
        tracker.record_cost("agent1", 0.75)
        tracker.record_cost("agent2", 3.50)
        
        assert tracker.get_agent_cost("agent1") == 2.00
        assert tracker.get_agent_cost("agent2") == 3.50
        assert tracker.get_total_cost() == 5.50


class TestSecurityGuard:
    """Tests for SecurityGuard class."""
    
    def test_tool_invocation_allowed(self):
        """Tool invocation passes all checks."""
        config = SecurityConfig(
            current_environment=Environment.DEVELOPMENT,
            global_policy=SecurityPolicy(
                name="permissive",
                rate_limit_requests_per_minute=100,
                tool_timeout_seconds=30.0,
                llm_timeout_seconds=120.0,
            )
        )
        
        guard = SecurityGuard(config)
        
        result = guard.check_tool_invocation("agent1", "web_search")
        
        assert result.allowed
    
    def test_tool_not_allowed_in_environment(self):
        """Tool blocked by environment restrictions."""
        from namel3ss.ast.security import EnvironmentProfile
        
        profile = EnvironmentProfile(
            name="test_prod",
            environment=Environment.PRODUCTION,
            denied_tools=["dangerous_tool"],
            enforce_rate_limits=False
        )
        
        config = SecurityConfig(
            current_environment=Environment.PRODUCTION,
            environments={Environment.PRODUCTION: profile}
        )
        
        guard = SecurityGuard(config)
        
        result = guard.check_tool_invocation("agent1", "dangerous_tool")
        
        assert not result.allowed
        assert "not allowed" in result.reason.lower()
    
    def test_tool_rate_limit_enforced(self):
        """Tool invocation respects rate limits."""
        config = SecurityConfig(
            current_environment=Environment.PRODUCTION,
            global_policy=SecurityPolicy(
                name="strict",
                rate_limit_requests_per_minute=2,
                rate_limit_scope="agent"
            )
        )
        
        guard = SecurityGuard(config)
        
        # First two calls should succeed
        result1 = guard.check_tool_invocation("agent1", "tool1")
        assert result1.allowed
        guard.record_tool_completion("agent1", "tool1", success=True)
        
        result2 = guard.check_tool_invocation("agent1", "tool1")
        assert result2.allowed
        guard.record_tool_completion("agent1", "tool1", success=True)
        
        # Third call should be rate-limited
        result3 = guard.check_tool_invocation("agent1", "tool1")
        assert not result3.allowed
        assert "rate limit" in result3.reason.lower()
    
    def test_llm_invocation_allowed(self):
        """LLM invocation passes all checks."""
        config = SecurityConfig(
            global_policy=SecurityPolicy(
                name="permissive",
                max_tokens_per_request=10000
            )
        )
        
        guard = SecurityGuard(config)
        
        result = guard.check_llm_invocation("agent1", "gpt-4", requested_tokens=1000)
        
        assert result.allowed
    
    def test_llm_token_limit_enforced(self):
        """LLM invocation respects token limits."""
        config = SecurityConfig(
            current_environment=Environment.PRODUCTION,
            global_policy=SecurityPolicy(
                name="strict",
                max_tokens_per_request=1000
            )
        )
        
        guard = SecurityGuard(config)
        
        # Request exceeds limit
        result = guard.check_llm_invocation("agent1", "gpt-4", requested_tokens=2000)
        
        assert not result.allowed
        assert "token limit" in result.reason.lower()
    
    def test_llm_per_agent_token_limit(self):
        """LLM respects per-agent token limits."""
        from namel3ss.ast.security import EnvironmentProfile
        
        profile = EnvironmentProfile(
            name="test",
            environment=Environment.DEVELOPMENT,
            enforce_token_limits=True
        )
        
        config = SecurityConfig(
            current_environment=Environment.DEVELOPMENT,
            environments={Environment.DEVELOPMENT: profile},
            global_policy=SecurityPolicy(
                name="limited",
                max_tokens_per_request=1500,  # Per-request limit
                max_tokens_per_agent=4000     # Per-agent limit
            )
        )
        
        guard = SecurityGuard(config)
        
        # Use tokens
        guard.record_llm_completion("agent1", "gpt-4", tokens_used=3000)
        
        # Next request would exceed per-agent limit (3000 + 1200 = 4200 > 4000)
        result = guard.check_llm_invocation("agent1", "gpt-4", requested_tokens=1200)
        
        assert not result.allowed
        assert "agent token limit" in result.reason.lower()
    
    def test_llm_cost_limit_enforced(self):
        """LLM invocation respects cost limits."""
        from namel3ss.ast.security import EnvironmentProfile
        
        profile = EnvironmentProfile(
            name="test_prod",
            environment=Environment.PRODUCTION,
            enforce_cost_limits=True
        )
        
        config = SecurityConfig(
            current_environment=Environment.PRODUCTION,
            environments={Environment.PRODUCTION: profile},
            global_policy=SecurityPolicy(
                name="budget",
                max_cost_per_request=0.50
            )
        )
        
        guard = SecurityGuard(config)
        
        # Request exceeds cost limit
        result = guard.check_llm_invocation(
            "agent1",
            "gpt-4",
            requested_tokens=1000,
            estimated_cost=0.75
        )
        
        assert not result.allowed
        assert "cost limit" in result.reason.lower()
    
    def test_audit_log_records_events(self):
        """Audit log captures security events."""
        config = SecurityConfig(audit_enabled=True)
        guard = SecurityGuard(config)
        
        # Make some calls
        guard.check_tool_invocation("agent1", "tool1")
        guard.check_llm_invocation("agent1", "gpt-4", 1000)
        
        audit_log = guard.get_audit_log()
        
        assert len(audit_log) == 2
        assert audit_log[0].event_type == "tool_call"
        assert audit_log[1].event_type == "llm_call"
    
    def test_agent_scope_isolation(self):
        """Different agents have independent rate limits."""
        from namel3ss.ast.security import EnvironmentProfile
        
        profile = EnvironmentProfile(
            name="test",
            environment=Environment.DEVELOPMENT,
            enforce_rate_limits=True
        )
        
        config = SecurityConfig(
            current_environment=Environment.DEVELOPMENT,
            environments={Environment.DEVELOPMENT: profile},
            global_policy=SecurityPolicy(
                name="per_agent",
                rate_limit_requests_per_minute=2,
                rate_limit_scope="agent"
            )
        )
        
        guard = SecurityGuard(config)
        
        # Agent1: Check and record twice
        result = guard.check_tool_invocation("agent1", "tool1")
        assert result.allowed
        guard.record_tool_completion("agent1", "tool1")
        
        result = guard.check_tool_invocation("agent1", "tool1")
        assert result.allowed
        guard.record_tool_completion("agent1", "tool1")
        
        # Agent1: Third request should be denied
        result1 = guard.check_tool_invocation("agent1", "tool1")
        assert not result1.allowed
        
        # Agent2 should still be allowed
        result2 = guard.check_tool_invocation("agent2", "tool1")
        assert result2.allowed
    
    def test_reset_clears_state(self):
        """Reset clears all runtime state."""
        guard = SecurityGuard()
        
        # Build up state
        guard.record_tool_completion("agent1", "tool1")
        guard.record_llm_completion("agent1", "gpt-4", 1000)
        guard.check_tool_invocation("agent1", "tool1")
        
        assert len(guard.get_audit_log()) > 0
        assert guard.token_counter.get_total_usage() > 0
        
        # Reset
        guard.reset()
        
        assert len(guard.get_audit_log()) == 0
        assert guard.token_counter.get_total_usage() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
