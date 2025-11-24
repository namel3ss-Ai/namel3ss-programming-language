"""
Tests for Namel3ss security model - capability validation.

Tests static validation of security constraints:
- Tool access validation
- Capability grants
- Permission levels
- Policy validation
"""

import pytest

from namel3ss.ast.agents import AgentDefinition
from namel3ss.ast.ai_tools import ToolDefinition
from namel3ss.ast.application import App
from namel3ss.ast.security import (
    PermissionLevel,
    Environment,
    SecurityPolicy,
)
from namel3ss.security.validation import (
    validate_tool_access,
    validate_capability_grant,
    validate_permission_level,
    validate_security_policy,
    validate_application_security,
    SecurityValidator,
)
from namel3ss.security.config import SecurityConfig, reset_security_config


@pytest.fixture(autouse=True)
def reset_config():
    """Reset security config before each test."""
    reset_security_config()
    yield
    reset_security_config()


class TestToolAccessValidation:
    """Tests for validate_tool_access function."""
    
    def test_valid_tool_access(self):
        """Agent with proper tool grant can access tool."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            tool_names=["web_search"],
            capabilities=["network", "http_read"],
            permission_level="network"
        )
        
        tool = ToolDefinition(
            name="web_search",
            description="Search the web",
            permission_level="network",
            required_capabilities=["network", "http_read"]
        )
        
        app = App(name="test", tools=[tool], agents=[agent])
        
        result = validate_tool_access(agent, tool, app)
        
        assert result.allowed
        assert len(result.violations) == 0
    
    def test_tool_not_in_agent_list(self):
        """Agent cannot access tool not in its tools list."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            tool_names=[],  # No tools granted
            capabilities=["network"],
            permission_level="network"
        )
        
        tool = ToolDefinition(
            name="web_search",
            description="Search the web",
            permission_level="network",
            required_capabilities=["network"]
        )
        
        app = App(name="test", tools=[tool], agents=[agent])
        
        result = validate_tool_access(agent, tool, app)
        
        assert not result.allowed
        assert len(result.violations) > 0
        assert "not in agent" in result.violations[0].lower()
    
    def test_missing_capability(self):
        """Agent lacks required capability for tool."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            tool_names=["web_search"],
            capabilities=[],  # No capabilities
            permission_level="network"
        )
        
        tool = ToolDefinition(
            name="web_search",
            description="Search the web",
            permission_level="network",
            required_capabilities=["network", "http_read"]
        )
        
        app = App(name="test", tools=[tool], agents=[agent])
        
        result = validate_tool_access(agent, tool, app)
        
        assert not result.allowed
        assert any("capabilit" in v.lower() for v in result.violations)
    
    def test_insufficient_permission_level(self):
        """Agent permission level insufficient for tool."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            tool_names=["admin_tool"],
            capabilities=["admin"],
            permission_level="read_only"  # Too low
        )
        
        tool = ToolDefinition(
            name="admin_tool",
            description="Admin operations",
            permission_level="admin",  # Requires admin
            required_capabilities=["admin"]
        )
        
        app = App(name="test", tools=[tool], agents=[agent])
        
        result = validate_tool_access(agent, tool, app)
        
        assert not result.allowed
        assert any("permission" in v.lower() for v in result.violations)


class TestCapabilityValidation:
    """Tests for validate_capability_grant function."""
    
    def test_all_capabilities_granted(self):
        """Agent has all required capabilities."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            capabilities=["network", "http_read", "http_write"]
        )
        
        tool = ToolDefinition(
            name="api_tool",
            description="API tool",
            required_capabilities=["network", "http_write"]
        )
        
        result = validate_capability_grant(agent, tool)
        
        assert result.allowed
        assert len(result.violations) == 0
    
    def test_missing_one_capability(self):
        """Agent missing one required capability."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            capabilities=["network"]
        )
        
        tool = ToolDefinition(
            name="api_tool",
            description="API tool",
            required_capabilities=["network", "http_write"]
        )
        
        result = validate_capability_grant(agent, tool)
        
        assert not result.allowed
        assert "http_write" in result.violations[0]
    
    def test_no_capabilities_required(self):
        """Tool requires no capabilities."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            capabilities=[]
        )
        
        tool = ToolDefinition(
            name="simple_tool",
            description="Simple tool",
            required_capabilities=[]
        )
        
        result = validate_capability_grant(agent, tool)
        
        assert result.allowed


class TestPermissionLevelValidation:
    """Tests for validate_permission_level function."""
    
    def test_equal_permission_levels(self):
        """Agent and tool have same permission level."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            permission_level="read_write"
        )
        
        tool = ToolDefinition(
            name="db_tool",
            description="Database tool",
            permission_level="read_write"
        )
        
        result = validate_permission_level(agent, tool)
        
        assert result.allowed
    
    def test_agent_higher_permission(self):
        """Agent has higher permission than tool requires."""
        agent = AgentDefinition(
            name="admin_agent",
            llm_name="gpt-4",
            permission_level="admin"
        )
        
        tool = ToolDefinition(
            name="read_tool",
            description="Read-only tool",
            permission_level="read_only"
        )
        
        result = validate_permission_level(agent, tool)
        
        assert result.allowed
    
    def test_agent_lower_permission(self):
        """Agent has lower permission than tool requires."""
        agent = AgentDefinition(
            name="reader",
            llm_name="gpt-4",
            permission_level="read_only"
        )
        
        tool = ToolDefinition(
            name="admin_tool",
            description="Admin tool",
            permission_level="admin"
        )
        
        result = validate_permission_level(agent, tool)
        
        assert not result.allowed
        assert "insufficient" in result.violations[0].lower()
    
    def test_invalid_agent_permission(self):
        """Invalid permission level for agent."""
        agent = AgentDefinition(
            name="bad_agent",
            llm_name="gpt-4",
            permission_level="invalid_level"
        )
        
        tool = ToolDefinition(
            name="tool",
            description="Tool",
            permission_level="read_only"
        )
        
        result = validate_permission_level(agent, tool)
        
        assert not result.allowed
        assert "invalid" in result.violations[0].lower()
    
    def test_permission_warning_on_elevated(self):
        """Warning generated for elevated permissions in dev mode."""
        config = SecurityConfig(current_environment=Environment.DEVELOPMENT)
        
        agent = AgentDefinition(
            name="admin",
            llm_name="gpt-4",
            permission_level="admin"
        )
        
        tool = ToolDefinition(
            name="tool",
            description="Tool",
            permission_level="read_only"
        )
        
        result = validate_permission_level(agent, tool, config)
        
        # Should be allowed but with warning
        assert result.allowed
        assert len(result.warnings) > 0
        assert "elevated" in result.warnings[0].lower()


class TestSecurityPolicyValidation:
    """Tests for validate_security_policy function."""
    
    def test_valid_policy(self):
        """Well-formed policy passes validation."""
        policy = SecurityPolicy(
            name="test_policy",
            rate_limit_requests_per_minute=60,
            tool_timeout_seconds=30.0,
            llm_timeout_seconds=120.0,
            max_tokens_per_request=4000,
            max_concurrent_tool_calls=10,
            max_concurrent_llm_calls=5,
        )
        
        result = validate_security_policy(policy)
        
        assert result.allowed
        assert len(result.violations) == 0
    
    def test_negative_timeout(self):
        """Negative timeout is invalid."""
        policy = SecurityPolicy(
            name="bad_policy",
            tool_timeout_seconds=-5.0  # Invalid
        )
        
        result = validate_security_policy(policy)
        
        assert not result.allowed
        assert "timeout" in result.violations[0].lower()
    
    def test_negative_rate_limit(self):
        """Negative rate limit is invalid."""
        policy = SecurityPolicy(
            name="bad_policy",
            rate_limit_requests_per_minute=-10  # Invalid
        )
        
        result = validate_security_policy(policy)
        
        assert not result.allowed
        assert "rate_limit" in result.violations[0].lower()
    
    def test_zero_rate_limit_warning(self):
        """Zero rate limit generates warning."""
        policy = SecurityPolicy(
            name="strict_policy",
            rate_limit_requests_per_minute=0  # Will block everything
        )
        
        result = validate_security_policy(policy)
        
        # Technically valid but warns
        assert result.allowed
        assert len(result.warnings) > 0
        assert "block all" in result.warnings[0].lower()
    
    def test_invalid_fail_mode(self):
        """Invalid fail mode is caught."""
        policy = SecurityPolicy(
            name="bad_policy",
            fail_mode="invalid"  # Must be 'closed' or 'open'
        )
        
        result = validate_security_policy(policy)
        
        assert not result.allowed
        assert "fail_mode" in result.violations[0].lower()
    
    def test_negative_concurrency_limit(self):
        """Negative or zero concurrency limit is invalid."""
        policy = SecurityPolicy(
            name="bad_policy",
            max_concurrent_tool_calls=0  # Invalid
        )
        
        result = validate_security_policy(policy)
        
        assert not result.allowed
        assert "concurrent" in result.violations[0].lower()


class TestApplicationValidation:
    """Tests for validate_application_security function."""
    
    def test_valid_application(self):
        """Application with proper security setup passes."""
        tool = ToolDefinition(
            name="web_search",
            description="Search",
            permission_level="network",
            required_capabilities=["network"]
        )
        
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            tool_names=["web_search"],
            capabilities=["network"],
            permission_level="network"
        )
        
        app = App(
            name="test_app",
            tools=[tool],
            agents=[agent]
        )
        
        result = validate_application_security(app)
        
        assert result.allowed
        assert len(result.violations) == 0
    
    def test_undeclared_tool_reference(self):
        """Agent references undeclared tool."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            tool_names=["nonexistent_tool"],  # Not declared
            capabilities=["network"],
            permission_level="network"
        )
        
        app = App(
            name="test_app",
            tools=[],  # No tools
            agents=[agent]
        )
        
        result = validate_application_security(app)
        
        assert not result.allowed
        assert "undeclared" in result.violations[0].lower()
    
    def test_multiple_violations(self):
        """Application with multiple security violations."""
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            permission_level="admin",
            required_capabilities=["admin"]
        )
        
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            permission_level="network",
            required_capabilities=["network", "http_write"]
        )
        
        agent1 = AgentDefinition(
            name="agent1",
            llm_name="gpt-4",
            tool_names=["tool1"],
            capabilities=[],  # Missing capabilities
            permission_level="read_only"  # Insufficient permission
        )
        
        agent2 = AgentDefinition(
            name="agent2",
            llm_name="gpt-4",
            tool_names=["tool2", "nonexistent"],  # One undeclared
            capabilities=["network"],  # Missing http_write
            permission_level="network"
        )
        
        app = App(
            name="test_app",
            tools=[tool1, tool2],
            agents=[agent1, agent2]
        )
        
        result = validate_application_security(app)
        
        assert not result.allowed
        assert len(result.violations) >= 3  # Multiple issues


class TestSecurityValidator:
    """Tests for SecurityValidator class."""
    
    def test_validator_accumulates_errors(self):
        """Validator accumulates errors across multiple checks."""
        validator = SecurityValidator()
        
        agent = AgentDefinition(
            name="bad_agent",
            llm_name="gpt-4",
            tool_names=["tool1", "tool2"],
            capabilities=[],
            permission_level="read_only"
        )
        
        tool1 = ToolDefinition(
            name="tool1",
            description="Tool 1",
            permission_level="admin",
            required_capabilities=["admin"]
        )
        
        tool2 = ToolDefinition(
            name="tool2",
            description="Tool 2",
            permission_level="network",
            required_capabilities=["network"]
        )
        
        app = App(name="test", tools=[tool1, tool2], agents=[agent])
        
        # Run multiple validations
        validator.validate_tool_access(agent, tool1, app)
        validator.validate_tool_access(agent, tool2, app)
        
        assert validator.has_errors()
        assert len(validator.errors) >= 2
    
    def test_validator_clear(self):
        """Validator can clear accumulated errors."""
        validator = SecurityValidator()
        
        # Generate some errors
        policy = SecurityPolicy(name="bad", tool_timeout_seconds=-1.0)
        validator.validate_policy(policy)
        
        assert validator.has_errors()
        
        # Clear
        validator.clear()
        
        assert not validator.has_errors()
        assert len(validator.errors) == 0
    
    def test_validator_summary(self):
        """Validator generates formatted summary."""
        validator = SecurityValidator()
        
        policy = SecurityPolicy(name="bad", tool_timeout_seconds=-1.0)
        validator.validate_policy(policy)
        
        summary = validator.get_summary()
        
        assert "Security Errors" in summary
        assert "timeout" in summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
