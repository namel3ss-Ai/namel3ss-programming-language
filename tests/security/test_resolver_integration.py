"""
Tests for resolver security integration.

Tests that the resolver properly validates security constraints during resolution.
"""

import pytest

from namel3ss.ast import App, Module, Program
from namel3ss.ast.agents import AgentDefinition
from namel3ss.ast.ai_tools import ToolDefinition, LLMDefinition
from namel3ss.ast.security import PermissionLevel
from namel3ss.resolver import resolve_program, ModuleResolutionError
from namel3ss.security.config import reset_security_config


@pytest.fixture(autouse=True)
def reset_security():
    """Reset security config before each test."""
    reset_security_config()
    yield
    reset_security_config()


class TestResolverSecurityIntegration:
    """Tests for resolver security validation."""
    
    @pytest.mark.skip(reason="Typechecker expects legacy tool.type field")
    def test_valid_agent_tool_access(self):
        """Agent with proper capabilities can access tool."""
        llm = LLMDefinition(name="gpt-4", provider="openai", model="gpt-4")
        
        tool = ToolDefinition(
            name="web_search",
            description="Search the web",
            required_capabilities=["http_read"],
            permission_level=PermissionLevel.NETWORK
        )
        
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            capabilities=["http_read", "http_write"],
            permission_level=PermissionLevel.NETWORK,
            tool_names=["web_search"]
        )
        
        app = App(
            name="test_app",
            llms=[llm],
            tools=[tool],
            agents=[agent]
        )
        
        module = Module(
            name="main",
            path="main.n3",
            body=[app],
            has_explicit_app=True
        )
        
        program = Program(modules=[module])
        
        # Should not raise
        resolved = resolve_program(program)
        assert resolved.app is not None
    
    def test_agent_missing_capability(self):
        """Agent without required capability cannot access tool."""
        tool = ToolDefinition(
            name="web_search",
            description="Search the web",
            required_capabilities=["http_read", "network_access"],
            permission_level=PermissionLevel.NETWORK
        )
        
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            capabilities=["http_read"],  # Missing network_access
            permission_level=PermissionLevel.NETWORK,
            tool_names=["web_search"]
        )
        
        llm = LLMDefinition(name="gpt-4", provider="openai", model="gpt-4")
        
        app = App(
            name="test_app",
            llms=[llm],
            tools=[tool],
            agents=[agent]
        )
        
        module = Module(
            name="main",
            path="main.n3",
            body=[app],
            has_explicit_app=True
        )
        
        program = Program(modules=[module])
        
        # Should raise security violation
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        assert "capabilit" in str(exc_info.value).lower()
    
    def test_agent_insufficient_permission_level(self):
        """Agent with lower permission level cannot access tool."""
        tool = ToolDefinition(
            name="file_writer",
            description="Write files",
            required_capabilities=["filesystem_write"],
            permission_level=PermissionLevel.FILESYSTEM
        )
        
        agent = AgentDefinition(
            name="writer",
            llm_name="gpt-4",
            capabilities=["filesystem_write"],
            permission_level=PermissionLevel.READ_ONLY,  # Too low
            tool_names=["file_writer"]
        )
        
        llm = LLMDefinition(name="gpt-4", provider="openai", model="gpt-4")
        
        app = App(
            name="test_app",
            llms=[llm],
            tools=[tool],
            agents=[agent]
        )
        
        module = Module(
            name="main",
            path="main.n3",
            body=[app],
            has_explicit_app=True
        )
        
        program = Program(modules=[module])
        
        # Should raise security violation
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        assert "permission" in str(exc_info.value).lower()
    
    def test_agent_references_undeclared_tool(self):
        """Agent cannot reference tool that doesn't exist."""
        agent = AgentDefinition(
            name="researcher",
            llm_name="gpt-4",
            capabilities=["http_read"],
            permission_level=PermissionLevel.NETWORK,
            tool_names=["nonexistent_tool"]
        )
        
        llm = LLMDefinition(name="gpt-4", provider="openai", model="gpt-4")
        
        app = App(
            name="test_app",
            llms=[llm],
            tools=[],  # No tools declared
            agents=[agent]
        )
        
        module = Module(
            name="main",
            path="main.n3",
            body=[app],
            has_explicit_app=True
        )
        
        program = Program(modules=[module])
        
        # Should raise security violation
        with pytest.raises(ModuleResolutionError) as exc_info:
            resolve_program(program)
        
        assert "nonexistent_tool" in str(exc_info.value)
    
    def test_empty_app_no_security_errors(self):
        """Empty app with no agents or tools passes validation."""
        app = App(
            name="test_app",
            tools=[],
            agents=[]
        )
        
        module = Module(
            name="main",
            path="main.n3",
            body=[app],
            has_explicit_app=True
        )
        
        program = Program(modules=[module])
        
        # Should not raise
        resolved = resolve_program(program)
        assert resolved.app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
