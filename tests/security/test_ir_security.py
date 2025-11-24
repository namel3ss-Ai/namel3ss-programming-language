"""
Tests for IR security metadata.

Tests that security metadata is correctly represented in the IR and can be
serialized/deserialized.
"""

import pytest
from dataclasses import asdict
from namel3ss.ir.spec import (
    BackendIR,
    AgentSpec,
    ToolSpec,
    EndpointIR,
    TypeSpec,
    HTTPMethod,
)


def _simple_type_spec():
    """Helper to create a simple TypeSpec for testing."""
    return TypeSpec(kind="object", fields=[])


class TestAgentSpecSecurity:
    """Test AgentSpec security metadata."""
    
    def test_agent_spec_with_security(self):
        """Agent can have security metadata."""
        agent = AgentSpec(
            name="secure_agent",
            nodes=[{"id": "node1", "type": "llm"}],
            edges=[],
            entry_point="node1",
            handoff_logic={},
            state_schema=_simple_type_spec(),
            allowed_tools=["web_search", "calculator"],
            capabilities=["internet_access", "compute"],
            permission_level="STANDARD",
            security_policy={
                "max_cost_per_run": 1.0,
                "max_tokens": 1000,
                "allowed_domains": ["example.com"]
            }
        )
        
        assert agent.allowed_tools == ["web_search", "calculator"]
        assert agent.capabilities == ["internet_access", "compute"]
        assert agent.permission_level == "STANDARD"
        assert agent.security_policy["max_cost_per_run"] == 1.0
    
    def test_agent_spec_without_security(self):
        """Agent without security has default empty values."""
        agent = AgentSpec(
            name="simple_agent",
            nodes=[],
            edges=[],
            entry_point="start",
            handoff_logic={},
            state_schema=_simple_type_spec()
        )
        
        assert agent.allowed_tools == []
        assert agent.capabilities == []
        assert agent.permission_level is None
        assert agent.security_policy is None
    
    def test_agent_spec_serialization(self):
        """Agent security metadata serializes correctly."""
        agent = AgentSpec(
            name="agent1",
            nodes=[],
            edges=[],
            entry_point="start",
            handoff_logic={},
            state_schema=_simple_type_spec(),
            allowed_tools=["tool1", "tool2"],
            capabilities=["cap1"],
            permission_level="ELEVATED"
        )
        
        data = asdict(agent)
        assert data["allowed_tools"] == ["tool1", "tool2"]
        assert data["capabilities"] == ["cap1"]
        assert data["permission_level"] == "ELEVATED"


class TestToolSpecSecurity:
    """Test ToolSpec security metadata."""
    
    def test_tool_spec_with_security(self):
        """Tool can have security metadata."""
        tool = ToolSpec(
            name="web_search",
            description="Search the web",
            input_schema=_simple_type_spec(),
            output_schema=_simple_type_spec(),
            implementation_type="http",
            implementation_ref="https://api.example.com/search",
            required_capabilities=["internet_access"],
            permission_level="STANDARD",
            timeout_seconds=30.0,
            rate_limit_per_minute=60
        )
        
        assert tool.required_capabilities == ["internet_access"]
        assert tool.permission_level == "STANDARD"
        assert tool.timeout_seconds == 30.0
        assert tool.rate_limit_per_minute == 60
    
    def test_tool_spec_without_security(self):
        """Tool without security has default empty values."""
        tool = ToolSpec(
            name="simple_tool",
            description="Simple tool",
            input_schema=_simple_type_spec(),
            output_schema=_simple_type_spec(),
            implementation_type="python",
            implementation_ref="module.function"
        )
        
        assert tool.required_capabilities == []
        assert tool.permission_level is None
        assert tool.timeout_seconds is None
        assert tool.rate_limit_per_minute is None
    
    def test_tool_spec_serialization(self):
        """Tool security metadata serializes correctly."""
        tool = ToolSpec(
            name="tool1",
            description="Test tool",
            input_schema=_simple_type_spec(),
            output_schema=_simple_type_spec(),
            implementation_type="python",
            implementation_ref="test.func",
            required_capabilities=["filesystem"],
            permission_level="ELEVATED",
            timeout_seconds=10.0
        )
        
        data = asdict(tool)
        assert data["required_capabilities"] == ["filesystem"]
        assert data["permission_level"] == "ELEVATED"
        assert data["timeout_seconds"] == 10.0


class TestEndpointIRSecurity:
    """Test EndpointIR security metadata."""
    
    def test_endpoint_with_security(self):
        """Endpoint can have security metadata."""
        endpoint = EndpointIR(
            path="/api/data",
            method=HTTPMethod.GET,
            input_schema=_simple_type_spec(),
            output_schema=_simple_type_spec(),
            handler_type="agent",
            handler_ref="data_agent",
            auth_required=True,
            required_permission_level="STANDARD",
            allowed_capabilities=["database_read"]
        )
        
        assert endpoint.required_permission_level == "STANDARD"
        assert endpoint.allowed_capabilities == ["database_read"]
    
    def test_endpoint_without_security(self):
        """Endpoint without security has default values."""
        endpoint = EndpointIR(
            path="/api/public",
            method=HTTPMethod.GET,
            input_schema=_simple_type_spec(),
            output_schema=_simple_type_spec(),
            handler_type="agent",
            handler_ref="public_agent"
        )
        
        assert endpoint.required_permission_level is None
        assert endpoint.allowed_capabilities == []


class TestBackendIRSecurity:
    """Test BackendIR security configuration."""
    
    def test_backend_ir_with_security(self):
        """BackendIR can have security configuration."""
        backend = BackendIR(
            app_name="secure_app",
            app_version="1.0.0",
            endpoints=[],
            prompts=[],
            agents=[],
            tools=[],
            chains=[],
            datasets=[],
            frames=[],
            insights=[],
            memory=None,
            security_config={
                "default_environment": "production",
                "cost_tracking_enabled": True
            },
            agent_tool_mappings={
                "agent1": ["tool1", "tool2"],
                "agent2": ["tool2", "tool3"]
            },
            capability_requirements={
                "tool1": ["internet_access"],
                "tool2": ["compute"],
                "tool3": ["database_write"]
            },
            permission_levels={
                "agent1": "STANDARD",
                "agent2": "ELEVATED",
                "tool1": "STANDARD"
            }
        )
        
        assert backend.security_config["default_environment"] == "production"
        assert backend.agent_tool_mappings["agent1"] == ["tool1", "tool2"]
        assert backend.capability_requirements["tool1"] == ["internet_access"]
        assert backend.permission_levels["agent1"] == "STANDARD"
    
    def test_backend_ir_without_security(self):
        """BackendIR without security has default values."""
        backend = BackendIR(
            app_name="simple_app",
            app_version="1.0.0",
            endpoints=[],
            prompts=[],
            agents=[],
            tools=[],
            chains=[],
            datasets=[],
            frames=[],
            insights=[],
            memory=None
        )
        
        assert backend.security_config is None
        assert backend.agent_tool_mappings == {}
        assert backend.capability_requirements == {}
        assert backend.permission_levels == {}
    
    def test_backend_ir_serialization(self):
        """BackendIR security configuration serializes correctly."""
        backend = BackendIR(
            app_name="app",
            app_version="1.0.0",
            endpoints=[],
            prompts=[],
            agents=[],
            tools=[],
            chains=[],
            datasets=[],
            frames=[],
            insights=[],
            memory=None,
            security_config={"env": "dev"},
            agent_tool_mappings={"a1": ["t1"]},
            capability_requirements={"t1": ["c1"]},
            permission_levels={"a1": "STANDARD"}
        )
        
        data = asdict(backend)
        assert data["security_config"] == {"env": "dev"}
        assert data["agent_tool_mappings"] == {"a1": ["t1"]}
        assert data["capability_requirements"] == {"t1": ["c1"]}
        assert data["permission_levels"] == {"a1": "STANDARD"}


class TestSecurityMetadataIntegration:
    """Test integration of security metadata across IR components."""
    
    def test_complete_secure_application(self):
        """Full application with security metadata."""
        # Create agent with security
        agent = AgentSpec(
            name="secure_agent",
            nodes=[],
            edges=[],
            entry_point="start",
            handoff_logic={},
            state_schema=_simple_type_spec(),
            allowed_tools=["web_search"],
            capabilities=["internet_access"],
            permission_level="STANDARD"
        )
        
        # Create tool with security
        tool = ToolSpec(
            name="web_search",
            description="Search",
            input_schema=_simple_type_spec(),
            output_schema=_simple_type_spec(),
            implementation_type="http",
            implementation_ref="https://api.example.com",
            required_capabilities=["internet_access"],
            permission_level="STANDARD",
            timeout_seconds=30.0
        )
        
        # Create endpoint with security
        endpoint = EndpointIR(
            path="/api/search",
            method=HTTPMethod.POST,
            input_schema=_simple_type_spec(),
            output_schema=_simple_type_spec(),
            handler_type="agent",
            handler_ref="secure_agent",
            auth_required=True,
            required_permission_level="STANDARD"
        )
        
        # Create backend IR with security
        backend = BackendIR(
            app_name="secure_search_app",
            app_version="1.0.0",
            endpoints=[endpoint],
            prompts=[],
            agents=[agent],
            tools=[tool],
            chains=[],
            datasets=[],
            frames=[],
            insights=[],
            memory=None,
            security_config={
                "default_environment": "production",
                "cost_tracking_enabled": True
            },
            agent_tool_mappings={
                "secure_agent": ["web_search"]
            },
            capability_requirements={
                "web_search": ["internet_access"]
            },
            permission_levels={
                "secure_agent": "STANDARD",
                "web_search": "STANDARD"
            }
        )
        
        # Verify security metadata is correctly set
        assert len(backend.agents) == 1
        assert backend.agents[0].allowed_tools == ["web_search"]
        assert backend.agents[0].capabilities == ["internet_access"]
        
        assert len(backend.tools) == 1
        assert backend.tools[0].required_capabilities == ["internet_access"]
        assert backend.tools[0].timeout_seconds == 30.0
        
        assert len(backend.endpoints) == 1
        assert backend.endpoints[0].required_permission_level == "STANDARD"
        
        assert backend.agent_tool_mappings["secure_agent"] == ["web_search"]
        assert backend.capability_requirements["web_search"] == ["internet_access"]
    
    def test_backward_compatibility(self):
        """IR without security metadata still works."""
        # Old-style IR without security fields
        agent = AgentSpec(
            name="legacy_agent",
            nodes=[],
            edges=[],
            entry_point="start",
            handoff_logic={},
            state_schema=_simple_type_spec()
        )
        
        tool = ToolSpec(
            name="legacy_tool",
            description="Old tool",
            input_schema=_simple_type_spec(),
            output_schema=_simple_type_spec(),
            implementation_type="python",
            implementation_ref="module.func"
        )
        
        backend = BackendIR(
            app_name="legacy_app",
            app_version="1.0.0",
            endpoints=[],
            prompts=[],
            agents=[agent],
            tools=[tool],
            chains=[],
            datasets=[],
            frames=[],
            insights=[],
            memory=None
        )
        
        # Should work without errors
        assert backend.app_name == "legacy_app"
        assert agent.allowed_tools == []
        assert tool.required_capabilities == []
