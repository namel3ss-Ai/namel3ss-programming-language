"""
Tests for IR builder security metadata extraction.

Verifies that security metadata flows from AST through BackendState to IR.
"""

import pytest
from namel3ss.ast import App
from namel3ss.ast.ai import ToolDefinition, LLMDefinition
from namel3ss.ast.agents import AgentDefinition
from namel3ss.ir.builder import build_backend_ir


class TestIRBuilderAgentSecurity:
    """Test agent security metadata extraction."""
    
    def test_agent_with_security_fields(self):
        """Agent security fields flow from AST to IR."""
        app = App(
            name="test_app",
            agents=[
                AgentDefinition(
                    name="secure_agent",
                    llm_name="gpt4",
                    tool_names=["web_search", "calculator"],
                    capabilities=["HTTP_READ", "COMPUTE"],
                    permission_level="STANDARD",
                )
            ],
            llms=[
                LLMDefinition(name="gpt4", model="gpt-4")
            ]
        )
        
        ir = build_backend_ir(app)
        
        assert len(ir.agents) == 1
        agent = ir.agents[0]
        assert agent.name == "secure_agent"
        assert agent.allowed_tools == ["web_search", "calculator"]
        assert agent.capabilities == ["HTTP_READ", "COMPUTE"]
        assert agent.permission_level == "STANDARD"
    
    def test_agent_without_security_fields(self):
        """Agent without security has empty security fields."""
        app = App(
            name="test_app",
            agents=[
                AgentDefinition(
                    name="simple_agent",
                    llm_name="gpt4",
                )
            ],
            llms=[
                LLMDefinition(name="gpt4", model="gpt-4")
            ]
        )
        
        ir = build_backend_ir(app)
        
        assert len(ir.agents) == 1
        agent = ir.agents[0]
        assert agent.allowed_tools == []
        assert agent.capabilities == []
        assert agent.permission_level is None
        assert agent.security_policy is None


class TestIRBuilderToolSecurity:
    """Test tool security metadata extraction."""
    
    def test_tool_with_security_fields(self):
        """Tool security fields flow from AST to IR."""
        app = App(
            name="test_app",
            tools=[
                ToolDefinition(
                    name="web_search",
                    description="Search the web",
                    parameters={"query": {"type": "string"}},
                    required_capabilities=["HTTP_READ", "NETWORK"],
                    permission_level="STANDARD",
                    timeout_seconds=30.0,
                    rate_limit_per_minute=60,
                )
            ]
        )
        
        ir = build_backend_ir(app)
        
        assert len(ir.tools) == 1
        tool = ir.tools[0]
        assert tool.name == "web_search"
        assert tool.required_capabilities == ["HTTP_READ", "NETWORK"]
        assert tool.permission_level == "STANDARD"
        assert tool.timeout_seconds == 30.0
        assert tool.rate_limit_per_minute == 60
    
    def test_tool_without_security_fields(self):
        """Tool without security has empty security fields."""
        app = App(
            name="test_app",
            tools=[
                ToolDefinition(
                    name="simple_tool",
                    description="Simple tool",
                    parameters={},
                )
            ]
        )
        
        ir = build_backend_ir(app)
        
        assert len(ir.tools) == 1
        tool = ir.tools[0]
        assert tool.required_capabilities == []
        assert tool.permission_level is None
        assert tool.timeout_seconds is None
        assert tool.rate_limit_per_minute is None


class TestIRBuilderSecurityMappings:
    """Test global security mappings in BackendIR."""
    
    def test_agent_tool_mappings_collected(self):
        """Agent-tool mappings are collected at IR level."""
        app = App(
            name="test_app",
            agents=[
                AgentDefinition(
                    name="agent1",
                    llm_name="gpt4",
                    tool_names=["tool1", "tool2"],
                ),
                AgentDefinition(
                    name="agent2",
                    llm_name="gpt4",
                    tool_names=["tool2", "tool3"],
                ),
            ],
            llms=[
                LLMDefinition(name="gpt4", model="gpt-4")
            ],
            tools=[
                ToolDefinition(name="tool1", description="Tool 1", parameters={}),
                ToolDefinition(name="tool2", description="Tool 2", parameters={}),
                ToolDefinition(name="tool3", description="Tool 3", parameters={}),
            ]
        )
        
        ir = build_backend_ir(app)
        
        assert ir.agent_tool_mappings["agent1"] == ["tool1", "tool2"]
        assert ir.agent_tool_mappings["agent2"] == ["tool2", "tool3"]
    
    def test_capability_requirements_collected(self):
        """Tool capability requirements are collected at IR level."""
        app = App(
            name="test_app",
            tools=[
                ToolDefinition(
                    name="web_search",
                    description="Search",
                    parameters={},
                    required_capabilities=["HTTP_READ", "NETWORK"],
                ),
                ToolDefinition(
                    name="db_query",
                    description="Query",
                    parameters={},
                    required_capabilities=["DATABASE_READ"],
                ),
            ]
        )
        
        ir = build_backend_ir(app)
        
        assert ir.capability_requirements["web_search"] == ["HTTP_READ", "NETWORK"]
        assert ir.capability_requirements["db_query"] == ["DATABASE_READ"]
    
    def test_permission_levels_collected(self):
        """Permission levels for agents and tools are collected at IR level."""
        app = App(
            name="test_app",
            agents=[
                AgentDefinition(
                    name="admin_agent",
                    llm_name="gpt4",
                    permission_level="ADMIN",
                ),
                AgentDefinition(
                    name="readonly_agent",
                    llm_name="gpt4",
                    permission_level="READ_ONLY",
                ),
            ],
            llms=[
                LLMDefinition(name="gpt4", model="gpt-4")
            ],
            tools=[
                ToolDefinition(
                    name="risky_tool",
                    description="Risky",
                    parameters={},
                    permission_level="ADMIN",
                ),
                ToolDefinition(
                    name="safe_tool",
                    description="Safe",
                    parameters={},
                    permission_level="READ_ONLY",
                ),
            ]
        )
        
        ir = build_backend_ir(app)
        
        assert ir.permission_levels["admin_agent"] == "ADMIN"
        assert ir.permission_levels["readonly_agent"] == "READ_ONLY"
        assert ir.permission_levels["risky_tool"] == "ADMIN"
        assert ir.permission_levels["safe_tool"] == "READ_ONLY"
    
    def test_security_config_present(self):
        """Security config is extracted if available."""
        app = App(name="test_app")
        
        ir = build_backend_ir(app)
        
        # Security config may or may not be present depending on environment
        # Just verify it's either None or a dict
        assert ir.security_config is None or isinstance(ir.security_config, dict)


class TestIRBuilderIntegration:
    """Test complete security metadata flow through IR builder."""
    
    def test_complete_secure_application(self):
        """Complete application with security metadata flows through IR."""
        app = App(
            name="secure_app",
            agents=[
                AgentDefinition(
                    name="secure_agent",
                    llm_name="gpt4",
                    tool_names=["web_search"],
                    capabilities=["HTTP_READ", "NETWORK"],
                    permission_level="STANDARD",
                )
            ],
            llms=[
                LLMDefinition(name="gpt4", model="gpt-4")
            ],
            tools=[
                ToolDefinition(
                    name="web_search",
                    description="Search the web",
                    parameters={"query": {"type": "string"}},
                    required_capabilities=["HTTP_READ", "NETWORK"],
                    permission_level="STANDARD",
                    timeout_seconds=30.0,
                    rate_limit_per_minute=60,
                )
            ]
        )
        
        ir = build_backend_ir(app)
        
        # Verify agent security
        assert len(ir.agents) == 1
        agent = ir.agents[0]
        assert agent.allowed_tools == ["web_search"]
        assert agent.capabilities == ["HTTP_READ", "NETWORK"]
        assert agent.permission_level == "STANDARD"
        
        # Verify tool security
        assert len(ir.tools) == 1
        tool = ir.tools[0]
        assert tool.required_capabilities == ["HTTP_READ", "NETWORK"]
        assert tool.permission_level == "STANDARD"
        assert tool.timeout_seconds == 30.0
        assert tool.rate_limit_per_minute == 60
        
        # Verify global mappings
        assert ir.agent_tool_mappings["secure_agent"] == ["web_search"]
        assert ir.capability_requirements["web_search"] == ["HTTP_READ", "NETWORK"]
        assert ir.permission_levels["secure_agent"] == "STANDARD"
        assert ir.permission_levels["web_search"] == "STANDARD"
