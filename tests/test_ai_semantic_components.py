"""
Integration test for AI semantic components end-to-end pipeline.

Tests the complete flow: .ai file → Parser → AST → IR → Codegen → React
"""
import pytest
from pathlib import Path
from namel3ss.parser import Parser
from namel3ss.ir.builder import build_frontend_ir
from namel3ss.ast.pages import (
    ChatThread,
    AgentPanel,
    ToolCallView,
    LogView,
    EvaluationResult,
    DiffView,
)


class TestAIComponentsParsing:
    """Test parsing of AI semantic components."""
    
    def test_parse_chat_thread(self):
        """Test chat_thread parsing."""
        source = '''
page "test":
  chat_thread "conv":
    messages_binding: "agent.messages"
    streaming_enabled: true
    show_timestamps: true
'''
        parser = Parser(source.strip())
        module = parser.parse()
        app = module.body[0]
        
        assert app is not None
        assert len(app.pages) == 1
        page = app.pages[0]
        assert len(page.statements) == 1
        stmt = page.statements[0]
        assert isinstance(stmt, ChatThread)
        assert stmt.id == "conv"
        assert stmt.messages_binding == "agent.messages"
        assert stmt.streaming_enabled is True
        assert stmt.show_timestamps is True
    
    def test_parse_agent_panel(self):
        """Test agent_panel parsing."""
        source = '''
page "test":
  agent_panel "status":
    agent_binding: "current_agent"
    show_tokens: true
    show_cost: true
'''
        parser = Parser(source.strip())
        module = parser.parse()
        app = module.body[0]
        
        page = app.pages[0]
        assert len(page.statements) == 1
        stmt = page.statements[0]
        assert isinstance(stmt, AgentPanel)
        assert stmt.id == "status"
        assert stmt.agent_binding == "current_agent"
        assert stmt.show_tokens is True
        assert stmt.show_cost is True
    
    def test_parse_tool_call_view(self):
        """Test tool_call_view parsing."""
        source = '''
page "test":
  tool_call_view "tools":
    calls_binding: "run.tools"
    show_inputs: true
    show_outputs: true
    expandable: true
'''
        parser = Parser(source.strip())
        module = parser.parse()
        app = module.body[0]
        page = app.pages[0]
        
        stmt = page.statements[0]
        assert isinstance(stmt, ToolCallView)
        assert stmt.id == "tools"
        assert stmt.calls_binding == "run.tools"
        assert stmt.expandable is True
    
    def test_parse_log_view(self):
        """Test log_view parsing."""
        source = '''
page "test":
  log_view "logs":
    logs_binding: "system.logs"
    search_enabled: true
    virtualized: true
    max_entries: 1000
'''
        parser = Parser(source.strip())
        module = parser.parse()
        app = module.body[0]
        page = app.pages[0]
        
        stmt = page.statements[0]
        assert isinstance(stmt, LogView)
        assert stmt.id == "logs"
        assert stmt.logs_binding == "system.logs"
        assert stmt.search_enabled is True
        assert stmt.virtualized is True
        assert stmt.max_entries == 1000
    
    def test_parse_evaluation_result(self):
        """Test evaluation_result parsing."""
        source = '''
page "test":
  evaluation_result "eval":
    eval_run_binding: "eval.run_123"
    show_summary: true
    show_histograms: true
    primary_metric: "accuracy"
    max_error_examples: 10
'''
        parser = Parser(source.strip())
        module = parser.parse()
        app = module.body[0]
        page = app.pages[0]
        
        stmt = page.statements[0]
        assert isinstance(stmt, EvaluationResult)
        assert stmt.id == "eval"
        assert stmt.eval_run_binding == "eval.run_123"
        assert stmt.primary_metric == "accuracy"
        assert stmt.max_error_examples == 10
    
    def test_parse_diff_view(self):
        """Test diff_view parsing."""
        source = '''
page "test":
  diff_view "comparison":
    left_binding: "v1.output"
    right_binding: "v2.output"
    mode: "split"
    content_type: "code"
    language: "python"
    show_line_numbers: true
'''
        parser = Parser(source.strip())
        module = parser.parse()
        app = module.body[0]
        page = app.pages[0]
        
        stmt = page.statements[0]
        assert isinstance(stmt, DiffView)
        assert stmt.id == "comparison"
        assert stmt.left_binding == "v1.output"
        assert stmt.right_binding == "v2.output"
        assert stmt.mode == "split"
        assert stmt.content_type == "code"
        assert stmt.language == "python"
    
    def test_all_components_in_one_page(self):
        """Test parsing all AI components in a single page."""
        source = '''
page "dashboard":
  chat_thread "conv":
    messages_binding: "agent.messages"
  
  agent_panel "status":
    agent_binding: "agent"
  
  tool_call_view "tools":
    calls_binding: "run.tools"
  
  log_view "logs":
    logs_binding: "system.logs"
  
  evaluation_result "eval":
    eval_run_binding: "eval.run"
  
  diff_view "diff":
    left_binding: "v1"
    right_binding: "v2"
'''
        parser = Parser(source.strip())
        module = parser.parse()
        app = module.body[0]
        page = app.pages[0]
        
        assert len(page.statements) == 6
        assert isinstance(page.statements[0], ChatThread)
        assert isinstance(page.statements[1], AgentPanel)
        assert isinstance(page.statements[2], ToolCallView)
        assert isinstance(page.statements[3], LogView)
        assert isinstance(page.statements[4], EvaluationResult)
        assert isinstance(page.statements[5], DiffView)


class TestAIComponentsIRGeneration:
    """Test IR generation for AI components."""
    
    def test_chat_thread_ir_generation(self):
        """Test ChatThread AST → IR conversion."""
        source = '''
page "test":
  chat_thread "conv":
    messages_binding: "agent.messages"
    streaming_enabled: true
'''
        parser = Parser(source.strip())
        module = parser.parse()
        app = module.body[0]
        
        frontend_ir = build_frontend_ir(app)
        
        # Check that IR was generated
        assert frontend_ir is not None
        assert len(frontend_ir.pages) == 1
        page_ir = frontend_ir.pages[0]
        assert len(page_ir.components) == 1
        
        component_ir = page_ir.components[0]
        assert component_ir.type == "chat_thread"
        assert component_ir.props["messages_binding"] == "agent.messages"
        assert component_ir.props["streaming_enabled"] is True


class TestAIComponentsCodegen:
    """Test React component generation."""
    
    def test_ai_components_module_imports(self):
        """Test that AI components module can be imported."""
        from namel3ss.codegen.frontend.react.ai_components import (
            write_chat_thread_component,
            write_agent_panel_component,
            write_tool_call_view_component,
            write_log_view_component,
            write_evaluation_result_component,
            write_diff_view_component,
            write_all_ai_components,
        )
        
        # All functions should be callable
        assert callable(write_chat_thread_component)
        assert callable(write_agent_panel_component)
        assert callable(write_tool_call_view_component)
        assert callable(write_log_view_component)
        assert callable(write_evaluation_result_component)
        assert callable(write_diff_view_component)
        assert callable(write_all_ai_components)
    
    def test_ai_components_tsx_generation(self, tmp_path):
        """Test React TSX file generation for AI components."""
        from namel3ss.codegen.frontend.react.ai_components import write_all_ai_components
        
        components_dir = tmp_path / "components"
        components_dir.mkdir()
        
        write_all_ai_components(components_dir)
        
        # Check all 6 component files were created
        assert (components_dir / "ChatThread.tsx").exists()
        assert (components_dir / "AgentPanel.tsx").exists()
        assert (components_dir / "ToolCallView.tsx").exists()
        assert (components_dir / "LogView.tsx").exists()
        assert (components_dir / "EvaluationResult.tsx").exists()
        assert (components_dir / "DiffView.tsx").exists()
        
        # Verify content structure
        chat_thread_content = (components_dir / "ChatThread.tsx").read_text(encoding='utf-8')
        assert "export default function ChatThread" in chat_thread_content
        assert "ChatThreadProps" in chat_thread_content
        assert "messages_binding" in chat_thread_content
        
        agent_panel_content = (components_dir / "AgentPanel.tsx").read_text(encoding='utf-8')
        assert "export default function AgentPanel" in agent_panel_content
        assert "agent_binding" in agent_panel_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
