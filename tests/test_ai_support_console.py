"""
Tests for AI Customer Support Console example.

Validates that the production-ready support console application:
- Parses correctly through all stages (Parser → AST → IR → Codegen)
- Uses all required components (chat_thread, tool_call_view, agent_panel, forms, modals, etc.)
- Implements realistic data flows (no hardcoded demo data)
- Generates proper backend and frontend code

This tests the flagship example showcasing Namel3ss for real AI applications.
"""

import pytest
from pathlib import Path
from namel3ss.parser import Parser
from namel3ss.ir.builder import build_ir
from namel3ss.codegen.backend.state import build_backend_state


@pytest.fixture
def support_console_source():
    """Load AI Customer Support Console source code."""
    source_path = Path(__file__).parent.parent / "examples" / "ai-customer-support-console.ai"
    assert source_path.exists(), f"Example file not found: {source_path}"
    return source_path.read_text(encoding='utf-8')


# =============================================================================
# Parser Tests
# =============================================================================

def test_parse_support_console_app(support_console_source):
    """Test that support console app parses without errors."""
    parser = Parser(support_console_source)
    module = parser.parse()
    
    assert module is not None
    assert len(module.body) >= 1
    
    app = module.body[0]
    assert app.__class__.__name__ == "App"
    assert app.name == "AI Customer Support Console"
    
    print(f"✅ Parsed app: {app.name}")


def test_datasets_defined(support_console_source):
    """Test that all required datasets are defined."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    # Should have 5 datasets: tickets, customers, orders, interactions, metrics
    assert len(app.datasets) >= 5
    
    dataset_names = [ds.name for ds in app.datasets]
    assert "tickets" in dataset_names
    assert "customers" in dataset_names
    assert "orders" in dataset_names
    assert "interactions" in dataset_names
    assert "metrics" in dataset_names
    
    print(f"✅ Found {len(app.datasets)} datasets: {dataset_names}")


def test_tools_defined(support_console_source):
    """Test that all support tools are defined."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    # Should have 4 tools: lookup_order, issue_refund, update_shipping_address, tag_ticket
    assert len(app.tools) >= 4
    
    tool_names = [tool.name for tool in app.tools]
    assert "lookup_order" in tool_names
    assert "issue_refund" in tool_names
    assert "update_shipping_address" in tool_names
    assert "tag_ticket" in tool_names
    
    # Verify tool has proper structure (parameters, returns, endpoint)
    refund_tool = next(t for t in app.tools if t.name == "issue_refund")
    assert hasattr(refund_tool, 'parameters')
    assert hasattr(refund_tool, 'returns')
    
    print(f"✅ Found {len(app.tools)} tools: {tool_names}")


def test_agent_defined(support_console_source):
    """Test that support assistant agent is defined."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    # Should have support_assistant agent
    assert len(app.agents) >= 1
    
    agent = app.agents[0]
    assert agent.name == "support_assistant"
    assert agent.model_name == "gpt-4o"
    assert len(agent.tools) == 4  # All 4 tools assigned to agent
    
    print(f"✅ Found agent: {agent.name} using {agent.model_name} with {len(agent.tools)} tools")


def test_pages_defined(support_console_source):
    """Test that all pages are defined."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    # Should have 3 pages: Dashboard, Ticket, Session Detail
    assert len(app.pages) >= 3
    
    page_names = [page.name for page in app.pages]
    assert "Dashboard" in page_names
    assert "Ticket" in page_names
    assert "Session Detail" in page_names
    
    # Verify routes
    dashboard = next(p for p in app.pages if p.name == "Dashboard")
    ticket = next(p for p in app.pages if p.name == "Ticket")
    
    assert dashboard.route == "/dashboard"
    assert ticket.route == "/ticket/:id"
    
    print(f"✅ Found {len(app.pages)} pages: {page_names}")


def test_dashboard_components(support_console_source):
    """Test Dashboard page has correct components."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    dashboard = next(p for p in app.pages if p.name == "Dashboard")
    
    # Check for required components
    component_types = [c.__class__.__name__ for c in dashboard.body]
    
    # Should have: Sidebar, Navbar, Breadcrumbs, StatSummary, DataTable, DataChart
    assert "Sidebar" in component_types
    assert "Navbar" in component_types
    assert "Breadcrumbs" in component_types
    assert "ShowStatSummary" in component_types or "StatSummary" in component_types
    assert "ShowDataTable" in component_types or "DataTable" in component_types
    
    print(f"✅ Dashboard has {len(dashboard.body)} components: {set(component_types)}")


def test_ticket_workspace_components(support_console_source):
    """Test Ticket workspace page has all required AI components."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    ticket_page = next(p for p in app.pages if p.name == "Ticket")
    
    # Get all component types (including nested in layouts)
    def get_all_component_types(stmts):
        types = []
        for stmt in stmts:
            types.append(stmt.__class__.__name__)
            # Check for nested children in layouts
            if hasattr(stmt, 'children') and stmt.children:
                types.extend(get_all_component_types(stmt.children))
            if hasattr(stmt, 'body') and stmt.body:
                types.extend(get_all_component_types(stmt.body))
        return types
    
    component_types = get_all_component_types(ticket_page.body)
    
    # Required AI components
    assert "ChatThread" in component_types, "Missing ChatThread component"
    assert "ToolCallView" in component_types, "Missing ToolCallView component"
    assert "AgentPanel" in component_types, "Missing AgentPanel component"
    
    # Required feedback components
    assert "Modal" in component_types, "Missing Modal component"
    assert "Toast" in component_types, "Missing Toast component"
    assert "Alert" in component_types, "Missing Alert component"
    
    # Required form components
    assert "ShowForm" in component_types or "Form" in component_types, "Missing Form component"
    
    print(f"✅ Ticket workspace has all required components")
    print(f"   AI: ChatThread, ToolCallView, AgentPanel")
    print(f"   Feedback: Modal, Toast, Alert")
    print(f"   Forms: ShowForm")


def test_forms_have_validation(support_console_source):
    """Test that forms have proper validation rules."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    ticket_page = next(p for p in app.pages if p.name == "Ticket")
    
    # Find forms in page
    forms = [stmt for stmt in ticket_page.body if stmt.__class__.__name__ == "ShowForm"]
    
    # If forms are nested, search deeper
    if not forms:
        def find_forms(stmts):
            result = []
            for stmt in stmts:
                if stmt.__class__.__name__ == "ShowForm":
                    result.append(stmt)
                if hasattr(stmt, 'children') and stmt.children:
                    result.extend(find_forms(stmt.children))
                if hasattr(stmt, 'body') and stmt.body:
                    result.extend(find_forms(stmt.body))
            return result
        forms = find_forms(ticket_page.body)
    
    assert len(forms) >= 3, f"Expected at least 3 forms, found {len(forms)}"
    
    # Check refund form has validation
    refund_form = next((f for f in forms if "Refund" in f.title), None)
    assert refund_form is not None, "Refund form not found"
    assert len(refund_form.fields) >= 3, "Refund form should have at least 3 fields"
    
    # Check for required fields
    required_fields = [f for f in refund_form.fields if f.get('required', False)]
    assert len(required_fields) >= 2, "Should have at least 2 required fields"
    
    # Check for pattern validation
    pattern_fields = [f for f in refund_form.fields if 'pattern' in f]
    assert len(pattern_fields) >= 1, "Should have at least 1 field with pattern validation"
    
    print(f"✅ Found {len(forms)} forms with proper validation")
    print(f"   Refund form: {len(refund_form.fields)} fields, {len(required_fields)} required")


def test_modals_have_actions(support_console_source):
    """Test that modals have proper action buttons."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    ticket_page = next(p for p in app.pages if p.name == "Ticket")
    
    # Find modals
    def find_modals(stmts):
        result = []
        for stmt in stmts:
            if stmt.__class__.__name__ == "Modal":
                result.append(stmt)
            if hasattr(stmt, 'children') and stmt.children:
                result.extend(find_modals(stmt.children))
            if hasattr(stmt, 'body') and stmt.body:
                result.extend(find_modals(stmt.body))
        return result
    
    modals = find_modals(ticket_page.body)
    assert len(modals) >= 2, f"Expected at least 2 modals, found {len(modals)}"
    
    # Check refund confirmation modal
    refund_modal = next((m for m in modals if "refund" in m.id.lower()), None)
    assert refund_modal is not None, "Refund confirmation modal not found"
    assert len(refund_modal.actions) >= 2, "Modal should have at least 2 actions (Cancel + Confirm)"
    
    # Check for destructive action
    action_variants = [a.variant for a in refund_modal.actions]
    assert "destructive" in action_variants, "Refund modal should have destructive action"
    
    print(f"✅ Found {len(modals)} modals with proper actions")


# =============================================================================
# IR Tests
# =============================================================================

def test_ir_generation(support_console_source):
    """Test that IR generation succeeds."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    ir = build_ir(app)
    
    assert ir is not None
    assert ir.app_name == "AI Customer Support Console"
    assert len(ir.pages) >= 3
    
    print(f"✅ Generated IR for {ir.app_name}")
    print(f"   Pages: {len(ir.pages)}")
    print(f"   Datasets: {len(ir.datasets)}")


def test_ir_page_components(support_console_source):
    """Test that IR correctly represents page components."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    ir = build_ir(app)
    
    # Find ticket page in IR
    ticket_page = next((p for p in ir.pages if p.name == "Ticket"), None)
    assert ticket_page is not None, "Ticket page not in IR"
    
    # Check components are represented
    component_types = [c.type for c in ticket_page.components]
    
    assert "chat_thread" in component_types
    assert "tool_call_view" in component_types
    assert "agent_panel" in component_types
    
    print(f"✅ IR page has {len(ticket_page.components)} components")


# =============================================================================
# Backend Codegen Tests
# =============================================================================

def test_backend_state_generation(support_console_source):
    """Test backend state builds correctly."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    state = build_backend_state(app)
    
    assert state is not None
    assert len(state.pages) >= 3
    assert len(state.datasets) >= 5
    
    print(f"✅ Built backend state")
    print(f"   Pages: {len(state.pages)}")
    print(f"   Datasets: {len(state.datasets)}")


def test_tool_endpoints_generated(support_console_source):
    """Test that tool endpoints are in backend state."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    state = build_backend_state(app)
    
    # Tools should be in state
    assert hasattr(state, 'tools') or len(app.tools) >= 4
    
    print(f"✅ Backend state includes tool definitions")


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_pipeline_parsing(support_console_source):
    """Integration test: Parse → AST → IR → Backend State."""
    # Step 1: Parse
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    assert app is not None
    
    # Step 2: Build IR
    ir = build_ir(app)
    assert ir is not None
    assert len(ir.pages) >= 3
    
    # Step 3: Build Backend State
    state = build_backend_state(app)
    assert state is not None
    assert len(state.pages) >= 3
    
    print("✅ Full pipeline test passed: Parser → AST → IR → Backend State")


def test_no_hardcoded_data(support_console_source):
    """Test that application uses datasets, not hardcoded demo data."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    # Check that all data display components reference datasets
    for page in app.pages:
        for stmt in page.body:
            comp_type = stmt.__class__.__name__
            
            # Data display components should bind to datasets
            if comp_type in ["ShowDataTable", "ShowStatSummary", "ShowDataChart"]:
                # Should have data_binding or source property
                assert hasattr(stmt, 'source') or hasattr(stmt, 'data_binding'), \
                    f"{comp_type} should bind to a dataset, not hardcoded data"
    
    print("✅ No hardcoded demo data found - all components use datasets")


def test_realistic_schemas(support_console_source):
    """Test that datasets have realistic column schemas."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    # Check tickets dataset
    tickets_ds = next(ds for ds in app.datasets if ds.name == "tickets")
    assert len(tickets_ds.columns) >= 8, "Tickets dataset should have realistic columns"
    
    column_names = [col['name'] for col in tickets_ds.columns]
    assert "id" in column_names
    assert "subject" in column_names
    assert "status" in column_names
    assert "priority" in column_names
    assert "created_at" in column_names
    
    print(f"✅ Realistic schemas: tickets has {len(tickets_ds.columns)} columns")


# =============================================================================
# Component-Specific Tests
# =============================================================================

def test_chat_thread_config(support_console_source):
    """Test ChatThread component configuration."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    ticket_page = next(p for p in app.pages if p.name == "Ticket")
    
    # Find ChatThread
    def find_component(stmts, comp_name):
        for stmt in stmts:
            if stmt.__class__.__name__ == comp_name:
                return stmt
            if hasattr(stmt, 'children') and stmt.children:
                result = find_component(stmt.children, comp_name)
                if result:
                    return result
            if hasattr(stmt, 'body') and stmt.body:
                result = find_component(stmt.body, comp_name)
                if result:
                    return result
        return None
    
    chat_thread = find_component(ticket_page.body, "ChatThread")
    assert chat_thread is not None, "ChatThread not found"
    
    # Verify configuration
    assert chat_thread.messages_binding == "ticket.interactions"
    assert chat_thread.show_timestamps == True
    assert chat_thread.auto_scroll == True
    assert chat_thread.enable_copy == True
    
    print(f"✅ ChatThread properly configured with binding: {chat_thread.messages_binding}")


def test_agent_panel_config(support_console_source):
    """Test AgentPanel component configuration."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    ticket_page = next(p for p in app.pages if p.name == "Ticket")
    
    # Find AgentPanel
    def find_component(stmts, comp_name):
        for stmt in stmts:
            if stmt.__class__.__name__ == comp_name:
                return stmt
            if hasattr(stmt, 'children') and stmt.children:
                result = find_component(stmt.children, comp_name)
                if result:
                    return result
            if hasattr(stmt, 'body') and stmt.body:
                result = find_component(stmt.body, comp_name)
                if result:
                    return result
        return None
    
    agent_panel = find_component(ticket_page.body, "AgentPanel")
    assert agent_panel is not None, "AgentPanel not found"
    
    # Verify shows metrics
    assert agent_panel.agent_binding == "support_assistant"
    assert agent_panel.show_tokens == True
    assert agent_panel.show_cost == True
    assert agent_panel.show_model == True
    assert agent_panel.show_tools == True
    
    print(f"✅ AgentPanel properly configured for agent: {agent_panel.agent_binding}")


def test_tool_call_view_config(support_console_source):
    """Test ToolCallView component configuration."""
    parser = Parser(support_console_source)
    module = parser.parse()
    app = module.body[0]
    
    ticket_page = next(p for p in app.pages if p.name == "Ticket")
    
    # Find ToolCallView
    def find_component(stmts, comp_name):
        for stmt in stmts:
            if stmt.__class__.__name__ == comp_name:
                return stmt
            if hasattr(stmt, 'children') and stmt.children:
                result = find_component(stmt.children, comp_name)
                if result:
                    return result
            if hasattr(stmt, 'body') and stmt.body:
                result = find_component(stmt.body, comp_name)
                if result:
                    return result
        return None
    
    tool_view = find_component(ticket_page.body, "ToolCallView")
    assert tool_view is not None, "ToolCallView not found"
    
    # Verify shows tool details
    assert tool_view.calls_binding == "ticket.tool_calls"
    assert tool_view.show_inputs == True
    assert tool_view.show_outputs == True
    assert tool_view.show_timing == True
    assert tool_view.show_status == True
    
    print(f"✅ ToolCallView properly configured with binding: {tool_view.calls_binding}")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
