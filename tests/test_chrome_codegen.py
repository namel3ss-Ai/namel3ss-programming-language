"""
Tests for chrome component React code generation.

Validates that IR specs are correctly serialized to React component props.
"""

import pytest
from namel3ss.ir.spec import (
    IRSidebar,
    IRNavbar,
    IRBreadcrumbs,
    IRCommandPalette,
    IRNavItem,
    IRNavSection,
    IRNavbarAction,
    IRBreadcrumbItem,
    IRCommandSource,
)
from namel3ss.codegen.frontend.react.pages import (
    serialize_nav_item,
    serialize_nav_section,
    serialize_navbar_action,
    serialize_breadcrumb_item,
    serialize_command_source,
)


class TestSidebarSerialization:
    """Test sidebar IR → React props serialization."""
    
    def test_basic_nav_item_serialization(self):
        """Test nav item converts to correct props."""
        nav_item = IRNavItem(
            id="dashboard",
            label="Dashboard",
            route="/",
            icon="📊"
        )
        
        result = serialize_nav_item(nav_item)
        
        assert result["id"] == "dashboard"
        assert result["label"] == "Dashboard"
        assert result["route"] == "/"
        assert result["icon"] == "📊"
    
    def test_nav_item_with_badge(self):
        """Test nav item with badge serializes correctly."""
        nav_item = IRNavItem(
            id="analytics",
            label="Analytics",
            route="/analytics",
            badge={"text": "New", "variant": "info"}
        )
        
        result = serialize_nav_item(nav_item)
        
        assert result["badge"] == {"text": "New", "variant": "info"}
    
    def test_nav_item_with_children(self):
        """Test nested nav items serialize recursively."""
        child1 = IRNavItem(id="child1", label="Child 1", route="/child1")
        child2 = IRNavItem(id="child2", label="Child 2", route="/child2")
        
        parent = IRNavItem(
            id="parent",
            label="Parent",
            route="/parent",
            children=[child1, child2]
        )
        
        result = serialize_nav_item(parent)
        
        assert len(result["children"]) == 2
        assert result["children"][0]["id"] == "child1"
        assert result["children"][1]["id"] == "child2"
    
    def test_nav_section_serialization(self):
        """Test nav section converts to correct props."""
        section = IRNavSection(
            id="settings",
            label="Settings",
            items=["profile", "security", "preferences"],
            collapsible=True,
            collapsed_by_default=False
        )
        
        result = serialize_nav_section(section)
        
        assert result["id"] == "settings"
        assert result["label"] == "Settings"
        assert result["items"] == ["profile", "security", "preferences"]
        assert result["collapsible"] is True
        assert result["collapsed_by_default"] is False


class TestNavbarSerialization:
    """Test navbar IR → React props serialization."""
    
    def test_basic_navbar_action(self):
        """Test navbar action converts to correct props."""
        action = IRNavbarAction(
            id="theme",
            label="Theme",
            icon="🎨",
            type="toggle"
        )
        
        result = serialize_navbar_action(action)
        
        assert result["id"] == "theme"
        assert result["label"] == "Theme"
        assert result["icon"] == "🎨"
        assert result["type"] == "toggle"
    
    def test_navbar_action_button_type(self):
        """Test navbar button action type."""
        action = IRNavbarAction(
            id="notifications",
            label="Notifications",
            icon="🔔",
            type="button"
        )
        
        result = serialize_navbar_action(action)
        
        assert result["id"] == "notifications"
        assert result["type"] == "button"
        assert result["icon"] == "🔔"
    
    def test_navbar_action_with_menu(self):
        """Test navbar action with menu items."""
        menu_item1 = IRNavItem(id="profile", label="Profile", route="/profile")
        menu_item2 = IRNavItem(id="logout", label="Logout", action="logout")
        
        action = IRNavbarAction(
            id="user",
            label="User",
            icon="👤",
            type="menu",
            menu_items=[menu_item1, menu_item2]
        )
        
        result = serialize_navbar_action(action)
        
        assert result["type"] == "menu"
        assert len(result["menu_items"]) == 2
        assert result["menu_items"][0]["id"] == "profile"
        assert result["menu_items"][1]["action"] == "logout"


class TestBreadcrumbsSerialization:
    """Test breadcrumbs IR → React props serialization."""
    
    def test_basic_breadcrumb_item(self):
        """Test breadcrumb item converts to correct props."""
        item = IRBreadcrumbItem(
            label="Home",
            route="/"
        )
        
        result = serialize_breadcrumb_item(item)
        
        assert result["label"] == "Home"
        assert result["route"] == "/"
    
    def test_breadcrumb_item_without_route(self):
        """Test breadcrumb item with no route (current page)."""
        item = IRBreadcrumbItem(
            label="Current Page"
        )
        
        result = serialize_breadcrumb_item(item)
        
        assert result["label"] == "Current Page"
        assert "route" not in result


class TestCommandPaletteSerialization:
    """Test command palette IR → React props serialization."""
    
    def test_routes_source(self):
        """Test routes source serialization."""
        source = IRCommandSource(
            type="routes"
        )
        
        result = serialize_command_source(source)
        
        assert result["type"] == "routes"
    
    def test_actions_source(self):
        """Test actions source serialization."""
        source = IRCommandSource(
            type="actions",
            filter="user.*"
        )
        
        result = serialize_command_source(source)
        
        assert result["type"] == "actions"
        assert result["filter"] == "user.*"
    
    def test_custom_source(self):
        """Test custom source with items."""
        source = IRCommandSource(
            type="custom",
            custom_items=[
                {"label": "Create Report", "action": "create_report"},
                {"label": "Export Data", "action": "export_data"}
            ]
        )
        
        result = serialize_command_source(source)
        
        assert result["type"] == "custom"
        assert len(result["custom_items"]) == 2
        assert result["custom_items"][0]["label"] == "Create Report"
    
    def test_api_source(self):
        """Test API-backed source serialization."""
        source = IRCommandSource(
            type="api",
            id="documents",
            endpoint="/api/search/documents",
            label="Search Documents"
        )
        
        result = serialize_command_source(source)
        
        assert result["type"] == "api"
        assert result["id"] == "documents"
        assert result["endpoint"] == "/api/search/documents"
        assert result["label"] == "Search Documents"


class TestChromeComponentIntegration:
    """Test complete chrome component serialization."""
    
    def test_sidebar_with_all_features(self):
        """Test sidebar with nested items and sections."""
        child_item = IRNavItem(id="sales", label="Sales Report", route="/reports/sales")
        parent_item = IRNavItem(
            id="reports",
            label="Reports",
            route="/reports",
            icon="📋",
            children=[child_item]
        )
        section = IRNavSection(
            id="settings",
            label="Settings",
            items=["profile", "security"],
            collapsible=True,
            collapsed_by_default=False
        )
        
        sidebar = IRSidebar(
            items=[parent_item],
            sections=[section],
            collapsible=True,
            width="normal",
            position="left",
            validated_routes=["/", "/reports", "/reports/sales"]
        )
        
        # Serialize items
        serialized_items = [serialize_nav_item(item) for item in sidebar.items]
        assert len(serialized_items) == 1
        assert len(serialized_items[0]["children"]) == 1
        
        # Serialize sections
        serialized_sections = [serialize_nav_section(sec) for sec in sidebar.sections]
        assert len(serialized_sections) == 1
        assert serialized_sections[0]["collapsible"] is True
    
    def test_navbar_with_multiple_actions(self):
        """Test navbar with various action types."""
        action1 = IRNavbarAction(id="theme", type="toggle", icon="🎨")
        action2 = IRNavbarAction(
            id="user",
            type="menu",
            icon="👤",
            menu_items=[
                IRNavItem(id="profile", label="Profile", route="/profile")
            ]
        )
        
        navbar = IRNavbar(
            logo="/assets/logo.png",
            title="My App",
            actions=[action1, action2],
            position="top",
            sticky=True,
            validated_actions=["theme", "logout"]
        )
        
        serialized_actions = [serialize_navbar_action(action) for action in navbar.actions]
        assert len(serialized_actions) == 2
        assert serialized_actions[0]["type"] == "toggle"
        assert serialized_actions[1]["type"] == "menu"
    
    def test_breadcrumbs_with_auto_derive(self):
        """Test breadcrumbs with auto-derivation enabled."""
        breadcrumbs = IRBreadcrumbs(
            items=[],
            auto_derive=True,
            separator="/"
        )
        
        assert breadcrumbs.auto_derive is True
        assert breadcrumbs.separator == "/"
    
    def test_command_palette_with_multiple_sources(self):
        """Test command palette with various source types."""
        source1 = IRCommandSource(type="routes")
        source2 = IRCommandSource(type="actions", filter=".*")
        source3 = IRCommandSource(
            type="api",
            id="users",
            endpoint="/api/search/users",
            label="Find Users"
        )
        
        palette = IRCommandPalette(
            shortcut="Ctrl+K",
            sources=[source1, source2, source3],
            placeholder="Search...",
            max_results=10
        )
        
        serialized_sources = [serialize_command_source(src) for src in palette.sources]
        assert len(serialized_sources) == 3
        assert serialized_sources[0]["type"] == "routes"
        assert serialized_sources[1]["type"] == "actions"
        assert serialized_sources[2]["type"] == "api"
        assert serialized_sources[2]["endpoint"] == "/api/search/users"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
