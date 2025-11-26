"""
Tests for Chrome Component Parsing (Sidebar, Navbar, Breadcrumbs, CommandPalette).

Tests cover:
- Basic chrome component syntax parsing
- Nested navigation structures
- Sections and grouping
- Icons, badges, and actions
- Auto-derivation and command sources
"""

import textwrap
import pytest
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ast.pages import (
    Sidebar,
    Navbar,
    Breadcrumbs,
    CommandPalette,
    NavItem,
    NavSection,
    NavbarAction,
    BreadcrumbItem,
    CommandSource,
)


def parse(source: str):
    """Helper to parse dedented source code."""
    return LegacyProgramParser(textwrap.dedent(source)).parse()


class TestSidebarParsing:
    """Test parsing of sidebar navigation components."""

    def test_basic_sidebar(self):
        """Test basic sidebar with simple items."""
        source = '''
            app "Test"

            page "Home" at "/":
                sidebar:
                    item "Dashboard" at "/" icon "📊"
                    item "Analytics" at "/analytics" icon "📈"
                    width: normal
                    position: left
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        sidebar = next((s for s in page.body if isinstance(s, Sidebar)), None)
        assert sidebar is not None, "Sidebar should be parsed"
        assert len(sidebar.items) == 2
        assert sidebar.items[0].label == "Dashboard"
        assert sidebar.items[0].route == "/"
        assert sidebar.items[0].icon == "📊"
        assert sidebar.width == "normal"
        assert sidebar.position == "left"

    def test_sidebar_with_nested_items(self):
        """Test sidebar with nested navigation items."""
        source = '''
            app "Test"

            page "Home" at "/":
                sidebar:
                    item "Reports" at "/reports" icon "📄":
                        item "Sales" at "/reports/sales"
                        item "Revenue" at "/reports/revenue"
                    width: normal
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        sidebar = next((s for s in page.body if isinstance(s, Sidebar)), None)
        assert sidebar is not None
        assert len(sidebar.items) == 1
        assert sidebar.items[0].label == "Reports"
        assert len(sidebar.items[0].children) == 2
        assert sidebar.items[0].children[0].label == "Sales"
        assert sidebar.items[0].children[1].label == "Revenue"

    def test_sidebar_with_sections(self):
        """Test sidebar with grouped sections."""
        source = '''
            app "Test"

            page "Home" at "/":
                sidebar:
                    item "Dashboard" at "/" icon "📊"
                    
                    section "Settings":
                        item "Profile" at "/settings/profile"
                        item "Security" at "/settings/security"
                        collapsible: true
                        collapsed by default: false
                    
                    collapsible: true
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        sidebar = next((s for s in page.body if isinstance(s, Sidebar)), None)
        assert sidebar is not None
        assert len(sidebar.items) == 1
        assert len(sidebar.sections) == 1
        assert sidebar.sections[0].label == "Settings"
        assert sidebar.sections[0].collapsible is True
        assert sidebar.sections[0].collapsed_by_default is False
        assert sidebar.collapsible is True

    def test_sidebar_with_badge(self):
        """Test sidebar items with badges."""
        source = '''
            app "Test"

            page "Home" at "/":
                sidebar:
                    item "Analytics" at "/analytics" icon "📈" badge {text: "New"}
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        sidebar = next((s for s in page.body if isinstance(s, Sidebar)), None)
        assert sidebar is not None
        assert len(sidebar.items) == 1
        assert sidebar.items[0].badge is not None
        assert sidebar.items[0].badge['text'] == "New"


class TestNavbarParsing:
    """Test parsing of navbar/topbar components."""

    def test_basic_navbar(self):
        """Test basic navbar with branding."""
        source = '''
            app "Test"

            page "Home" at "/":
                navbar:
                    logo: "/assets/logo.png"
                    title: "My App"
                    position: top
                    sticky: true
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        navbar = next((s for s in page.body if isinstance(s, Navbar)), None)
        assert navbar is not None
        assert navbar.logo == "/assets/logo.png"
        assert navbar.title == "My App"
        assert navbar.position == "top"
        assert navbar.sticky is True

    def test_navbar_with_actions(self):
        """Test navbar with action buttons."""
        source = '''
            app "Test"

            page "Home" at "/":
                navbar:
                    title: "App"
                    action "Theme" icon "🎨" type "toggle"
                    action "Settings" icon "⚙️" type "button"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        navbar = next((s for s in page.body if isinstance(s, Navbar)), None)
        assert navbar is not None
        assert len(navbar.actions) == 2
        assert navbar.actions[0].label == "Theme"
        assert navbar.actions[0].icon == "🎨"
        assert navbar.actions[0].type == "toggle"
        assert navbar.actions[1].type == "button"

    def test_navbar_with_menu(self):
        """Test navbar with dropdown menu."""
        source = '''
            app "Test"

            page "Home" at "/":
                navbar:
                    title: "App"
                    action "User" icon "👤" type "menu":
                        item "Profile" at "/profile"
                        item "Logout" action "logout"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        navbar = next((s for s in page.body if isinstance(s, Navbar)), None)
        assert navbar is not None
        assert len(navbar.actions) == 1
        assert navbar.actions[0].type == "menu"
        assert len(navbar.actions[0].menu_items) == 2
        assert navbar.actions[0].menu_items[0].label == "Profile"
        assert navbar.actions[0].menu_items[1].action == "logout"


class TestBreadcrumbsParsing:
    """Test parsing of breadcrumbs navigation."""

    def test_basic_breadcrumbs(self):
        """Test basic breadcrumbs with explicit items."""
        source = '''
            app "Test"

            page "Home" at "/":
                breadcrumbs:
                    item "Home" at "/"
                    item "Reports" at "/reports"
                    item "Sales"
                    separator: "/"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        breadcrumbs = next((s for s in page.body if isinstance(s, Breadcrumbs)), None)
        assert breadcrumbs is not None
        assert len(breadcrumbs.items) == 3
        assert breadcrumbs.items[0].label == "Home"
        assert breadcrumbs.items[0].route == "/"
        assert breadcrumbs.items[2].label == "Sales"
        assert breadcrumbs.items[2].route is None
        assert breadcrumbs.separator == "/"

    def test_breadcrumbs_auto_derive(self):
        """Test breadcrumbs with auto-derivation from route."""
        source = '''
            app "Test"

            page "Home" at "/":
                breadcrumbs:
                    auto derive: true
                    separator: ">"
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        breadcrumbs = next((s for s in page.body if isinstance(s, Breadcrumbs)), None)
        assert breadcrumbs is not None
        assert breadcrumbs.auto_derive is True
        assert breadcrumbs.separator == ">"


class TestCommandPaletteParsing:
    """Test parsing of command palette component."""

    def test_basic_command_palette(self):
        """Test basic command palette."""
        source = '''
            app "Test"

            page "Home" at "/":
                command palette:
                    shortcut: "ctrl+k"
                    source routes
                    source actions
                    placeholder: "Search..."
                    max results: 10
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        cmd_palette = next((s for s in page.body if isinstance(s, CommandPalette)), None)
        assert cmd_palette is not None
        assert cmd_palette.shortcut == "ctrl+k"
        assert len(cmd_palette.sources) == 2
        assert cmd_palette.sources[0].type == "routes"
        assert cmd_palette.sources[1].type == "actions"
        assert cmd_palette.placeholder == "Search..."
        assert cmd_palette.max_results == 10

    def test_command_palette_with_custom_sources(self):
        """Test command palette with custom data sources."""
        source = '''
            app "Test"

            page "Home" at "/":
                command palette:
                    shortcut: "ctrl+k"
                    source custom:
                        item "Export Data" action "export"
                        item "Import Data" action "import"
                    placeholder: "Type a command..."
                
                show text "Hello"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        cmd_palette = next((s for s in page.body if isinstance(s, CommandPalette)), None)
        assert cmd_palette is not None
        assert len(cmd_palette.sources) == 1
        assert cmd_palette.sources[0].type == "custom"
        assert len(cmd_palette.sources[0].custom_items) == 2
        assert cmd_palette.sources[0].custom_items[0]['label'] == "Export Data"
        assert cmd_palette.sources[0].custom_items[0]['action'] == "export"


class TestChromeIntegration:
    """Test multiple chrome components together."""

    def test_all_chrome_components_on_page(self):
        """Test page with all chrome components integrated."""
        source = '''
            app "Test"

            page "Dashboard" at "/":
                sidebar:
                    item "Home" at "/" icon "🏠"
                
                navbar:
                    title: "Test App"
                    action "User" icon "👤" type "button"
                
                breadcrumbs:
                    item "Home" at "/"
                    item "Dashboard"
                
                command palette:
                    shortcut: "ctrl+k"
                    source routes
                
                show text "Content"
        '''
        module = parse(source)
        app = module.body[0]
        page = app.pages[0]
        
        sidebar = next((s for s in page.body if isinstance(s, Sidebar)), None)
        navbar = next((s for s in page.body if isinstance(s, Navbar)), None)
        breadcrumbs = next((s for s in page.body if isinstance(s, Breadcrumbs)), None)
        cmd_palette = next((s for s in page.body if isinstance(s, CommandPalette)), None)
        
        assert sidebar is not None, "Sidebar should be parsed"
        assert navbar is not None, "Navbar should be parsed"
        assert breadcrumbs is not None, "Breadcrumbs should be parsed"
        assert cmd_palette is not None, "Command palette should be parsed"
        
        assert len(sidebar.items) == 1
        assert len(navbar.actions) == 1
        assert len(breadcrumbs.items) == 2
        assert len(cmd_palette.sources) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
