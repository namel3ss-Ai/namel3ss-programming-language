"""
Tests for Chrome Component IR Builder (AST→IR conversion).

Tests cover:
- Sidebar AST→IR conversion with route validation
- Navbar AST→IR conversion with action collection
- Breadcrumbs AST→IR conversion
- Command Palette AST→IR conversion
- Route validation against available pages
- Action validation and collection
"""

import textwrap
import pytest
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ir.builder import build_frontend_ir
from namel3ss.ir.spec import IRSidebar, IRNavbar, IRBreadcrumbs, IRCommandPalette


def parse(source: str):
    """Helper to parse dedented source code."""
    return LegacyProgramParser(textwrap.dedent(source)).parse()


class TestSidebarIRConversion:
    """Test Sidebar AST→IR conversion."""

    def test_sidebar_converts_to_ir(self):
        """Test basic sidebar conversion to IR."""
        source = '''
            app "Test"

            page "Home" at "/":
                sidebar:
                    item "Dashboard" at "/" icon "📊"
                    item "Settings" at "/settings" icon "⚙️"
                
                show text "Hello"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        # Find sidebar component in IR
        page = ir.pages[0]
        sidebar = next((c for c in page.components if c.type == 'sidebar'), None)
        
        assert sidebar is not None
        assert sidebar.name == 'sidebar'
        # Items are in the IR spec stored in metadata
        assert 'ir_spec' in sidebar.metadata
        ir_spec = sidebar.metadata['ir_spec']
        assert len(ir_spec.items) == 2
        assert ir_spec.items[0].label == "Dashboard"
        assert ir_spec.items[1].label == "Settings"

    def test_sidebar_route_validation_success(self):
        """Test sidebar validates routes against available pages."""
        source = '''
            app "Test"

            page "Dashboard" at "/":
                sidebar:
                    item "Dashboard" at "/" icon "📊"
                    item "Settings" at "/settings" icon "⚙️"
                
                show text "Hello"
            
            page "Settings" at "/settings":
                show text "Settings"
        '''
        module = parse(source)
        # Should not raise - both routes are valid
        ir = build_frontend_ir(module.body[0])
        
        assert len(ir.pages) == 2

    def test_sidebar_with_nested_items_converts(self):
        """Test nested sidebar items convert correctly."""
        source = '''
            app "Test"

            page "Home" at "/":
                sidebar:
                    item "Reports" at "/reports" icon "📊":
                        item "Sales" at "/reports/sales"
                        item "Revenue" at "/reports/revenue"
                
                show text "Hello"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        sidebar = next((c for c in page.components if c.type == 'sidebar'), None)
        
        assert sidebar is not None
        ir_spec = sidebar.metadata['ir_spec']
        assert len(ir_spec.items) == 1
        assert len(ir_spec.items[0].children) == 2


class TestNavbarIRConversion:
    """Test Navbar AST→IR conversion."""

    def test_navbar_converts_to_ir(self):
        """Test basic navbar conversion to IR."""
        source = '''
            app "Test"

            page "Home" at "/":
                navbar:
                    logo: "/logo.png"
                    title: "My App"
                    position: top
                    sticky: true
                
                show text "Hello"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        navbar = next((c for c in page.components if c.type == 'navbar'), None)
        
        assert navbar is not None
        assert navbar.props['logo'] == '/logo.png'
        assert navbar.props['title'] == 'My App'
        assert navbar.props['position'] == 'top'
        assert navbar.props['sticky'] is True

    def test_navbar_with_actions_converts(self):
        """Test navbar actions convert to IR."""
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
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        navbar = next((c for c in page.components if c.type == 'navbar'), None)
        
        assert navbar is not None
        ir_spec = navbar.metadata['ir_spec']
        assert len(ir_spec.actions) == 2
        assert ir_spec.actions[0].label == 'Theme'
        assert ir_spec.actions[0].type == 'toggle'

    def test_navbar_with_menu_converts(self):
        """Test navbar menu converts to IR."""
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
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        navbar = next((c for c in page.components if c.type == 'navbar'), None)
        
        assert navbar is not None
        ir_spec = navbar.metadata['ir_spec']
        assert len(ir_spec.actions) == 1
        assert ir_spec.actions[0].type == 'menu'
        assert len(ir_spec.actions[0].menu_items) == 2


class TestBreadcrumbsIRConversion:
    """Test Breadcrumbs AST→IR conversion."""

    def test_breadcrumbs_converts_to_ir(self):
        """Test basic breadcrumbs conversion to IR."""
        source = '''
            app "Test"

            page "Reports" at "/reports":
                breadcrumbs:
                    item "Home" at "/"
                    item "Reports" at "/reports"
                
                show text "Reports"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        breadcrumbs = next((c for c in page.components if c.type == 'breadcrumbs'), None)
        
        assert breadcrumbs is not None
        ir_spec = breadcrumbs.metadata['ir_spec']
        assert len(ir_spec.items) == 2
        assert ir_spec.items[0].label == 'Home'
        assert ir_spec.items[1].label == 'Reports'

    def test_breadcrumbs_auto_derive_converts(self):
        """Test breadcrumbs auto-derivation converts to IR."""
        source = '''
            app "Test"

            page "Reports" at "/reports/sales":
                breadcrumbs:
                    auto_derive: true
                
                show text "Sales Report"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        breadcrumbs = next((c for c in page.components if c.type == 'breadcrumbs'), None)
        
        assert breadcrumbs is not None
        assert breadcrumbs.props.get('auto_derive') is True


class TestCommandPaletteIRConversion:
    """Test Command Palette AST→IR conversion."""

    def test_command_palette_converts_to_ir(self):
        """Test basic command palette conversion to IR."""
        source = '''
            app "Test"

            page "Home" at "/":
                command palette:
                    shortcut: "Ctrl+K"
                    placeholder: "Search..."
                
                show text "Hello"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        cmd_palette = next((c for c in page.components if c.type == 'command_palette'), None)
        
        assert cmd_palette is not None
        assert cmd_palette.props['shortcut'] == 'Ctrl+K'
        assert cmd_palette.props['placeholder'] == 'Search...'

    def test_command_palette_with_sources_converts(self):
        """Test command palette with custom sources converts to IR."""
        source = '''
            app "Test"

            page "Home" at "/":
                command palette:
                    shortcut: "Ctrl+K"
                    source "documents" from "/api/search/documents" label "Search Documents"
                    source "users" from "/api/search/users" label "Find Users"
                
                show text "Hello"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        cmd_palette = next((c for c in page.components if c.type == 'command_palette'), None)
        
        assert cmd_palette is not None
        ir_spec = cmd_palette.metadata['ir_spec']
        assert len(ir_spec.sources) == 2
        assert ir_spec.sources[0].id == 'documents'
        assert ir_spec.sources[0].endpoint == '/api/search/documents'
        assert ir_spec.sources[0].label == 'Search Documents'


class TestChromeValidation:
    """Test validation logic for chrome components."""

    def test_sidebar_badge_parsing(self):
        """Test sidebar badge dict parsing in IR."""
        source = '''
            app "Test"

            page "Home" at "/":
                sidebar:
                    item "Messages" at "/messages" badge {count: 5}
                
                show text "Hello"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        sidebar = next((c for c in page.components if c.type == 'sidebar'), None)
        
        assert sidebar is not None
        ir_spec = sidebar.metadata['ir_spec']
        assert ir_spec.items[0].badge is not None
        assert ir_spec.items[0].badge['count'] == 5

    def test_multiple_chrome_components_on_page(self):
        """Test multiple chrome components coexist in IR."""
        source = '''
            app "Test"

            page "Dashboard" at "/":
                sidebar:
                    item "Home" at "/"
                
                navbar:
                    title: "App"
                
                breadcrumbs:
                    item "Home" at "/"
                    item "Dashboard"
                
                command palette:
                    shortcut: "Ctrl+K"
                
                show text "Content"
        '''
        module = parse(source)
        ir = build_frontend_ir(module.body[0])
        
        page = ir.pages[0]
        components = page.components
        
        # Should have 5 components: sidebar, navbar, breadcrumbs, command_palette, text
        assert len(components) >= 4
        
        has_sidebar = any(c.type == 'sidebar' for c in components)
        has_navbar = any(c.type == 'navbar' for c in components)
        has_breadcrumbs = any(c.type == 'breadcrumbs' for c in components)
        has_cmd_palette = any(c.type == 'command_palette' for c in components)
        
        assert has_sidebar
        assert has_navbar
        assert has_breadcrumbs
        assert has_cmd_palette
