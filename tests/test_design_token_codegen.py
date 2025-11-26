"""
Tests for React codegen with design tokens.
Tests TypeScript utility generation and widget integration.
"""

import textwrap
import pytest
import tempfile
from pathlib import Path
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ir.builder import build_backend_ir
from namel3ss.codegen.frontend.react.main import generate_react_vite_site


class TestTypeScriptUtilityGeneration:
    """Test generation of designTokens.ts utility file."""
    
    def test_utility_file_created(self):
        """Test that designTokens.ts file is created."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            tokens_file = output_dir / 'src' / 'lib' / 'designTokens.ts'
            assert tokens_file.exists()
    
    def test_all_mapping_functions_present(self):
        """Test all 5 mapping functions are generated."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            tokens_file = output_dir / 'src' / 'lib' / 'designTokens.ts'
            content = tokens_file.read_text()
            
            assert 'export function mapButtonClasses' in content
            assert 'export function mapInputClasses' in content
            assert 'export function mapTableClasses' in content
            assert 'export function mapCardClasses' in content
            assert 'export function mapBadgeClasses' in content
            assert 'export function mapAlertClasses' in content
            assert 'export function mapDensityClasses' in content
    
    def test_theme_functions_present(self):
        """Test theme utility functions are generated."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            tokens_file = output_dir / 'src' / 'lib' / 'designTokens.ts'
            content = tokens_file.read_text()
            
            assert 'export function getThemeClassName' in content
            assert 'export function useSystemTheme' in content
            assert 'export function getColorSchemeStyles' in content
    
    def test_react_hooks_imported(self):
        """Test React hooks are imported for useSystemTheme."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            tokens_file = output_dir / 'src' / 'lib' / 'designTokens.ts'
            content = tokens_file.read_text()
            
            assert "import { useState, useEffect } from 'react'" in content
    
    def test_type_definitions_included(self):
        """Test TypeScript type definitions are included."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            tokens_file = output_dir / 'src' / 'lib' / 'designTokens.ts'
            content = tokens_file.read_text()
            
            # Type definitions for all token types
            assert 'VariantType' in content
            assert 'ToneType' in content
            assert 'SizeType' in content
            assert 'ThemeType' in content
            assert 'ColorSchemeType' in content


class TestFormWidgetIntegration:
    """Test FormWidget component integration with design tokens."""
    
    def test_form_widget_imports_design_tokens(self):
        """Test FormWidget imports design token utilities."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (variant=outlined):
            fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            form_widget = output_dir / 'src' / 'components' / 'FormWidget.tsx'
            content = form_widget.read_text()
            
            assert 'from "../lib/designTokens"' in content
    
    def test_form_widget_uses_map_button_and_input_classes(self):
        """Test FormWidget uses mapButtonClasses and mapInputClasses functions."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (variant=outlined, tone=primary):
            fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            form_widget = output_dir / 'src' / 'components' / 'FormWidget.tsx'
            content = form_widget.read_text()
            
            # Forms use button classes for submit buttons and input classes for fields
            assert 'mapButtonClasses(' in content or 'mapInputClasses(' in content
    
    def test_form_widget_uses_map_button_classes(self):
        """Test FormWidget uses mapButtonClasses for submit button."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login": fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            form_widget = output_dir / 'src' / 'components' / 'FormWidget.tsx'
            content = form_widget.read_text()
            
            assert 'mapButtonClasses(' in content
    
    def test_form_widget_uses_map_input_classes(self):
        """Test FormWidget uses mapInputClasses for input fields."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login": fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            form_widget = output_dir / 'src' / 'components' / 'FormWidget.tsx'
            content = form_widget.read_text()
            
            assert 'mapInputClasses(' in content
    
    def test_form_widget_field_level_overrides(self):
        """Test FormWidget handles field-level token overrides."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (size=md):
            fields:
              username: text
              password: text (size=sm)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            form_widget = output_dir / 'src' / 'components' / 'FormWidget.tsx'
            content = form_widget.read_text()
            
            # Should handle field.size || widget.size pattern
            assert 'field.size' in content or 'field?.size' in content


class TestTableWidgetIntegration:
    """Test TableWidget component integration with design tokens."""
    
    def test_table_widget_imports_design_tokens(self):
        """Test TableWidget imports design token utilities."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show table from model "User"
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            table_widget = output_dir / 'src' / 'components' / 'TableWidget.tsx'
            content = table_widget.read_text()
            
            assert 'from "../lib/designTokens"' in content
    
    def test_table_widget_uses_map_table_classes(self):
        """Test TableWidget uses mapTableClasses function."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show table from model "User" (variant=elevated, density=compact)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            table_widget = output_dir / 'src' / 'components' / 'TableWidget.tsx'
            content = table_widget.read_text()
            
            assert 'mapTableClasses(' in content
    
    def test_table_widget_handles_density(self):
        """Test TableWidget handles density token."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show table from model "User" (density=comfortable)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            table_widget = output_dir / 'src' / 'components' / 'TableWidget.tsx'
            content = table_widget.read_text()
            
            # Should pass density to mapTableClasses
            assert 'widget.density' in content or '"comfortable"' in content


class TestPageComponentIntegration:
    """Test page component integration with design tokens."""
    
    def test_page_imports_theme_utilities(self):
        """Test page component imports theme utilities."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            content = page_file.read_text()
            
            assert 'getThemeClassName' in content
            assert 'useSystemTheme' in content
    
    def test_page_imports_color_scheme_utilities(self):
        """Test page component imports color scheme utilities."""
        dsl = '''
        app "Test"
        page "Home" at "/" (color_scheme=indigo):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            content = page_file.read_text()
            
            assert 'getColorSchemeStyles' in content
    
    def test_page_imports_type_definitions(self):
        """Test page component imports type definitions."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark, color_scheme=blue):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            content = page_file.read_text()
            
            assert 'ThemeType' in content
            assert 'ColorSchemeType' in content
    
    def test_page_applies_theme_class(self):
        """Test page component applies theme class to container."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            content = page_file.read_text()
            
            assert 'className={themeClass}' in content
    
    def test_page_applies_color_scheme_styles(self):
        """Test page component applies color scheme styles."""
        dsl = '''
        app "Test"
        page "Home" at "/" (color_scheme=violet):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            content = page_file.read_text()
            
            assert 'colorSchemeStyles' in content
            assert 'style={{' in content or 'style=' in content
    
    def test_page_uses_system_theme_hook(self):
        """Test page uses useSystemTheme for system theme."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=system):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            content = page_file.read_text()
            
            assert 'useSystemTheme' in content
            assert 'theme === ' in content or 'theme ===' in content
    
    def test_page_definition_includes_theme(self):
        """Test PAGE_DEFINITION includes theme value."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            content = page_file.read_text()
            
            assert 'PAGE_DEFINITION' in content
            assert '"theme"' in content
            assert '"dark"' in content
    
    def test_page_definition_includes_color_scheme(self):
        """Test PAGE_DEFINITION includes colorScheme value."""
        dsl = '''
        app "Test"
        page "Home" at "/" (color_scheme=teal):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            content = page_file.read_text()
            
            assert '"colorScheme"' in content
            assert '"teal"' in content


class TestMultiplePageGeneration:
    """Test generation with multiple pages having different tokens."""
    
    def test_multiple_pages_with_different_themes(self):
        """Test generating multiple pages with different themes."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark):
          show form "F1": fields: name: text
        page "About" at "/about" (theme=light):
          show form "F2": fields: email: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            home_page = output_dir / 'src' / 'pages' / 'index.tsx'
            about_page = output_dir / 'src' / 'pages' / 'about.tsx'
            
            assert home_page.exists()
            assert about_page.exists()
            
            home_content = home_page.read_text()
            about_content = about_page.read_text()
            
            assert '"dark"' in home_content
            assert '"light"' in about_content
    
    def test_multiple_pages_with_different_color_schemes(self):
        """Test generating multiple pages with different color schemes."""
        dsl = '''
        app "Test"
        page "Dashboard" at "/" (color_scheme=indigo):
          show form "F1": fields: name: text
        page "Analytics" at "/analytics" (color_scheme=violet):
          show form "F2": fields: email: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            dashboard_page = output_dir / 'src' / 'pages' / 'index.tsx'
            analytics_page = output_dir / 'src' / 'pages' / 'analytics.tsx'
            
            dashboard_content = dashboard_page.read_text()
            analytics_content = analytics_page.read_text()
            
            assert '"indigo"' in dashboard_content
            assert '"violet"' in analytics_content


class TestCodegenEdgeCases:
    """Test edge cases in codegen."""
    
    def test_page_without_tokens(self):
        """Test generating page without any design tokens."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            page_file = output_dir / 'src' / 'pages' / 'index.tsx'
            assert page_file.exists()
            
            content = page_file.read_text()
            # Should still import utilities
            assert 'from "../lib/designTokens"' in content
    
    def test_widget_without_tokens(self):
        """Test generating widget without design tokens."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login": fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test-app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            form_widget = output_dir / 'src' / 'components' / 'FormWidget.tsx'
            content = form_widget.read_text()
            
            # Should use mapping functions with default/None values
            assert 'mapButtonClasses(' in content or 'mapInputClasses(' in content
