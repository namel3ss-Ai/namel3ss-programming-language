"""
End-to-end integration tests for design token system.
Tests complete pipeline from DSL to React with all features.
"""

import textwrap
import pytest
import tempfile
from pathlib import Path
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ir.builder import build_backend_ir
from namel3ss.codegen.frontend.react.main import generate_react_vite_site


class TestEndToEndPipeline:
    """Test complete design token pipeline."""
    
    def test_full_pipeline_all_token_types(self):
        """Test DSL → Parser → AST → IR → Codegen → React with all token types."""
        dsl = '''
        app "Medical Platform" (theme=dark, color_scheme=indigo)
        
        page "Dashboard" at "/" (theme=dark, color_scheme=indigo):
          show form "Patient Intake" (variant=outlined, tone=success, size=lg):
            fields:
              name: text
              email: text (size=md, tone=primary)
        
        page "Reports" at "/reports" (theme=light, color_scheme=blue):
          show table from model "Patient" (variant=elevated, density=compact)
        '''
        
        # Step 1: Parse DSL
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        ast = result.body[0]
        
        assert ast.name == "Medical Platform"
        assert len(ast.pages) == 2
        
        # Step 2: Build IR
        ir = build_backend_ir(ast)
        
        assert ir.frontend is not None
        assert len(ir.frontend.pages) == 2
        
        page1_ir = ir.frontend.pages[0]
        assert page1_ir.design_tokens.theme == "dark"
        assert page1_ir.design_tokens.color_scheme == "indigo"
        
        # Step 3: Generate React app
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'medical-platform'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            # Verify structure
            assert (output_dir / 'src').exists()
            assert (output_dir / 'src' / 'lib' / 'designTokens.ts').exists()
            assert (output_dir / 'src' / 'components' / 'FormWidget.tsx').exists()
            assert (output_dir / 'src' / 'components' / 'TableWidget.tsx').exists()
            assert (output_dir / 'src' / 'pages' / 'index.tsx').exists()
            assert (output_dir / 'src' / 'pages' / 'reports.tsx').exists()
            
            # Verify design tokens utility
            tokens_content = (output_dir / 'src' / 'lib' / 'designTokens.ts').read_text()
            assert 'export function mapButtonClasses' in tokens_content
            assert 'export function mapInputClasses' in tokens_content
            assert 'export function useSystemTheme' in tokens_content
            
            # Verify FormWidget integration
            form_content = (output_dir / 'src' / 'components' / 'FormWidget.tsx').read_text()
            assert 'from "../lib/designTokens"' in form_content
            assert 'mapButtonClasses(' in form_content or 'mapInputClasses(' in form_content
            
            # Verify page integration
            page_content = (output_dir / 'src' / 'pages' / 'index.tsx').read_text()
            assert 'getThemeClassName' in page_content
            assert 'colorSchemeStyles' in page_content
    
    def test_inheritance_end_to_end(self):
        """Test token inheritance works through entire pipeline."""
        dsl = '''
        app "Platform" (theme=dark, color_scheme=blue)
        
        page "Home" at "/":
          show form "Login" (variant=outlined, size=md):
            fields:
              username: text
              password: text (size=sm)
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        # IR should show inheritance
        page_ir = ir.frontend.pages[0]
        component_ir = page_ir.components[0]
        
        # Component inherits from page, which inherits from app
        assert component_ir.design_tokens.theme == "dark"
        assert component_ir.design_tokens.color_scheme == "blue"
        assert component_ir.design_tokens.variant == "outlined"
        assert component_ir.design_tokens.size == "md"
        
        # Field override in AST
        form_ast = ast.pages[0].body[0]
        assert form_ast.fields[1].size.value == "sm"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'platform'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            # Verify generated code handles overrides
            form_content = (output_dir / 'src' / 'components' / 'FormWidget.tsx').read_text()
            assert 'field.size' in form_content or 'field?.size' in form_content
    
    def test_theme_switching_end_to_end(self):
        """Test theme switching (light/dark/system) works end-to-end."""
        dsl = '''
        app "App"
        
        page "Light" at "/" (theme=light):
          show form "F1": fields: name: text
        
        page "Dark" at "/dark" (theme=dark):
          show form "F2": fields: email: text
        
        page "System" at "/system" (theme=system):
          show form "F3": fields: phone: text
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            # Check light page
            light_content = (output_dir / 'src' / 'pages' / 'index.tsx').read_text()
            assert '"light"' in light_content
            assert 'getThemeClassName' in light_content
            
            # Check dark page
            dark_content = (output_dir / 'src' / 'pages' / 'dark.tsx').read_text()
            assert '"dark"' in dark_content
            
            # Check system page
            system_content = (output_dir / 'src' / 'pages' / 'system.tsx').read_text()
            assert '"system"' in system_content
            assert 'useSystemTheme' in system_content
    
    def test_color_schemes_end_to_end(self):
        """Test all 8 color schemes work end-to-end."""
        color_schemes = ["blue", "green", "violet", "rose", "orange", "teal", "indigo", "slate"]
        
        dsl_pages = "\n".join([
            f'page "Page{i}" at "/{scheme}" (color_scheme={scheme}):\n  show form "F{i}": fields: name: text'
            for i, scheme in enumerate(color_schemes)
        ])
        
        dsl = f'app "ColorTest"\n{dsl_pages}'
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'colortest'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            # Verify each color scheme is in its page
            for scheme in color_schemes:
                page_file = output_dir / 'src' / 'pages' / f'{scheme}.tsx'
                if page_file.exists():
                    content = page_file.read_text()
                    assert f'"{scheme}"' in content
                    assert 'getColorSchemeStyles' in content


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_medical_dashboard_scenario(self):
        """Test realistic medical dashboard with multiple widgets and tokens."""
        dsl = '''
        app "HealthCare System" (theme=light, color_scheme=teal)
        
        page "Patient Dashboard" at "/" (theme=light):
          show form "Quick Add" (variant=elevated, tone=primary, size=md):
            fields:
              patient_name: text
              age: number (size=sm)
              condition: text
          
          show table from model "Patient" (variant=outlined, density=comfortable)
        
        page "Analytics" at "/analytics" (theme=dark, color_scheme=indigo):
          show form "Report Generator" (variant=subtle, tone=neutral, size=lg):
            fields:
              start_date: date
              end_date: date
              report_type: text
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'healthcare'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            # Verify complete app structure
            assert (output_dir / 'src' / 'lib' / 'designTokens.ts').exists()
            assert (output_dir / 'src' / 'components' / 'FormWidget.tsx').exists()
            assert (output_dir / 'src' / 'components' / 'TableWidget.tsx').exists()
            assert (output_dir / 'src' / 'pages' / 'index.tsx').exists()
            assert (output_dir / 'src' / 'pages' / 'analytics.tsx').exists()
            
            # Verify Dashboard page
            dashboard_content = (output_dir / 'src' / 'pages' / 'index.tsx').read_text()
            assert '"light"' in dashboard_content
            assert '"teal"' in dashboard_content or 'teal' in dashboard_content
            
            # Verify Analytics page
            analytics_content = (output_dir / 'src' / 'pages' / 'analytics.tsx').read_text()
            assert '"dark"' in analytics_content
            assert '"indigo"' in analytics_content or 'indigo' in analytics_content
    
    def test_multi_form_page_with_different_tokens(self):
        """Test page with multiple forms using different tokens."""
        dsl = '''
        app "Forms Demo"
        
        page "Forms" at "/":
          show form "Login" (variant=outlined, tone=primary, size=md):
            fields:
              username: text
              password: text
          
          show form "Register" (variant=elevated, tone=success, size=lg):
            fields:
              email: text
              name: text
          
          show form "Feedback" (variant=subtle, tone=neutral, size=sm):
            fields:
              message: text
              rating: number
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        # Verify IR has all components with correct tokens
        page_ir = ir.frontend.pages[0]
        assert len(page_ir.components) == 3
        
        assert page_ir.components[0].design_tokens.variant == "outlined"
        assert page_ir.components[0].design_tokens.tone == "primary"
        
        assert page_ir.components[1].design_tokens.variant == "elevated"
        assert page_ir.components[1].design_tokens.tone == "success"
        
        assert page_ir.components[2].design_tokens.variant == "subtle"
        assert page_ir.components[2].design_tokens.tone == "neutral"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'formsdemo'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            # Verify page renders all forms
            page_content = (output_dir / 'src' / 'pages' / 'index.tsx').read_text()
            assert 'FormWidget' in page_content
    
    def test_complex_field_overrides(self):
        """Test complex field-level override patterns."""
        dsl = '''
        app "App"
        
        page "Form" at "/":
          show form "Registration" (variant=outlined, tone=neutral, size=md):
            fields:
              name: text
              email: text (tone=primary)
              phone: text (size=sm)
              address: text (size=lg, variant=subtle)
              notes: text (tone=warning, size=xs)
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        # Verify AST has all field overrides
        form_ast = ast.pages[0].body[0]
        assert form_ast.fields[0].tone is None  # inherits
        assert form_ast.fields[1].tone.value == "primary"  # overrides tone
        assert form_ast.fields[2].size.value == "sm"  # overrides size
        assert form_ast.fields[3].size.value == "lg"  # overrides multiple
        assert form_ast.fields[3].variant.value == "subtle"
        assert form_ast.fields[4].tone.value == "warning"  # overrides both
        assert form_ast.fields[4].size.value == "xs"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'app'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            # Verify FormWidget handles overrides
            form_content = (output_dir / 'src' / 'components' / 'FormWidget.tsx').read_text()
            assert 'mapInputClasses(' in form_content


class TestRegressionScenarios:
    """Test scenarios that previously caused issues."""
    
    def test_no_app_level_tokens(self):
        """Test app without any global tokens."""
        dsl = '''
        app "Test"
        
        page "Home" at "/" (theme=dark):
          show form "F": fields: name: text
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            assert (output_dir / 'src' / 'pages' / 'index.tsx').exists()
    
    def test_no_component_tokens(self):
        """Test components without any tokens."""
        dsl = '''
        app "Test" (theme=dark)
        
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            form_content = (output_dir / 'src' / 'components' / 'FormWidget.tsx').read_text()
            assert 'mapButtonClasses(' in form_content or 'mapInputClasses(' in form_content
    
    def test_mixed_token_presence(self):
        """Test some pages with tokens, others without."""
        dsl = '''
        app "Test"
        
        page "With" at "/" (theme=dark, color_scheme=blue):
          show form "F1": fields: name: text
        
        page "Without" at "/without":
          show form "F2": fields: email: text
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            with_content = (output_dir / 'src' / 'pages' / 'index.tsx').read_text()
            without_content = (output_dir / 'src' / 'pages' / 'without.tsx').read_text()
            
            assert '"dark"' in with_content
            assert '"blue"' in with_content
            # without page should still import utilities
            assert 'from "../lib/designTokens"' in without_content


class TestValidationAndErrorHandling:
    """Test validation and error handling in pipeline."""
    
    def test_all_token_values_valid(self):
        """Test that all standard token values work."""
        variants = ["elevated", "outlined", "ghost", "subtle"]
        tones = ["neutral", "primary", "success", "warning", "danger"]
        sizes = ["xs", "sm", "md", "lg", "xl"]
        
        for variant in variants:
            for tone in tones:
                for size in sizes:
                    dsl = f'''
                    app "Test"
                    page "Home" at "/":
                      show form "F" (variant={variant}, tone={tone}, size={size}):
                        fields: name: text
                    '''
                    
                    parser = LegacyProgramParser(textwrap.dedent(dsl))
                    ast = parser.parse().body[0]
                    ir = build_backend_ir(ast)
                    
                    component_ir = ir.frontend.pages[0].components[0]
                    assert component_ir.design_tokens.variant == variant
                    assert component_ir.design_tokens.tone == tone
                    assert component_ir.design_tokens.size == size
    
    def test_all_themes_valid(self):
        """Test all theme values work."""
        themes = ["light", "dark", "system"]
        
        for theme in themes:
            dsl = f'''
            app "Test"
            page "Home" at "/" (theme={theme}):
              show form "F": fields: name: text
            '''
            
            parser = LegacyProgramParser(textwrap.dedent(dsl))
            ast = parser.parse().body[0]
            ir = build_backend_ir(ast)
            
            page_ir = ir.frontend.pages[0]
            assert page_ir.design_tokens.theme == theme
    
    def test_all_color_schemes_valid(self):
        """Test all color scheme values work."""
        color_schemes = ["blue", "green", "violet", "rose", "orange", "teal", "indigo", "slate"]
        
        for color_scheme in color_schemes:
            dsl = f'''
            app "Test"
            page "Home" at "/" (color_scheme={color_scheme}):
              show form "F": fields: name: text
            '''
            
            parser = LegacyProgramParser(textwrap.dedent(dsl))
            ast = parser.parse().body[0]
            ir = build_backend_ir(ast)
            
            page_ir = ir.frontend.pages[0]
            assert page_ir.design_tokens.color_scheme == color_scheme


class TestGeneratedCodeQuality:
    """Test quality of generated code."""
    
    def test_generated_typescript_is_valid(self):
        """Test generated TypeScript has valid syntax."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            tokens_file = output_dir / 'src' / 'lib' / 'designTokens.ts'
            content = tokens_file.read_text()
            
            # Basic TypeScript validation
            assert content.count('export function') >= 8
            assert 'import { useState, useEffect } from' in content
            assert content.count('{') == content.count('}')  # Balanced braces
            assert content.count('(') == content.count(')')  # Balanced parens
    
    def test_generated_components_import_correctly(self):
        """Test generated components have correct imports."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / 'test'
            generate_react_vite_site(ast, str(output_dir), backend_ir=ir)
            
            form_widget = output_dir / 'src' / 'components' / 'FormWidget.tsx'
            content = form_widget.read_text()
            
            # Check imports
            assert 'import' in content
            assert 'from "../lib/designTokens"' in content
            assert 'export default function' in content or 'export function' in content
