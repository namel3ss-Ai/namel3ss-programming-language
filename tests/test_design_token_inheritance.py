"""
Tests for design token inheritance in IR builder.
Tests app→page→component→field cascading with overrides.
"""

import pytest
import textwrap
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ir.builder import build_backend_ir
from namel3ss.ast.design_tokens import (
    VariantType,
    ToneType,
    SizeType,
    DensityType,
    ThemeType,
    ColorSchemeType,
)


class TestPageInheritanceFromApp:
    """Test design token inheritance from app to page level."""
    
    def test_page_inherits_app_theme(self):
        """Test page inherits theme from app when not specified."""
        dsl = '''
        app "Test" (theme=dark)
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        # Page should inherit app theme
        page_ir = ir.frontend.pages[0]
        assert page_ir.design_tokens.theme == "dark"
    
    def test_page_overrides_app_theme(self):
        """Test page overrides app theme when specified."""
        dsl = '''
        app "Test" (theme=dark)
        page "Home" at "/" (theme=light):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        # Page overrides to light
        page_ir = ir.frontend.pages[0]
        assert page_ir.design_tokens.theme == "light"
    
    def test_page_inherits_app_color_scheme(self):
        """Test page inherits color scheme from app."""
        dsl = '''
        app "Test" (color_scheme=indigo)
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        assert page_ir.design_tokens.color_scheme == "indigo"
    
    def test_page_overrides_app_color_scheme(self):
        """Test page overrides app color scheme."""
        dsl = '''
        app "Test" (color_scheme=blue)
        page "Home" at "/" (color_scheme=violet):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        assert page_ir.design_tokens.color_scheme == "violet"


class TestComponentInheritanceFromPage:
    """Test design token inheritance from page to component level."""
    
    def test_component_inherits_page_theme(self):
        """Test component inherits theme from page."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        # Component inherits page theme
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.theme == "dark"
    
    def test_component_with_no_page_theme(self):
        """Test component when page has no theme."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (variant=outlined): fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.theme is None
        assert component_ir.design_tokens.variant == "outlined"


class TestComponentLevelTokens:
    """Test component-level token specification and inheritance."""
    
    def test_component_variant_specified(self):
        """Test component with variant specified."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (variant=elevated): fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.variant == "elevated"
    
    def test_component_tone_specified(self):
        """Test component with tone specified."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (tone=primary): fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.tone == "primary"
    
    def test_component_size_specified(self):
        """Test component with size specified."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (size=lg): fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.size == "lg"
    
    def test_component_all_tokens(self):
        """Test component with all token types."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark, color_scheme=blue):
          show form "F" (variant=outlined, tone=success, size=md):
            fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.theme == "dark"
        assert component_ir.design_tokens.color_scheme == "blue"
        assert component_ir.design_tokens.variant == "outlined"
        assert component_ir.design_tokens.tone == "success"
        assert component_ir.design_tokens.size == "md"


class TestFieldInheritanceFromComponent:
    """Test design token inheritance from component to field level."""
    
    def test_field_inherits_component_variant(self):
        """Test field inherits variant from component."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (variant=outlined):
            fields:
              username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        # Field should inherit variant=outlined from form
        assert component_ir.design_tokens.variant == "outlined"
    
    def test_field_inherits_component_tone(self):
        """Test field inherits tone from component."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (tone=primary):
            fields:
              username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.tone == "primary"
    
    def test_field_inherits_component_size(self):
        """Test field inherits size from component."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (size=lg):
            fields:
              username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.size == "lg"


class TestFieldLevelOverrides:
    """Test field-level token overrides."""
    
    def test_field_overrides_size(self):
        """Test field overrides component size."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (size=md):
            fields:
              username: text
              password: text (size=sm)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        # Component has size=md
        assert component_ir.design_tokens.size == "md"
        
        # Field overrides are stored in field nodes
        form_ast = ast.pages[0].body[0]
        assert form_ast.fields[0].size is None  # inherits
        assert form_ast.fields[1].size == SizeType.SM  # overrides
    
    def test_field_overrides_tone(self):
        """Test field overrides component tone."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (tone=neutral):
            fields:
              username: text
              email: text (tone=primary)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.tone == "neutral"
        
        form_ast = ast.pages[0].body[0]
        assert form_ast.fields[0].tone is None
        assert form_ast.fields[1].tone == ToneType.PRIMARY
    
    def test_field_overrides_variant(self):
        """Test field overrides component variant."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (variant=outlined):
            fields:
              username: text
              password: text (variant=ghost)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.variant == "outlined"
        
        form_ast = ast.pages[0].body[0]
        assert form_ast.fields[0].variant is None
        assert form_ast.fields[1].variant == VariantType.GHOST
    
    def test_field_multiple_overrides(self):
        """Test field overrides multiple tokens."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (variant=outlined, tone=neutral, size=md):
            fields:
              username: text
              email: text (size=sm, tone=primary)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        component_ir = ir.frontend.pages[0].components[0]
        assert component_ir.design_tokens.variant == "outlined"
        assert component_ir.design_tokens.tone == "neutral"
        assert component_ir.design_tokens.size == "md"
        
        form_ast = ast.pages[0].body[0]
        # email field overrides size and tone
        assert form_ast.fields[1].size == SizeType.SM
        assert form_ast.fields[1].tone == ToneType.PRIMARY
        assert form_ast.fields[1].variant is None  # doesn't override


class TestFullInheritanceChain:
    """Test complete inheritance chain: app→page→component→field."""
    
    def test_four_level_inheritance_no_overrides(self):
        """Test all levels inherit when no overrides."""
        dsl = '''
        app "Test" (theme=dark, color_scheme=blue)
        page "Home" at "/":
          show form "F":
            fields:
              username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        component_ir = page_ir.components[0]
        
        # Page inherits from app
        assert page_ir.design_tokens.theme == "dark"
        assert page_ir.design_tokens.color_scheme == "blue"
        
        # Component inherits from page
        assert component_ir.design_tokens.theme == "dark"
        assert component_ir.design_tokens.color_scheme == "blue"
    
    def test_four_level_inheritance_page_override(self):
        """Test page overrides app, component inherits."""
        dsl = '''
        app "Test" (theme=dark)
        page "Home" at "/" (theme=light):
          show form "F":
            fields:
              username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        component_ir = page_ir.components[0]
        
        # Page overrides app
        assert page_ir.design_tokens.theme == "light"
        
        # Component inherits from page
        assert component_ir.design_tokens.theme == "light"
    
    def test_four_level_inheritance_component_override(self):
        """Test component adds tokens, field inherits."""
        dsl = '''
        app "Test" (theme=dark)
        page "Home" at "/":
          show form "F" (variant=outlined, size=lg):
            fields:
              username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        component_ir = page_ir.components[0]
        
        # Component has theme from page + its own tokens
        assert component_ir.design_tokens.theme == "dark"
        assert component_ir.design_tokens.variant == "outlined"
        assert component_ir.design_tokens.size == "lg"
    
    def test_four_level_inheritance_field_override(self):
        """Test complete chain with field override."""
        dsl = '''
        app "Test" (theme=dark, color_scheme=indigo)
        page "Home" at "/":
          show form "F" (variant=outlined, tone=success, size=md):
            fields:
              username: text
              email: text (size=sm, tone=primary)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        component_ir = page_ir.components[0]
        
        # Component inherits + adds
        assert component_ir.design_tokens.theme == "dark"
        assert component_ir.design_tokens.color_scheme == "indigo"
        assert component_ir.design_tokens.variant == "outlined"
        assert component_ir.design_tokens.tone == "success"
        assert component_ir.design_tokens.size == "md"
        
        # Field overrides are in AST
        form_ast = ast.pages[0].body[0]
        assert form_ast.fields[0].size is None  # inherits md
        assert form_ast.fields[1].size == SizeType.SM  # overrides to sm
        assert form_ast.fields[1].tone == ToneType.PRIMARY  # overrides to primary


class TestMultipleComponentsInheritance:
    """Test inheritance with multiple components on same page."""
    
    def test_multiple_components_inherit_independently(self):
        """Test each component inherits from page independently."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark):
          show form "Login" (variant=outlined):
            fields: username: text
          show form "Register" (variant=elevated):
            fields: email: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        component1_ir = page_ir.components[0]
        component2_ir = page_ir.components[1]
        
        # Both inherit theme from page
        assert component1_ir.design_tokens.theme == "dark"
        assert component2_ir.design_tokens.theme == "dark"
        
        # Each has its own variant
        assert component1_ir.design_tokens.variant == "outlined"
        assert component2_ir.design_tokens.variant == "elevated"
    
    def test_multiple_components_different_tokens(self):
        """Test components with different token combinations."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F1" (tone=primary, size=lg):
            fields: name: text
          show form "F2" (tone=success, size=sm):
            fields: email: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        component1_ir = page_ir.components[0]
        component2_ir = page_ir.components[1]
        
        assert component1_ir.design_tokens.tone == "primary"
        assert component1_ir.design_tokens.size == "lg"
        assert component2_ir.design_tokens.tone == "success"
        assert component2_ir.design_tokens.size == "sm"


class TestMultiplePagesInheritance:
    """Test inheritance with multiple pages."""
    
    def test_multiple_pages_inherit_from_app(self):
        """Test multiple pages all inherit from app."""
        dsl = '''
        app "Test" (theme=dark, color_scheme=blue)
        page "Home" at "/":
          show form "F1": fields: name: text
        page "About" at "/about":
          show form "F2": fields: email: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page1_ir = ir.frontend.pages[0]
        page2_ir = ir.frontend.pages[1]
        
        # Both pages inherit from app
        assert page1_ir.design_tokens.theme == "dark"
        assert page1_ir.design_tokens.color_scheme == "blue"
        assert page2_ir.design_tokens.theme == "dark"
        assert page2_ir.design_tokens.color_scheme == "blue"
    
    def test_multiple_pages_with_overrides(self):
        """Test pages can override app tokens independently."""
        dsl = '''
        app "Test" (theme=dark)
        page "Home" at "/" (theme=light):
          show form "F1": fields: name: text
        page "Dashboard" at "/dashboard" (theme=system):
          show form "F2": fields: email: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page1_ir = ir.frontend.pages[0]
        page2_ir = ir.frontend.pages[1]
        
        # Each page overrides independently
        assert page1_ir.design_tokens.theme == "light"
        assert page2_ir.design_tokens.theme == "system"


class TestNoneHandling:
    """Test handling of None/missing tokens in inheritance."""
    
    def test_none_tokens_do_not_propagate(self):
        """Test None at parent doesn't override child."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        # Page has theme, no color_scheme
        assert page_ir.design_tokens.theme == "dark"
        assert page_ir.design_tokens.color_scheme is None
    
    def test_partial_token_inheritance(self):
        """Test inheritance when only some tokens are set."""
        dsl = '''
        app "Test" (theme=dark)
        page "Home" at "/" (color_scheme=blue):
          show form "F" (variant=outlined):
            fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        ast = parser.parse().body[0]
        ir = build_backend_ir(ast)
        
        page_ir = ir.frontend.pages[0]
        component_ir = page_ir.components[0]
        
        # Component inherits theme from app, color_scheme from page
        assert component_ir.design_tokens.theme == "dark"
        assert component_ir.design_tokens.color_scheme == "blue"
        assert component_ir.design_tokens.variant == "outlined"
