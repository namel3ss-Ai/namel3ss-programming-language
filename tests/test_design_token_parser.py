"""
Tests for parsing design tokens from DSL syntax.
Tests all token types, field-level overrides, and error handling.
"""

import pytest
import textwrap
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ast.design_tokens import (
    VariantType,
    ToneType,
    SizeType,
    DensityType,
    ThemeType,
    ColorSchemeType,
)


class TestPageLevelTokenParsing:
    """Test parsing design tokens at page level."""
    
    def test_parse_page_theme_light(self):
        """Test parsing light theme on page."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=light):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        
        assert app.pages[0].theme == ThemeType.LIGHT
    
    def test_parse_page_theme_dark(self):
        """Test parsing dark theme on page."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        
        assert app.pages[0].theme == ThemeType.DARK
    
    def test_parse_page_theme_system(self):
        """Test parsing system theme on page."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=system):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        
        assert app.pages[0].theme == ThemeType.SYSTEM
    
    def test_parse_page_color_scheme(self):
        """Test parsing color scheme on page."""
        dsl = '''
        app "Test"
        page "Home" at "/" (color_scheme=indigo):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        
        assert app.pages[0].color_scheme == ColorSchemeType.INDIGO
    
    def test_parse_page_theme_and_color_scheme(self):
        """Test parsing both theme and color scheme."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=dark, color_scheme=violet):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        
        assert app.pages[0].theme == ThemeType.DARK
        assert app.pages[0].color_scheme == ColorSchemeType.VIOLET
    
    def test_parse_page_no_tokens(self):
        """Test page without design tokens."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        
        assert app.pages[0].theme is None
        assert app.pages[0].color_scheme is None


class TestComponentLevelTokenParsing:
    """Test parsing design tokens at component level."""
    
    def test_parse_form_variant(self):
        """Test parsing form variant token."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (variant=outlined):
            fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        form = app.pages[0].body[0]
        
        assert form.variant == VariantType.OUTLINED
    
    def test_parse_form_tone(self):
        """Test parsing form tone token."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (tone=primary):
            fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        form = app.pages[0].body[0]
        
        assert form.tone == ToneType.PRIMARY
    
    def test_parse_form_size(self):
        """Test parsing form size token."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (size=lg):
            fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        form = app.pages[0].body[0]
        
        assert form.size == SizeType.LG
    
    def test_parse_form_all_tokens(self):
        """Test parsing all form tokens together."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (variant=elevated, tone=success, size=md):
            fields: username: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        form = app.pages[0].body[0]
        
        assert form.variant == VariantType.ELEVATED
        assert form.tone == ToneType.SUCCESS
        assert form.size == SizeType.MD
    
    def test_parse_table_density(self):
        """Test parsing table density token."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show table from model "User" (density=compact)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        table = app.pages[0].body[0]
        
        assert table.density == DensityType.COMPACT


class TestFieldLevelTokenParsing:
    """Test parsing field-level design token overrides."""
    
    def test_parse_field_size_override(self):
        """Test parsing size override on individual field."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (size=md):
            fields:
              username: text
              password: text (size=sm)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        form = app.pages[0].body[0]
        
        assert form.size == SizeType.MD
        assert form.fields[0].size is None  # username inherits
        assert form.fields[1].size == SizeType.SM  # password overrides
    
    def test_parse_field_tone_override(self):
        """Test parsing tone override on individual field."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (tone=neutral):
            fields:
              username: text
              email: text (tone=primary)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        form = app.pages[0].body[0]
        
        assert form.tone == ToneType.NEUTRAL
        assert form.fields[0].tone is None  # inherits
        assert form.fields[1].tone == ToneType.PRIMARY  # overrides
    
    def test_parse_field_variant_override(self):
        """Test parsing variant override on individual field."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (variant=outlined):
            fields:
              username: text
              password: text (variant=ghost)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        form = app.pages[0].body[0]
        
        assert form.variant == VariantType.OUTLINED
        assert form.fields[0].variant is None
        assert form.fields[1].variant == VariantType.GHOST
    
    def test_parse_field_multiple_overrides(self):
        """Test parsing multiple token overrides on one field."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (variant=outlined, tone=neutral, size=md):
            fields:
              username: text
              email: text (size=sm, tone=primary)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        form = app.pages[0].body[0]
        
        # Form level
        assert form.variant == VariantType.OUTLINED
        assert form.tone == ToneType.NEUTRAL
        assert form.size == SizeType.MD
        
        # Field level
        assert form.fields[1].size == SizeType.SM
        assert form.fields[1].tone == ToneType.PRIMARY
        assert form.fields[1].variant is None  # Not overridden


class TestAllTokenTypesInDSL:
    """Test parsing all token types in comprehensive DSL."""
    
    def test_parse_all_variants(self):
        """Test parsing all variant types."""
        variants = ["elevated", "outlined", "ghost", "subtle"]
        
        for variant in variants:
            dsl = f'''
            app "Test"
            page "Home" at "/":
              show form "F" (variant={variant}):
                fields: name: text
            '''
            parser = LegacyProgramParser(textwrap.dedent(dsl))
            result = parser.parse()
            app = result.body[0]
            form = app.pages[0].body[0]
            
            assert form.variant.value == variant
    
    def test_parse_all_tones(self):
        """Test parsing all tone types."""
        tones = ["neutral", "primary", "success", "warning", "danger"]
        
        for tone in tones:
            dsl = f'''
            app "Test"
            page "Home" at "/":
              show form "F" (tone={tone}):
                fields: name: text
            '''
            parser = LegacyProgramParser(textwrap.dedent(dsl))
            result = parser.parse()
            app = result.body[0]
            form = app.pages[0].body[0]
            
            assert form.tone.value == tone
    
    def test_parse_all_sizes(self):
        """Test parsing all size types."""
        sizes = ["xs", "sm", "md", "lg", "xl"]
        
        for size in sizes:
            dsl = f'''
            app "Test"
            page "Home" at "/":
              show form "F" (size={size}):
                fields: name: text
            '''
            parser = LegacyProgramParser(textwrap.dedent(dsl))
            result = parser.parse()
            app = result.body[0]
            form = app.pages[0].body[0]
            
            assert form.size.value == size
    
    def test_parse_all_densities(self):
        """Test parsing all density types."""
        densities = ["comfortable", "compact"]
        
        for density in densities:
            dsl = f'''
            app "Test"
            page "Home" at "/":
              show table from model "User" (density={density})
            '''
            parser = LegacyProgramParser(textwrap.dedent(dsl))
            result = parser.parse()
            app = result.body[0]
            table = app.pages[0].body[0]
            
            assert table.density.value == density
    
    def test_parse_all_themes(self):
        """Test parsing all theme types."""
        themes = ["light", "dark", "system"]
        
        for theme in themes:
            dsl = f'''
            app "Test"
            page "Home" at "/" (theme={theme}):
              show form "F": fields: name: text
            '''
            parser = LegacyProgramParser(textwrap.dedent(dsl))
            result = parser.parse()
            app = result.body[0]
            
            assert app.pages[0].theme.value == theme
    
    def test_parse_all_color_schemes(self):
        """Test parsing all color scheme types."""
        color_schemes = ["blue", "green", "violet", "rose", "orange", "teal", "indigo", "slate"]
        
        for color_scheme in color_schemes:
            dsl = f'''
            app "Test"
            page "Home" at "/" (color_scheme={color_scheme}):
              show form "F": fields: name: text
            '''
            parser = LegacyProgramParser(textwrap.dedent(dsl))
            result = parser.parse()
            app = result.body[0]
            
            assert app.pages[0].color_scheme.value == color_scheme


class TestComplexDSLParsing:
    """Test parsing complex DSL with multiple pages and components."""
    
    def test_parse_multiple_pages_with_tokens(self):
        """Test parsing multiple pages each with different tokens."""
        dsl = '''
        app "Platform"
        page "Dashboard" at "/" (theme=dark, color_scheme=indigo):
          show form "Login": fields: username: text
        
        page "Settings" at "/settings" (theme=light, color_scheme=blue):
          show form "Preferences": fields: email: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        
        assert len(app.pages) == 2
        assert app.pages[0].theme == ThemeType.DARK
        assert app.pages[0].color_scheme == ColorSchemeType.INDIGO
        assert app.pages[1].theme == ThemeType.LIGHT
        assert app.pages[1].color_scheme == ColorSchemeType.BLUE
    
    def test_parse_multiple_components_with_tokens(self):
        """Test parsing multiple components with different tokens."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "Login" (variant=outlined, tone=primary):
            fields: username: text
          show form "Register" (variant=elevated, tone=success):
            fields: email: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        
        form1 = app.pages[0].body[0]
        form2 = app.pages[0].body[1]
        
        assert form1.variant == VariantType.OUTLINED
        assert form1.tone == ToneType.PRIMARY
        assert form2.variant == VariantType.ELEVATED
        assert form2.tone == ToneType.SUCCESS
    
    def test_parse_comprehensive_token_usage(self):
        """Test parsing DSL with all token types used together."""
        dsl = '''
        app "Medical"
        page "Intake" at "/" (theme=system, color_scheme=teal):
          show form "Patient" (variant=outlined, tone=success, size=lg):
            fields:
              name: text
              email: text (size=md, tone=primary)
              phone: text (size=sm)
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        result = parser.parse()
        app = result.body[0]
        page = app.pages[0]
        form = page.body[0]
        
        # Page level
        assert page.theme == ThemeType.SYSTEM
        assert page.color_scheme == ColorSchemeType.TEAL
        
        # Component level
        assert form.variant == VariantType.OUTLINED
        assert form.tone == ToneType.SUCCESS
        assert form.size == SizeType.LG
        
        # Field level
        assert form.fields[0].size is None  # inherits lg
        assert form.fields[1].size == SizeType.MD  # overrides
        assert form.fields[1].tone == ToneType.PRIMARY  # overrides
        assert form.fields[2].size == SizeType.SM  # overrides


class TestParserErrorHandling:
    """Test parser error handling for invalid tokens."""
    
    def test_invalid_variant_value(self):
        """Test parsing with invalid variant raises error."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (variant=invalid): fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        
        with pytest.raises(Exception):  # Parser should raise on invalid enum
            parser.parse()
    
    def test_invalid_tone_value(self):
        """Test parsing with invalid tone raises error."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (tone=error): fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        
        with pytest.raises(Exception):
            parser.parse()
    
    def test_invalid_size_value(self):
        """Test parsing with invalid size raises error."""
        dsl = '''
        app "Test"
        page "Home" at "/":
          show form "F" (size=medium): fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        
        with pytest.raises(Exception):
            parser.parse()
    
    def test_invalid_theme_value(self):
        """Test parsing with invalid theme raises error."""
        dsl = '''
        app "Test"
        page "Home" at "/" (theme=auto):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        
        with pytest.raises(Exception):
            parser.parse()
    
    def test_invalid_color_scheme_value(self):
        """Test parsing with invalid color scheme raises error."""
        dsl = '''
        app "Test"
        page "Home" at "/" (color_scheme=red):
          show form "F": fields: name: text
        '''
        parser = LegacyProgramParser(textwrap.dedent(dsl))
        
        with pytest.raises(Exception):
            parser.parse()
