"""
Test design token implementation across parser, AST, and validation.
"""
import textwrap
import pytest
from namel3ss.parser.program import LegacyProgramParser
from namel3ss.ast.design_tokens import (
    VariantType,
    ToneType,
    DensityType,
    SizeType,
    ThemeType,
    ColorSchemeType,
    validate_variant,
    validate_tone,
    validate_density,
    validate_size,
    validate_theme,
    validate_color_scheme,
)
from namel3ss.parser.base import N3SyntaxError


def parse(source: str):
    """Helper to parse dedented source code."""
    return LegacyProgramParser(textwrap.dedent(source)).parse()


# =============================================================================
# Design Token Type Validation Tests
# =============================================================================

def test_validate_variant_valid():
    """Test that all valid variant values are accepted."""
    assert validate_variant("elevated") == VariantType.ELEVATED
    assert validate_variant("outlined") == VariantType.OUTLINED
    assert validate_variant("ghost") == VariantType.GHOST
    assert validate_variant("subtle") == VariantType.SUBTLE


def test_validate_variant_invalid():
    """Test that invalid variant values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid variant"):
        validate_variant("fancy")
    with pytest.raises(ValueError, match="Invalid variant"):
        validate_variant("primary")


def test_validate_tone_valid():
    """Test that all valid tone values are accepted."""
    assert validate_tone("neutral") == ToneType.NEUTRAL
    assert validate_tone("primary") == ToneType.PRIMARY
    assert validate_tone("success") == ToneType.SUCCESS
    assert validate_tone("warning") == ToneType.WARNING
    assert validate_tone("danger") == ToneType.DANGER


def test_validate_tone_invalid():
    """Test that invalid tone values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid tone"):
        validate_tone("error")
    with pytest.raises(ValueError, match="Invalid tone"):
        validate_tone("info")


def test_validate_density_valid():
    """Test that all valid density values are accepted."""
    assert validate_density("comfortable") == DensityType.COMFORTABLE
    assert validate_density("compact") == DensityType.COMPACT


def test_validate_density_invalid():
    """Test that invalid density values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid density"):
        validate_density("spacious")
    with pytest.raises(ValueError, match="Invalid density"):
        validate_density("dense")


def test_validate_size_valid():
    """Test that all valid size values are accepted."""
    assert validate_size("xs") == SizeType.XS
    assert validate_size("sm") == SizeType.SM
    assert validate_size("md") == SizeType.MD
    assert validate_size("lg") == SizeType.LG
    assert validate_size("xl") == SizeType.XL


def test_validate_size_invalid():
    """Test that invalid size values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid size"):
        validate_size("xxl")
    with pytest.raises(ValueError, match="Invalid size"):
        validate_size("small")


def test_validate_theme_valid():
    """Test that all valid theme values are accepted."""
    assert validate_theme("light") == ThemeType.LIGHT
    assert validate_theme("dark") == ThemeType.DARK
    assert validate_theme("system") == ThemeType.SYSTEM


def test_validate_theme_invalid():
    """Test that invalid theme values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid theme"):
        validate_theme("auto")


def test_validate_color_scheme_valid():
    """Test that all valid color scheme values are accepted."""
    assert validate_color_scheme("blue") == ColorSchemeType.BLUE
    assert validate_color_scheme("green") == ColorSchemeType.GREEN
    assert validate_color_scheme("violet") == ColorSchemeType.VIOLET
    assert validate_color_scheme("rose") == ColorSchemeType.ROSE
    assert validate_color_scheme("orange") == ColorSchemeType.ORANGE
    assert validate_color_scheme("teal") == ColorSchemeType.TEAL
    assert validate_color_scheme("indigo") == ColorSchemeType.INDIGO
    assert validate_color_scheme("slate") == ColorSchemeType.SLATE


def test_validate_color_scheme_invalid():
    """Test that invalid color scheme values raise ValueError."""
    with pytest.raises(ValueError, match="Invalid color_scheme"):
        validate_color_scheme("red")
    with pytest.raises(ValueError, match="Invalid color_scheme"):
        validate_color_scheme("purple")


# =============================================================================
# Parser Tests - Page Level Design Tokens
# =============================================================================

def test_parse_page_with_theme():
    """Test parsing page with theme token."""
    source = """app "Test"
  page "home":
    theme: light
    show text "Hello"
"""
    module = parse(source)
    
    assert len(module.apps) == 1
    app = module.body[0]
    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.theme == ThemeType.LIGHT


def test_parse_page_with_color_scheme():
    """Test parsing page with color_scheme token."""
    source = """app "Test"
  page "home":
    color_scheme: blue
    show text "Hello"
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    assert page.color_scheme == ColorSchemeType.BLUE


def test_parse_page_with_theme_and_color_scheme():
    """Test parsing page with both theme and color_scheme."""
    source = """app "Test"
  page "home":
    theme: dark
    color_scheme: violet
    show text "Hello"
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    assert page.theme == ThemeType.DARK
    assert page.color_scheme == ColorSchemeType.VIOLET


# =============================================================================
# Parser Tests - Component Level Design Tokens
# =============================================================================

def test_parse_show_card_with_design_tokens():
    """Test parsing ShowCard with design tokens."""
    source = """app "Test"
  page "home":
    show card "Items" from dataset items:
      variant: elevated
      tone: primary
      density: compact
      size: md
      item:
        header:
          title: name
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    card = page.body[0]
    assert card.variant == VariantType.ELEVATED
    assert card.tone == ToneType.PRIMARY
    assert card.density == DensityType.COMPACT
    assert card.size == SizeType.MD


def test_parse_show_list_with_design_tokens():
    """Test parsing ShowList with design tokens."""
    source = """app "Test"
  page "home":
    show list "Items" from dataset items:
      variant: outlined
      tone: success
      density: comfortable
      size: lg
      item:
        header:
          title: name
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    list_comp = page.body[0]
    assert list_comp.variant == VariantType.OUTLINED
    assert list_comp.tone == ToneType.SUCCESS
    assert list_comp.density == DensityType.COMFORTABLE
    assert list_comp.size == SizeType.LG


def test_parse_show_table_with_design_tokens():
    """Test parsing ShowTable with design tokens."""
    source = """app "Test"
  page "home":
    show table "Data" from dataset records:
      columns: id, name, status
      variant: ghost
      tone: warning
      density: compact
      size: sm
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    table = page.body[0]
    assert table.variant == VariantType.GHOST
    assert table.tone == ToneType.WARNING
    assert table.density == DensityType.COMPACT
    assert table.size == SizeType.SM


def test_parse_show_form_with_design_tokens():
    """Test parsing ShowForm with design tokens."""
    source = """app "Test"
  page "home":
    show form "Edit":
      variant: subtle
      tone: danger
      density: comfortable
      size: xl
      fields:
        - name: email
          component: text_input
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    form = page.body[0]
    assert form.variant == VariantType.SUBTLE
    assert form.tone == ToneType.DANGER
    assert form.density == DensityType.COMFORTABLE
    assert form.size == SizeType.XL


def test_parse_form_field_with_design_tokens():
    """Test parsing FormField with design tokens."""
    source = """app "Test"
  page "home":
    show form "Edit":
      fields:
        - name: email
          component: text_input
          variant: outlined
          tone: primary
          size: md
"""
    module = parse(source)
    
    form = module.body[0].pages[0].body[0]
    field = form.fields[0]
    assert field.variant == VariantType.OUTLINED
    assert field.tone == ToneType.PRIMARY
    assert field.size == SizeType.MD


def test_parse_modal_with_design_tokens():
    """Test parsing Modal with design tokens."""
    source = """app "Test"
  page "home":
    modal "confirm":
      title: "Confirm Action"
      variant: elevated
      tone: warning
      modal_size: lg
      dismissible: true
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    modal = page.body[0]
    assert modal.variant == VariantType.ELEVATED
    assert modal.tone == ToneType.WARNING
    assert modal.modal_size == SizeType.LG


def test_parse_toast_with_design_tokens():
    """Test parsing Toast with design tokens."""
    source = """app "Test"
  page "home":
    toast "notification":
      title: "Success"
      toast_variant: subtle
      tone: success
      size: md
      duration: 3000
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    toast = page.body[0]
    assert toast.toast_variant == VariantType.SUBTLE
    assert toast.tone == ToneType.SUCCESS
    assert toast.size == SizeType.MD


# =============================================================================
# Parser Error Tests - Invalid Design Tokens
# =============================================================================

def test_parse_invalid_variant():
    """Test that invalid variant raises parse error."""
    source = """app "Test"
  page "home":
    show card "Items" from dataset items:
      variant: fancy
      item:
        header:
          title: name
"""
    parser = Parser(source)
    with pytest.raises(N3SyntaxError, match="Invalid variant"):
        parser.parse()


def test_parse_invalid_tone():
    """Test that invalid tone raises parse error."""
    source = """app "Test"
  page "home":
    show card "Items" from dataset items:
      tone: error
      item:
        header:
          title: name
"""
    parser = Parser(source)
    with pytest.raises(N3SyntaxError, match="Invalid tone"):
        parser.parse()


def test_parse_invalid_density():
    """Test that invalid density raises parse error."""
    source = """app "Test"
  page "home":
    show card "Items" from dataset items:
      density: spacious
      item:
        header:
          title: name
"""
    parser = Parser(source)
    with pytest.raises(N3SyntaxError, match="Invalid density"):
        parser.parse()


def test_parse_invalid_size():
    """Test that invalid size raises parse error."""
    source = """app "Test"
  page "home":
    show card "Items" from dataset items:
      size: xxl
      item:
        header:
          title: name
"""
    parser = Parser(source)
    with pytest.raises(N3SyntaxError, match="Invalid size"):
        parser.parse()


def test_parse_invalid_theme():
    """Test that invalid theme raises parse error."""
    source = """app "Test"
  page "home":
    theme: auto
    show text "Hello"
"""
    parser = Parser(source)
    with pytest.raises(N3SyntaxError, match="Invalid theme"):
        parser.parse()


def test_parse_invalid_color_scheme():
    """Test that invalid color_scheme raises parse error."""
    source = """app "Test"
  page "home":
    color_scheme: red
    show text "Hello"
"""
    parser = Parser(source)
    with pytest.raises(N3SyntaxError, match="Invalid color_scheme"):
        parser.parse()


# =============================================================================
# Integration Tests - Multiple Components
# =============================================================================

def test_parse_complex_page_with_mixed_design_tokens():
    """Test parsing a complex page with multiple components using design tokens."""
    source = """app "Test"
  page "dashboard":
    theme: dark
    color_scheme: indigo
    
    show card "Users" from dataset users:
      variant: elevated
      tone: primary
      density: compact
      size: md
      item:
        header:
          title: name
    
    show list "Tasks" from dataset tasks:
      variant: outlined
      tone: success
      size: lg
      item:
        header:
          title: title
    
    show form "Create":
      variant: ghost
      tone: neutral
      density: comfortable
      size: md
      fields:
        - name: name
          component: text_input
          variant: outlined
          tone: primary
          size: sm
"""
    module = parse(source)
    
    page = module.body[0].pages[0]
    
    # Check page-level tokens
    assert page.theme == ThemeType.DARK
    assert page.color_scheme == ColorSchemeType.INDIGO
    
    # Check card tokens
    card = page.body[0]
    assert card.variant == VariantType.ELEVATED
    assert card.tone == ToneType.PRIMARY
    assert card.density == DensityType.COMPACT
    assert card.size == SizeType.MD
    
    # Check list tokens
    list_comp = page.body[1]
    assert list_comp.variant == VariantType.OUTLINED
    assert list_comp.tone == ToneType.SUCCESS
    assert list_comp.size == SizeType.LG
    
    # Check form tokens
    form = page.body[2]
    assert form.variant == VariantType.GHOST
    assert form.tone == ToneType.NEUTRAL
    assert form.density == DensityType.COMFORTABLE
    assert form.size == SizeType.MD
    
    # Check field tokens
    field = form.fields[0]
    assert field.variant == VariantType.OUTLINED
    assert field.tone == ToneType.PRIMARY
    assert field.size == SizeType.SM


def test_parse_all_color_schemes():
    """Test that all color schemes can be parsed."""
    color_schemes = ["blue", "green", "violet", "rose", "orange", "teal", "indigo", "slate"]
    
    for scheme in color_schemes:
        source = f"""
app "Test"
  page "home":
    color_scheme: {scheme}
    show text "Hello"
"""
        module = parse(source)
        page = module.body[0].pages[0]
        assert page.color_scheme is not None


def test_parse_all_variants():
    """Test that all variants can be parsed."""
    variants = ["elevated", "outlined", "ghost", "subtle"]
    
    for variant in variants:
        source = f"""
app "Test"
  page "home":
    show card "Items" from dataset items:
      variant: {variant}
      item:
        header:
          title: name
"""
        module = parse(source)
        card = module.body[0].pages[0].body[0]
        assert card.variant is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
