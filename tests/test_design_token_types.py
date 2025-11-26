"""
Unit tests for design token type system.
Tests all 6 token types: Variant, Tone, Size, Density, Theme, ColorScheme.
"""

import pytest
from namel3ss.ast.design_tokens import (
    VariantType,
    ToneType,
    SizeType,
    DensityType,
    ThemeType,
    ColorSchemeType,
)


class TestVariantType:
    """Test VariantType enum validation."""
    
    def test_all_variant_values(self):
        """Ensure all variant types are accessible."""
        assert VariantType.ELEVATED.value == "elevated"
        assert VariantType.OUTLINED.value == "outlined"
        assert VariantType.GHOST.value == "ghost"
        assert VariantType.SUBTLE.value == "subtle"
    
    def test_variant_count(self):
        """Verify we have exactly 4 variants."""
        assert len(VariantType) == 4
    
    def test_variant_from_string(self):
        """Test creating VariantType from string values."""
        assert VariantType("elevated") == VariantType.ELEVATED
        assert VariantType("outlined") == VariantType.OUTLINED
        assert VariantType("ghost") == VariantType.GHOST
        assert VariantType("subtle") == VariantType.SUBTLE
    
    def test_variant_invalid_value(self):
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError):
            VariantType("invalid")
    
    def test_variant_case_sensitive(self):
        """Verify variant values are case-sensitive."""
        with pytest.raises(ValueError):
            VariantType("ELEVATED")


class TestToneType:
    """Test ToneType enum validation."""
    
    def test_all_tone_values(self):
        """Ensure all tone types are accessible."""
        assert ToneType.NEUTRAL.value == "neutral"
        assert ToneType.PRIMARY.value == "primary"
        assert ToneType.SUCCESS.value == "success"
        assert ToneType.WARNING.value == "warning"
        assert ToneType.DANGER.value == "danger"
    
    def test_tone_count(self):
        """Verify we have exactly 5 tones."""
        assert len(ToneType) == 5
    
    def test_tone_from_string(self):
        """Test creating ToneType from string values."""
        assert ToneType("neutral") == ToneType.NEUTRAL
        assert ToneType("primary") == ToneType.PRIMARY
        assert ToneType("success") == ToneType.SUCCESS
        assert ToneType("warning") == ToneType.WARNING
        assert ToneType("danger") == ToneType.DANGER
    
    def test_tone_invalid_value(self):
        """Test that invalid tone raises ValueError."""
        with pytest.raises(ValueError):
            ToneType("error")  # Should be "danger"
    
    def test_tone_semantic_meaning(self):
        """Verify tone values match semantic color intentions."""
        # These are the standard semantic color names
        semantic_tones = {"neutral", "primary", "success", "warning", "danger"}
        actual_tones = {tone.value for tone in ToneType}
        assert actual_tones == semantic_tones


class TestSizeType:
    """Test SizeType enum validation."""
    
    def test_all_size_values(self):
        """Ensure all size types are accessible."""
        assert SizeType.XS.value == "xs"
        assert SizeType.SM.value == "sm"
        assert SizeType.MD.value == "md"
        assert SizeType.LG.value == "lg"
        assert SizeType.XL.value == "xl"
    
    def test_size_count(self):
        """Verify we have exactly 5 sizes."""
        assert len(SizeType) == 5
    
    def test_size_from_string(self):
        """Test creating SizeType from string values."""
        assert SizeType("xs") == SizeType.XS
        assert SizeType("sm") == SizeType.SM
        assert SizeType("md") == SizeType.MD
        assert SizeType("lg") == SizeType.LG
        assert SizeType("xl") == SizeType.XL
    
    def test_size_ordering(self):
        """Verify sizes are in logical order (smallest to largest)."""
        sizes = list(SizeType)
        expected_order = [SizeType.XS, SizeType.SM, SizeType.MD, SizeType.LG, SizeType.XL]
        assert sizes == expected_order
    
    def test_size_invalid_value(self):
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError):
            SizeType("medium")  # Should be "md"


class TestDensityType:
    """Test DensityType enum validation."""
    
    def test_all_density_values(self):
        """Ensure all density types are accessible."""
        assert DensityType.COMFORTABLE.value == "comfortable"
        assert DensityType.COMPACT.value == "compact"
    
    def test_density_count(self):
        """Verify we have exactly 2 densities."""
        assert len(DensityType) == 2
    
    def test_density_from_string(self):
        """Test creating DensityType from string values."""
        assert DensityType("comfortable") == DensityType.COMFORTABLE
        assert DensityType("compact") == DensityType.COMPACT
    
    def test_density_invalid_value(self):
        """Test that invalid density raises ValueError."""
        with pytest.raises(ValueError):
            DensityType("spacious")
    
    def test_density_use_cases(self):
        """Verify density values match intended use cases."""
        # Comfortable = default, spacious
        # Compact = data-heavy, dense layouts
        assert DensityType.COMFORTABLE.value == "comfortable"
        assert DensityType.COMPACT.value == "compact"


class TestThemeType:
    """Test ThemeType enum validation."""
    
    def test_all_theme_values(self):
        """Ensure all theme types are accessible."""
        assert ThemeType.LIGHT.value == "light"
        assert ThemeType.DARK.value == "dark"
        assert ThemeType.SYSTEM.value == "system"
    
    def test_theme_count(self):
        """Verify we have exactly 3 themes."""
        assert len(ThemeType) == 3
    
    def test_theme_from_string(self):
        """Test creating ThemeType from string values."""
        assert ThemeType("light") == ThemeType.LIGHT
        assert ThemeType("dark") == ThemeType.DARK
        assert ThemeType("system") == ThemeType.SYSTEM
    
    def test_theme_invalid_value(self):
        """Test that invalid theme raises ValueError."""
        with pytest.raises(ValueError):
            ThemeType("auto")  # Should be "system"
    
    def test_theme_system_behavior(self):
        """Verify system theme is for OS preference matching."""
        # System theme should dynamically switch based on OS
        assert ThemeType.SYSTEM.value == "system"


class TestColorSchemeType:
    """Test ColorSchemeType enum validation."""
    
    def test_all_color_scheme_values(self):
        """Ensure all color scheme types are accessible."""
        assert ColorSchemeType.BLUE.value == "blue"
        assert ColorSchemeType.GREEN.value == "green"
        assert ColorSchemeType.VIOLET.value == "violet"
        assert ColorSchemeType.ROSE.value == "rose"
        assert ColorSchemeType.ORANGE.value == "orange"
        assert ColorSchemeType.TEAL.value == "teal"
        assert ColorSchemeType.INDIGO.value == "indigo"
        assert ColorSchemeType.SLATE.value == "slate"
    
    def test_color_scheme_count(self):
        """Verify we have exactly 8 color schemes."""
        assert len(ColorSchemeType) == 8
    
    def test_color_scheme_from_string(self):
        """Test creating ColorSchemeType from string values."""
        assert ColorSchemeType("blue") == ColorSchemeType.BLUE
        assert ColorSchemeType("green") == ColorSchemeType.GREEN
        assert ColorSchemeType("violet") == ColorSchemeType.VIOLET
        assert ColorSchemeType("rose") == ColorSchemeType.ROSE
        assert ColorSchemeType("orange") == ColorSchemeType.ORANGE
        assert ColorSchemeType("teal") == ColorSchemeType.TEAL
        assert ColorSchemeType("indigo") == ColorSchemeType.INDIGO
        assert ColorSchemeType("slate") == ColorSchemeType.SLATE
    
    def test_color_scheme_invalid_value(self):
        """Test that invalid color scheme raises ValueError."""
        with pytest.raises(ValueError):
            ColorSchemeType("red")  # Should be "rose" or "danger"
    
    def test_color_scheme_variety(self):
        """Verify we have diverse color options."""
        color_schemes = {scheme.value for scheme in ColorSchemeType}
        # Should include cool, warm, and neutral colors
        assert "blue" in color_schemes  # Cool
        assert "orange" in color_schemes  # Warm
        assert "slate" in color_schemes  # Neutral


class TestTokenTypeInteroperability:
    """Test how token types work together."""
    
    def test_token_types_are_independent(self):
        """Verify each token type can be used independently."""
        # Should be able to create combinations
        variant = VariantType.ELEVATED
        tone = ToneType.PRIMARY
        size = SizeType.MD
        density = DensityType.COMFORTABLE
        theme = ThemeType.DARK
        color_scheme = ColorSchemeType.BLUE
        
        # All should be valid
        assert variant.value == "elevated"
        assert tone.value == "primary"
        assert size.value == "md"
        assert density.value == "comfortable"
        assert theme.value == "dark"
        assert color_scheme.value == "blue"
    
    def test_token_type_combinations(self):
        """Test common token combinations."""
        # Primary button: elevated + primary + medium
        assert VariantType.ELEVATED.value == "elevated"
        assert ToneType.PRIMARY.value == "primary"
        assert SizeType.MD.value == "md"
        
        # Secondary input: outlined + neutral + small
        assert VariantType.OUTLINED.value == "outlined"
        assert ToneType.NEUTRAL.value == "neutral"
        assert SizeType.SM.value == "sm"
        
        # Data table: elevated + neutral + compact
        assert VariantType.ELEVATED.value == "elevated"
        assert ToneType.NEUTRAL.value == "neutral"
        assert DensityType.COMPACT.value == "compact"
    
    def test_all_token_types_have_values(self):
        """Ensure no token type is empty."""
        assert len(VariantType) > 0
        assert len(ToneType) > 0
        assert len(SizeType) > 0
        assert len(DensityType) > 0
        assert len(ThemeType) > 0
        assert len(ColorSchemeType) > 0


class TestTokenTypeEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_none_values(self):
        """Verify None is not a valid token value."""
        with pytest.raises((ValueError, TypeError)):
            VariantType(None)
    
    def test_empty_string(self):
        """Verify empty string is not a valid token value."""
        with pytest.raises(ValueError):
            VariantType("")
    
    def test_numeric_values(self):
        """Verify numeric values are rejected."""
        with pytest.raises((ValueError, TypeError)):
            SizeType(42)
    
    def test_enum_equality(self):
        """Test enum equality comparison."""
        assert VariantType.ELEVATED == VariantType.ELEVATED
        assert VariantType.ELEVATED != VariantType.OUTLINED
        assert ToneType.PRIMARY != ToneType.SUCCESS
    
    def test_enum_membership(self):
        """Test checking if value is in enum."""
        assert "elevated" in [v.value for v in VariantType]
        assert "invalid" not in [v.value for v in VariantType]
    
    def test_token_immutability(self):
        """Verify token enums are immutable."""
        variant = VariantType.ELEVATED
        with pytest.raises(AttributeError):
            variant.value = "outlined"
