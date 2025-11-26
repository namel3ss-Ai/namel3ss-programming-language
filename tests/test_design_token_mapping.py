"""
Tests for Tailwind CSS class mapping functions.
Tests all mapping functions with various token combinations.
"""

import pytest
from namel3ss.codegen.frontend.design_token_mapping import (
    map_button_classes,
    map_input_classes,
    map_card_classes,
    map_table_classes,
    map_badge_classes,
    map_alert_classes,
    map_density_classes,
)


class TestButtonMapping:
    """Test map_button_classes function."""
    
    def test_elevated_primary_medium(self):
        """Test elevated primary button with medium size."""
        result = map_button_classes("elevated", "primary", "md")
        
        assert "bg-blue-600" in result
        assert "text-white" in result
        assert "hover:bg-blue-700" in result
        assert "py-2" in result
        assert "px-4" in result
        assert "rounded-md" in result
    
    def test_outlined_success_large(self):
        """Test outlined success button with large size."""
        result = map_button_classes("outlined", "success", "lg")
        
        assert "border-2" in result
        assert "border-green" in result or "green" in result
        assert "text-green" in result
        assert "hover:bg-gray" in result or "hover:bg-green" in result
        assert "py-3" in result
        assert "px-6" in result
    
    def test_ghost_danger_small(self):
        """Test ghost danger button with small size."""
        result = map_button_classes("ghost", "danger", "sm")
        
        assert "bg-transparent" in result
        assert "text-red-600" in result
        assert "hover:bg-gray" in result or "hover:bg-red" in result
        assert "py-1.5" in result
        assert "px-3" in result
    
    def test_subtle_neutral_xs(self):
        """Test subtle neutral button with xs size."""
        result = map_button_classes("subtle", "neutral", "xs")
        
        assert "bg-gray-100" in result
        assert "text-gray-700" in result or "text-gray" in result
        assert "hover:bg-gray-200" in result or "hover:bg-gray" in result
        assert "py-1" in result
        assert "px-2" in result
    
    def test_all_variants(self):
        """Test all variant types for buttons."""
        variants = ["elevated", "outlined", "ghost", "subtle"]
        
        for variant in variants:
            result = map_button_classes(variant, "primary", "md")
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_all_tones(self):
        """Test all tone types for buttons."""
        tones = ["neutral", "primary", "success", "warning", "danger"]
        
        for tone in tones:
            result = map_button_classes("elevated", tone, "md")
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_all_sizes(self):
        """Test all size types for buttons."""
        sizes = ["xs", "sm", "md", "lg", "xl"]
        
        for size in sizes:
            result = map_button_classes("elevated", "primary", size)
            assert isinstance(result, str)
            # Check size-specific padding is present
            if size == "xs":
                assert "py-1" in result
            elif size == "sm":
                assert "py-1.5" in result
            elif size == "md":
                assert "py-2" in result
            elif size == "lg":
                assert "py-3" in result
            elif size == "xl":
                assert "py-4" in result or "py-3.5" in result


class TestInputMapping:
    """Test map_input_classes function."""
    
    def test_outlined_primary_medium(self):
        """Test outlined primary input with medium size."""
        result = map_input_classes("outlined", "primary", "md")
        
        assert "border-2" in result
        assert "border-blue" in result
        assert "focus:border-blue" in result or "focus:ring-blue" in result
        assert "py-2" in result
        assert "px-4" in result
        assert "rounded-md" in result
    
    def test_elevated_neutral_large(self):
        """Test elevated neutral input with large size."""
        result = map_input_classes("elevated", "neutral", "lg")
        
        assert "border" in result
        assert "border-gray" in result
        assert "shadow-sm" in result
        assert "py-3" in result
        assert "px-5" in result
    
    def test_ghost_success_small(self):
        """Test ghost success input with small size."""
        result = map_input_classes("ghost", "success", "sm")
        
        assert "border-0" in result or "border-none" in result
        assert "bg-transparent" in result
        assert "py-1.5" in result
        assert "px-3" in result
    
    def test_subtle_warning_xs(self):
        """Test subtle warning input with xs size."""
        result = map_input_classes("subtle", "warning", "xs")
        
        assert "bg-gray" in result or "bg-orange" in result
        assert "py-1" in result
        assert "px-2" in result
    
    def test_all_variants(self):
        """Test all variant types for inputs."""
        variants = ["elevated", "outlined", "ghost", "subtle"]
        
        for variant in variants:
            result = map_input_classes(variant, "neutral", "md")
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_all_tones(self):
        """Test all tone types for inputs."""
        tones = ["neutral", "primary", "success", "warning", "danger"]
        
        for tone in tones:
            result = map_input_classes("outlined", tone, "md")
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_all_sizes(self):
        """Test all size types for inputs."""
        sizes = ["xs", "sm", "md", "lg", "xl"]
        
        for size in sizes:
            result = map_input_classes("outlined", "neutral", size)
            assert isinstance(result, str)
            assert "px-" in result  # All sizes should have padding


class TestBadgeMapping:
    """Test map_badge_classes function."""
    
    def test_subtle_primary_small(self):
        """Test subtle primary badge with small size."""
        result = map_badge_classes("subtle", "primary", "sm")
        
        assert "inline-flex" in result
        assert "rounded-full" in result
        assert isinstance(result, str)
    
    def test_elevated_success_medium(self):
        """Test elevated success badge with medium size."""
        result = map_badge_classes("elevated", "success", "md")
        
        assert "inline-flex" in result
        assert isinstance(result, str)
    
    def test_outlined_danger_large(self):
        """Test outlined danger badge with large size."""
        result = map_badge_classes("outlined", "danger", "lg")
        
        assert "inline-flex" in result
        assert isinstance(result, str)
    
    def test_all_sizes(self):
        """Test all size types for badges."""
        sizes = ["xs", "sm", "md", "lg", "xl"]
        
        for size in sizes:
            result = map_badge_classes("subtle", "neutral", size)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_all_combinations(self):
        """Test various badge combinations."""
        combinations = [
            ("elevated", "neutral", "md"),
            ("outlined", "primary", "sm"),
            ("subtle", "success", "xs"),
            ("ghost", "warning", "md"),
        ]
        
        for variant, tone, size in combinations:
            result = map_badge_classes(variant, tone, size)
            assert isinstance(result, str)
            assert len(result) > 0


class TestAlertMapping:
    """Test map_alert_classes function."""
    
    def test_subtle_success(self):
        """Test subtle success alert."""
        result = map_alert_classes("subtle", "success")
        
        assert "rounded-lg" in result
        assert "p-4" in result
        assert isinstance(result, str)
    
    def test_outlined_danger(self):
        """Test outlined danger alert."""
        result = map_alert_classes("outlined", "danger")
        
        assert "rounded-lg" in result
        assert isinstance(result, str)
    
    def test_elevated_warning(self):
        """Test elevated warning alert."""
        result = map_alert_classes("elevated", "warning")
        
        assert isinstance(result, str)
    
    def test_all_tones(self):
        """Test all tone types for alerts."""
        tones = ["neutral", "primary", "success", "warning", "danger"]
        
        for tone in tones:
            result = map_alert_classes("subtle", tone)
            assert isinstance(result, str)
            assert len(result) > 0


class TestDensityMapping:
    """Test map_density_classes function."""
    
    def test_comfortable_default(self):
        """Test comfortable density for default component."""
        result = map_density_classes("comfortable", "default")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_compact_default(self):
        """Test compact density for default component."""
        result = map_density_classes("compact", "default")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_comfortable_row(self):
        """Test comfortable density for row component."""
        result = map_density_classes("comfortable", "row")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_compact_row(self):
        """Test compact density for row component."""
        result = map_density_classes("compact", "row")
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestTableMapping:
    """Test map_table_classes function."""
    
    def test_elevated_neutral_comfortable(self):
        """Test elevated neutral table with comfortable density."""
        result = map_table_classes("elevated", "neutral", "md", "comfortable")
        
        assert "bg-white" in result
        assert "shadow" in result
        assert "border" in result
        assert "border-gray-200" in result
        # Comfortable = more spacing
    
    def test_outlined_primary_compact(self):
        """Test outlined primary table with compact density."""
        result = map_table_classes("outlined", "primary", "sm", "compact")
        
        assert "border" in result
        assert "blue" in result
        # Compact = less spacing
        assert "py-1.5" in result or "h-8" in result
    
    def test_subtle_success_comfortable(self):
        """Test subtle success table with comfortable density."""
        result = map_table_classes("subtle", "success", "lg", "comfortable")
        
        assert "bg-green-50" in result
    
    def test_ghost_danger_compact(self):
        """Test ghost danger table with compact density."""
        result = map_table_classes("ghost", "danger", "md", "compact")
        
        assert "border-0" in result or "red" in result
    
    def test_all_densities(self):
        """Test both density types for tables."""
        densities = ["comfortable", "compact"]
        
        for density in densities:
            result = map_table_classes("elevated", "neutral", "md", density)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_all_variants_with_density(self):
        """Test all variants work with density."""
        variants = ["elevated", "outlined", "ghost", "subtle"]
        
        for variant in variants:
            result = map_table_classes(variant, "neutral", "md", "comfortable")
            assert isinstance(result, str)
            assert len(result) > 0


class TestCardMapping:
    """Test map_card_classes function."""
    
    def test_elevated_primary(self):
        """Test elevated primary card."""
        result = map_card_classes("elevated", "primary")
        
        assert "bg-white" in result
        assert "shadow" in result  # Could be shadow-md or shadow-lg
        assert "border" in result
        assert "blue" in result
        assert "rounded-lg" in result
    
    def test_outlined_success(self):
        """Test outlined success card."""
        result = map_card_classes("outlined", "success")
        
        assert "border-2" in result
        assert "green" in result
        assert "bg-white" in result or "bg-green" in result
        assert "rounded-lg" in result
    
    def test_subtle_neutral(self):
        """Test subtle neutral card."""
        result = map_card_classes("subtle", "neutral")
        
        assert "bg-gray" in result
        assert "rounded-lg" in result
    
    def test_ghost_danger(self):
        """Test ghost danger card."""
        result = map_card_classes("ghost", "danger")
        
        assert "bg-transparent" in result
        assert "rounded-lg" in result
        assert "red" in result
    
    def test_all_variants(self):
        """Test all variant types for cards."""
        variants = ["elevated", "outlined", "ghost", "subtle"]
        
        for variant in variants:
            result = map_card_classes(variant, "neutral")
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_all_tones(self):
        """Test all tone types for cards."""
        tones = ["neutral", "primary", "success", "warning", "danger"]
        
        for tone in tones:
            result = map_card_classes("elevated", tone)
            assert isinstance(result, str)
            assert len(result) > 0


class TestMappingConsistency:
    """Test consistency across mapping functions."""
    
    def test_all_functions_return_strings(self):
        """Test all mapping functions return strings."""
        assert isinstance(map_button_classes("elevated", "primary", "md"), str)
        assert isinstance(map_input_classes("outlined", "neutral", "md"), str)
        assert isinstance(map_card_classes("elevated", "primary"), str)
        assert isinstance(map_table_classes("elevated", "neutral", "md", "comfortable"), str)
        assert isinstance(map_badge_classes("subtle", "primary", "sm"), str)
        assert isinstance(map_alert_classes("subtle", "success"), str)
    
    def test_all_functions_return_non_empty(self):
        """Test all mapping functions return non-empty strings."""
        assert len(map_button_classes("elevated", "primary", "md")) > 0
        assert len(map_input_classes("outlined", "neutral", "md")) > 0
        assert len(map_card_classes("elevated", "primary")) > 0
        assert len(map_table_classes("elevated", "neutral", "md", "comfortable")) > 0
        assert len(map_badge_classes("subtle", "primary", "sm")) > 0
        assert len(map_alert_classes("subtle", "success")) > 0
    
    def test_consistent_variant_behavior(self):
        """Test variant type behaves consistently across functions."""
        # Elevated should add shadows or solid backgrounds
        button = map_button_classes("elevated", "primary", "md")
        card = map_card_classes("elevated", "primary")
        
        assert "shadow" in button or "shadow" in card
        
        # Outlined should add borders
        button_outlined = map_button_classes("outlined", "primary", "md")
        card_outlined = map_card_classes("outlined", "primary")
        
        assert "border" in button_outlined or "border" in card_outlined
    
    def test_consistent_tone_colors(self):
        """Test tone colors are consistent across functions."""
        # Primary = blue
        button_primary = map_button_classes("elevated", "primary", "md")
        input_primary = map_input_classes("outlined", "primary", "md")
        
        assert "blue" in button_primary or "blue" in input_primary
        
        # Success = green
        button_success = map_button_classes("elevated", "success", "md")
        input_success = map_input_classes("outlined", "success", "md")
        
        assert "green" in button_success or "green" in input_success
    
    def test_consistent_size_scaling(self):
        """Test size scaling is consistent."""
        # Large should have more padding than medium
        button_md = map_button_classes("elevated", "primary", "md")
        button_lg = map_button_classes("elevated", "primary", "lg")
        
        assert "py-2" in button_md
        assert "py-3" in button_lg
        
        # XL should have most padding
        button_xl = map_button_classes("elevated", "primary", "xl")
        assert "py-3" in button_xl or "py-4" in button_xl


class TestNoneHandling:
    """Test handling of None values in mapping functions."""
    
    def test_button_with_none_values(self):
        """Test button mapping with None values."""
        # Functions should handle None gracefully
        result = map_button_classes(None, "primary", "md")
        assert isinstance(result, str)
    
    def test_input_with_none_values(self):
        """Test input mapping with None values."""
        result = map_input_classes("outlined", None, "md")
        assert isinstance(result, str)
    
    def test_card_with_none_values(self):
        """Test card mapping with None values."""
        result = map_card_classes(None, "primary")
        assert isinstance(result, str)
    
    def test_table_with_none_values(self):
        """Test table mapping with None values."""
        result = map_table_classes("elevated", "neutral", "md", None)
        assert isinstance(result, str)
    
    def test_badge_with_none_values(self):
        """Test badge mapping with None values."""
        result = map_badge_classes(None, "primary", "sm")
        assert isinstance(result, str)
    
    def test_alert_with_none_values(self):
        """Test alert mapping with None values."""
        result = map_alert_classes(None, "success")
        assert isinstance(result, str)


class TestEdgeCases:
    """Test edge cases in mapping functions."""
    
    def test_invalid_variant_fallback(self):
        """Test invalid variant falls back gracefully."""
        result = map_button_classes("invalid", "primary", "md")
        assert isinstance(result, str)
        # Should fall back to default variant
    
    def test_invalid_tone_fallback(self):
        """Test invalid tone falls back gracefully."""
        result = map_button_classes("elevated", "invalid", "md")
        assert isinstance(result, str)
        # Should fall back to neutral or default
    
    def test_invalid_size_fallback(self):
        """Test invalid size falls back gracefully."""
        result = map_button_classes("elevated", "primary", "invalid")
        assert isinstance(result, str)
        # Should fall back to medium or default
    
    def test_extreme_combinations(self):
        """Test unusual but valid combinations."""
        # Ghost + danger is uncommon but valid
        result = map_button_classes("ghost", "danger", "xl")
        assert isinstance(result, str)
        assert "transparent" in result
        assert "red" in result or "danger" in result
    
    def test_all_tokens_none(self):
        """Test when all tokens are None."""
        result = map_button_classes(None, None, None)
        assert isinstance(result, str)
        # Should return sensible defaults


class TestTailwindClassValidity:
    """Test that generated classes are valid Tailwind CSS."""
    
    def test_classes_are_space_separated(self):
        """Test classes are space-separated strings."""
        result = map_button_classes("elevated", "primary", "md")
        classes = result.split()
        assert len(classes) > 1
    
    def test_no_duplicate_classes_button(self):
        """Test no excessive duplicate classes in button output."""
        result = map_button_classes("elevated", "primary", "md")
        classes = result.split()
        # Allow some duplicates for specificity, but not excessive
        assert len(classes) == len(set(classes)) or len(classes) - len(set(classes)) < 3
    
    def test_no_duplicate_classes_input(self):
        """Test no excessive duplicate classes in input output."""
        result = map_input_classes("outlined", "neutral", "lg")
        classes = result.split()
        assert len(classes) == len(set(classes)) or len(classes) - len(set(classes)) < 3
    
    def test_classes_follow_tailwind_naming(self):
        """Test classes follow Tailwind naming conventions."""
        result = map_button_classes("outlined", "success", "lg")
        
        # Should have Tailwind-style classes
        assert any(c.startswith("border") for c in result.split())
        assert any(c.startswith("text-") for c in result.split())
        assert any(c.startswith("p") for c in result.split())  # padding classes
    
    def test_hover_states_included(self):
        """Test hover states are included where appropriate."""
        button = map_button_classes("elevated", "primary", "md")
        assert "hover:" in button
        
        input_field = map_input_classes("outlined", "primary", "md")
        assert "focus:" in input_field
