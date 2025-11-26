"""
Design tokens and variants for namel3ss UI components.

This module defines the core visual design tokens that can be applied across
UI components, providing a consistent, production-ready design system.

Design tokens are first-class language concepts that flow through:
DSL → AST → IR → Codegen → React UI

These tokens are UI-library-agnostic at the AST/IR level and map to concrete
design system implementations (Tailwind/shadcn) during codegen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal


# =============================================================================
# CORE DESIGN TOKEN ENUMS
# =============================================================================

class VariantType(str, Enum):
    """
    Visual style variant for components.
    
    Affects background, borders, shadows, and hover/focus/active states.
    Applicable to: buttons, inputs, cards, alerts, badges, etc.
    """
    ELEVATED = "elevated"  # Raised surface with shadow
    OUTLINED = "outlined"  # Border emphasis, transparent background
    GHOST = "ghost"        # Minimal visual weight, transparent
    SUBTLE = "subtle"      # Soft background, minimal borders


class ToneType(str, Enum):
    """
    Semantic color tone for components.
    
    Controls color palette (background, border, text, icon) and related states.
    Applicable to: buttons, alerts, badges, progress indicators, etc.
    """
    NEUTRAL = "neutral"    # Default, non-semantic color
    PRIMARY = "primary"    # Primary brand color
    SUCCESS = "success"    # Success/positive feedback
    WARNING = "warning"    # Warning/caution
    DANGER = "danger"      # Error/destructive actions


class DensityType(str, Enum):
    """
    Spacing density for components and layouts.
    
    Influences vertical padding, row height, and form field spacing.
    Applicable to: rows, lists, tables, forms, data grids, etc.
    """
    COMFORTABLE = "comfortable"  # Standard spacing for readability
    COMPACT = "compact"          # Reduced spacing for information density


class SizeType(str, Enum):
    """
    Size scale for components.
    
    Affects font size, padding, height, and possibly icon sizing.
    Applicable to: buttons, inputs, cards, badges, icons, etc.
    """
    XS = "xs"   # Extra small
    SM = "sm"   # Small
    MD = "md"   # Medium (default)
    LG = "lg"   # Large
    XL = "xl"   # Extra large


class ThemeType(str, Enum):
    """
    Application-level theme mode.
    
    Controls the overall color scheme and appearance of the app.
    Applicable at: app/page level
    """
    LIGHT = "light"      # Light theme
    DARK = "dark"        # Dark theme
    SYSTEM = "system"    # Follow system preference


class ColorSchemeType(str, Enum):
    """
    Brand color scheme for the application.
    
    Defines the primary color palette used throughout the app.
    Applicable at: app/page level
    """
    BLUE = "blue"
    GREEN = "green"
    VIOLET = "violet"
    ROSE = "rose"
    ORANGE = "orange"
    TEAL = "teal"
    INDIGO = "indigo"
    SLATE = "slate"


# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_VARIANT: VariantType = VariantType.ELEVATED
DEFAULT_TONE: ToneType = ToneType.NEUTRAL
DEFAULT_DENSITY: DensityType = DensityType.COMFORTABLE
DEFAULT_SIZE: SizeType = SizeType.MD
DEFAULT_THEME: ThemeType = ThemeType.SYSTEM
DEFAULT_COLOR_SCHEME: ColorSchemeType = ColorSchemeType.BLUE


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_variant(value: str) -> VariantType:
    """
    Validate and convert a string to VariantType.
    
    Raises:
        ValueError: If value is not a valid variant
    """
    try:
        return VariantType(value)
    except ValueError:
        valid = ", ".join(v.value for v in VariantType)
        raise ValueError(
            f"Invalid variant '{value}'. Must be one of: {valid}"
        )


def validate_tone(value: str) -> ToneType:
    """
    Validate and convert a string to ToneType.
    
    Raises:
        ValueError: If value is not a valid tone
    """
    try:
        return ToneType(value)
    except ValueError:
        valid = ", ".join(t.value for t in ToneType)
        raise ValueError(
            f"Invalid tone '{value}'. Must be one of: {valid}"
        )


def validate_density(value: str) -> DensityType:
    """
    Validate and convert a string to DensityType.
    
    Raises:
        ValueError: If value is not a valid density
    """
    try:
        return DensityType(value)
    except ValueError:
        valid = ", ".join(d.value for d in DensityType)
        raise ValueError(
            f"Invalid density '{value}'. Must be one of: {valid}"
        )


def validate_size(value: str) -> SizeType:
    """
    Validate and convert a string to SizeType.
    
    Raises:
        ValueError: If value is not a valid size
    """
    try:
        return SizeType(value)
    except ValueError:
        valid = ", ".join(s.value for s in SizeType)
        raise ValueError(
            f"Invalid size '{value}'. Must be one of: {valid}"
        )


def validate_theme(value: str) -> ThemeType:
    """
    Validate and convert a string to ThemeType.
    
    Raises:
        ValueError: If value is not a valid theme
    """
    try:
        return ThemeType(value)
    except ValueError:
        valid = ", ".join(t.value for t in ThemeType)
        raise ValueError(
            f"Invalid theme '{value}'. Must be one of: {valid}"
        )


def validate_color_scheme(value: str) -> ColorSchemeType:
    """
    Validate and convert a string to ColorSchemeType.
    
    Raises:
        ValueError: If value is not a valid color scheme
    """
    try:
        return ColorSchemeType(value)
    except ValueError:
        valid = ", ".join(c.value for c in ColorSchemeType)
        raise ValueError(
            f"Invalid color_scheme '{value}'. Must be one of: {valid}"
        )


# =============================================================================
# DESIGN TOKEN MIXINS FOR AST NODES
# =============================================================================

@dataclass
class ComponentDesignTokens:
    """
    Design tokens applicable to individual components.
    
    This can be mixed into AST component nodes that support visual customization.
    """
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    density: Optional[DensityType] = None
    size: Optional[SizeType] = None


@dataclass
class AppLevelDesignTokens:
    """
    Design tokens applicable at application/page level.
    
    These tokens affect the entire app or page and cascade down to components.
    """
    theme: Optional[ThemeType] = None
    color_scheme: Optional[ColorSchemeType] = None


# =============================================================================
# TYPE ALIASES FOR CONVENIENCE
# =============================================================================

# Type aliases for use in AST nodes and type hints
VariantLiteral = Literal["elevated", "outlined", "ghost", "subtle"]
ToneLiteral = Literal["neutral", "primary", "success", "warning", "danger"]
DensityLiteral = Literal["comfortable", "compact"]
SizeLiteral = Literal["xs", "sm", "md", "lg", "xl"]
ThemeLiteral = Literal["light", "dark", "system"]
ColorSchemeLiteral = Literal["blue", "green", "violet", "rose", "orange", "teal", "indigo", "slate"]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "VariantType",
    "ToneType",
    "DensityType",
    "SizeType",
    "ThemeType",
    "ColorSchemeType",
    # Defaults
    "DEFAULT_VARIANT",
    "DEFAULT_TONE",
    "DEFAULT_DENSITY",
    "DEFAULT_SIZE",
    "DEFAULT_THEME",
    "DEFAULT_COLOR_SCHEME",
    # Validation
    "validate_variant",
    "validate_tone",
    "validate_density",
    "validate_size",
    "validate_theme",
    "validate_color_scheme",
    # Mixins
    "ComponentDesignTokens",
    "AppLevelDesignTokens",
    # Type Literals
    "VariantLiteral",
    "ToneLiteral",
    "DensityLiteral",
    "SizeLiteral",
    "ThemeLiteral",
    "ColorSchemeLiteral",
]
