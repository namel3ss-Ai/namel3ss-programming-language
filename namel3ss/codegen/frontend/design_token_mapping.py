"""
Central design token mapping layer for namel3ss.

This module provides the canonical mapping from design tokens (variant, tone,
density, size, theme, color_scheme) to concrete Tailwind CSS classes and shadcn/ui
component props.

All codegen modules should use this mapping layer to ensure consistent visual
behavior across all generated components.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class VariantType(str, Enum):
    """Visual style variant"""
    ELEVATED = "elevated"
    OUTLINED = "outlined"
    GHOST = "ghost"
    SUBTLE = "subtle"


class ToneType(str, Enum):
    """Semantic color tone"""
    NEUTRAL = "neutral"
    PRIMARY = "primary"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"


class DensityType(str, Enum):
    """Spacing density"""
    COMFORTABLE = "comfortable"
    COMPACT = "compact"


class SizeType(str, Enum):
    """Component size"""
    XS = "xs"
    SM = "sm"
    MD = "md"
    LG = "lg"
    XL = "xl"


# =============================================================================
# BUTTON MAPPINGS
# =============================================================================

BUTTON_VARIANT_CLASSES: Dict[str, str] = {
    "elevated": "shadow-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 active:bg-gray-100 dark:active:bg-gray-600",
    "outlined": "bg-transparent border-2 hover:bg-gray-50 dark:hover:bg-gray-800",
    "ghost": "bg-transparent border-0 hover:bg-gray-100 dark:hover:bg-gray-800",
    "subtle": "bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700",
}

BUTTON_TONE_CLASSES: Dict[str, Dict[str, str]] = {
    "neutral": {
        "text": "text-gray-700 dark:text-gray-200",
        "border": "border-gray-300 dark:border-gray-600",
        "hover": "hover:border-gray-400 dark:hover:border-gray-500",
    },
    "primary": {
        "text": "text-blue-600 dark:text-blue-400",
        "border": "border-blue-500 dark:border-blue-600",
        "hover": "hover:border-blue-600 dark:hover:border-blue-500",
        "bg_solid": "bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white",
    },
    "success": {
        "text": "text-green-600 dark:text-green-400",
        "border": "border-green-500 dark:border-green-600",
        "hover": "hover:border-green-600 dark:hover:border-green-500",
        "bg_solid": "bg-green-600 hover:bg-green-700 dark:bg-green-500 dark:hover:bg-green-600 text-white",
    },
    "warning": {
        "text": "text-orange-600 dark:text-orange-400",
        "border": "border-orange-500 dark:border-orange-600",
        "hover": "hover:border-orange-600 dark:hover:border-orange-500",
        "bg_solid": "bg-orange-600 hover:bg-orange-700 dark:bg-orange-500 dark:hover:bg-orange-600 text-white",
    },
    "danger": {
        "text": "text-red-600 dark:text-red-400",
        "border": "border-red-500 dark:border-red-600",
        "hover": "hover:border-red-600 dark:hover:border-red-500",
        "bg_solid": "bg-red-600 hover:bg-red-700 dark:bg-red-500 dark:hover:bg-red-600 text-white",
    },
}

BUTTON_SIZE_CLASSES: Dict[str, str] = {
    "xs": "px-2 py-1 text-xs",
    "sm": "px-3 py-1.5 text-sm",
    "md": "px-4 py-2 text-base",
    "lg": "px-6 py-3 text-lg",
    "xl": "px-8 py-4 text-xl",
}


# =============================================================================
# INPUT MAPPINGS
# =============================================================================

INPUT_VARIANT_CLASSES: Dict[str, str] = {
    "elevated": "shadow-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-offset-2",
    "outlined": "bg-transparent border-2 focus:ring-2",
    "ghost": "bg-transparent border-0 border-b-2 rounded-none focus:ring-0 focus:border-b-4",
    "subtle": "bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 focus:ring-2",
}

INPUT_TONE_CLASSES: Dict[str, str] = {
    "neutral": "border-gray-300 dark:border-gray-600 focus:border-gray-500 dark:focus:border-gray-400 focus:ring-gray-500 dark:focus:ring-gray-400",
    "primary": "border-blue-300 dark:border-blue-600 focus:border-blue-500 dark:focus:border-blue-400 focus:ring-blue-500 dark:focus:ring-blue-400",
    "success": "border-green-300 dark:border-green-600 focus:border-green-500 dark:focus:border-green-400 focus:ring-green-500 dark:focus:ring-green-400",
    "warning": "border-orange-300 dark:border-orange-600 focus:border-orange-500 dark:focus:border-orange-400 focus:ring-orange-500 dark:focus:ring-orange-400",
    "danger": "border-red-300 dark:border-red-600 focus:border-red-500 dark:focus:border-red-400 focus:ring-red-500 dark:focus:ring-red-400",
}

INPUT_SIZE_CLASSES: Dict[str, str] = {
    "xs": "px-2 py-1 text-xs",
    "sm": "px-3 py-1.5 text-sm",
    "md": "px-4 py-2 text-base",
    "lg": "px-5 py-3 text-lg",
    "xl": "px-6 py-4 text-xl",
}


# =============================================================================
# CARD MAPPINGS
# =============================================================================

CARD_VARIANT_CLASSES: Dict[str, str] = {
    "elevated": "shadow-md bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700",
    "outlined": "bg-white dark:bg-gray-900 border-2 border-gray-300 dark:border-gray-600",
    "ghost": "bg-transparent border-0",
    "subtle": "bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700",
}

CARD_TONE_CLASSES: Dict[str, str] = {
    "neutral": "border-gray-200 dark:border-gray-700",
    "primary": "border-blue-200 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20",
    "success": "border-green-200 dark:border-green-700 bg-green-50 dark:bg-green-900/20",
    "warning": "border-orange-200 dark:border-orange-700 bg-orange-50 dark:bg-orange-900/20",
    "danger": "border-red-200 dark:border-red-700 bg-red-50 dark:bg-red-900/20",
}


# =============================================================================
# BADGE MAPPINGS
# =============================================================================

BADGE_VARIANT_CLASSES: Dict[str, str] = {
    "elevated": "shadow-sm",
    "outlined": "border-2 bg-transparent",
    "ghost": "border-0 bg-transparent",
    "subtle": "border",
}

BADGE_TONE_CLASSES: Dict[str, str] = {
    "neutral": "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-600",
    "primary": "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-200 border-blue-300 dark:border-blue-600",
    "success": "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-200 border-green-300 dark:border-green-600",
    "warning": "bg-orange-100 dark:bg-orange-900 text-orange-700 dark:text-orange-200 border-orange-300 dark:border-orange-600",
    "danger": "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200 border-red-300 dark:border-red-600",
}

BADGE_SIZE_CLASSES: Dict[str, str] = {
    "xs": "px-1.5 py-0.5 text-xs",
    "sm": "px-2 py-0.5 text-sm",
    "md": "px-2.5 py-1 text-sm",
    "lg": "px-3 py-1 text-base",
    "xl": "px-4 py-1.5 text-base",
}


# =============================================================================
# ALERT/TOAST MAPPINGS
# =============================================================================

ALERT_VARIANT_CLASSES: Dict[str, str] = {
    "elevated": "shadow-md border",
    "outlined": "border-2 bg-transparent",
    "ghost": "border-0",
    "subtle": "border",
}

ALERT_TONE_CLASSES: Dict[str, str] = {
    "neutral": "bg-gray-50 dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-800 dark:text-gray-100",
    "primary": "bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700 text-blue-800 dark:text-blue-100",
    "success": "bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700 text-green-800 dark:text-green-100",
    "warning": "bg-orange-50 dark:bg-orange-900/20 border-orange-300 dark:border-orange-700 text-orange-800 dark:text-orange-100",
    "danger": "bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700 text-red-800 dark:text-red-100",
}


# =============================================================================
# DENSITY MAPPINGS
# =============================================================================

DENSITY_SPACING_CLASSES: Dict[str, str] = {
    "comfortable": "gap-4 p-4",
    "compact": "gap-2 p-2",
}

DENSITY_ROW_HEIGHT_CLASSES: Dict[str, str] = {
    "comfortable": "h-12",
    "compact": "h-8",
}


# =============================================================================
# MAIN MAPPING FUNCTIONS
# =============================================================================

def map_button_classes(
    variant: Optional[str] = None,
    tone: Optional[str] = None,
    size: Optional[str] = None,
) -> str:
    """
    Map design tokens to Tailwind classes for buttons.
    
    Args:
        variant: Visual variant (elevated/outlined/ghost/subtle)
        tone: Semantic tone (neutral/primary/success/warning/danger)
        size: Size scale (xs/sm/md/lg/xl)
    
    Returns:
        Space-separated Tailwind class string
    """
    classes: List[str] = ["inline-flex items-center justify-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none"]
    
    # Base variant styles
    variant_key = variant or "elevated"
    classes.append(BUTTON_VARIANT_CLASSES.get(variant_key, BUTTON_VARIANT_CLASSES["elevated"]))
    
    # Tone-based colors
    tone_key = tone or "neutral"
    tone_config = BUTTON_TONE_CLASSES.get(tone_key, BUTTON_TONE_CLASSES["neutral"])
    
    # For elevated buttons with non-neutral tones, use solid background
    if variant_key == "elevated" and tone_key != "neutral":
        classes.append(tone_config.get("bg_solid", tone_config["text"]))
    else:
        classes.append(tone_config["text"])
        classes.append(tone_config.get("border", ""))
        classes.append(tone_config.get("hover", ""))
    
    # Size
    size_key = size or "md"
    classes.append(BUTTON_SIZE_CLASSES.get(size_key, BUTTON_SIZE_CLASSES["md"]))
    
    return " ".join(filter(None, classes))


def map_input_classes(
    variant: Optional[str] = None,
    tone: Optional[str] = None,
    size: Optional[str] = None,
) -> str:
    """
    Map design tokens to Tailwind classes for inputs.
    
    Args:
        variant: Visual variant (elevated/outlined/ghost/subtle)
        tone: Semantic tone (neutral/primary/success/warning/danger)
        size: Size scale (xs/sm/md/lg/xl)
    
    Returns:
        Space-separated Tailwind class string
    """
    classes: List[str] = ["w-full rounded-md transition-colors text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"]
    
    # Variant
    variant_key = variant or "elevated"
    classes.append(INPUT_VARIANT_CLASSES.get(variant_key, INPUT_VARIANT_CLASSES["elevated"]))
    
    # Tone
    tone_key = tone or "neutral"
    classes.append(INPUT_TONE_CLASSES.get(tone_key, INPUT_TONE_CLASSES["neutral"]))
    
    # Size
    size_key = size or "md"
    classes.append(INPUT_SIZE_CLASSES.get(size_key, INPUT_SIZE_CLASSES["md"]))
    
    return " ".join(filter(None, classes))


def map_card_classes(
    variant: Optional[str] = None,
    tone: Optional[str] = None,
) -> str:
    """
    Map design tokens to Tailwind classes for cards.
    
    Args:
        variant: Visual variant (elevated/outlined/ghost/subtle)
        tone: Semantic tone (neutral/primary/success/warning/danger)
    
    Returns:
        Space-separated Tailwind class string
    """
    classes: List[str] = ["rounded-lg overflow-hidden"]
    
    # Variant
    variant_key = variant or "elevated"
    classes.append(CARD_VARIANT_CLASSES.get(variant_key, CARD_VARIANT_CLASSES["elevated"]))
    
    # Tone
    tone_key = tone or "neutral"
    if tone_key != "neutral":
        classes.append(CARD_TONE_CLASSES.get(tone_key, ""))
    
    return " ".join(filter(None, classes))


def map_badge_classes(
    variant: Optional[str] = None,
    tone: Optional[str] = None,
    size: Optional[str] = None,
) -> str:
    """
    Map design tokens to Tailwind classes for badges.
    
    Args:
        variant: Visual variant (elevated/outlined/ghost/subtle)
        tone: Semantic tone (neutral/primary/success/warning/danger)
        size: Size scale (xs/sm/md/lg/xl)
    
    Returns:
        Space-separated Tailwind class string
    """
    classes: List[str] = ["inline-flex items-center rounded-full font-medium"]
    
    # Variant
    variant_key = variant or "subtle"
    classes.append(BADGE_VARIANT_CLASSES.get(variant_key, BADGE_VARIANT_CLASSES["subtle"]))
    
    # Tone
    tone_key = tone or "neutral"
    classes.append(BADGE_TONE_CLASSES.get(tone_key, BADGE_TONE_CLASSES["neutral"]))
    
    # Size
    size_key = size or "sm"
    classes.append(BADGE_SIZE_CLASSES.get(size_key, BADGE_SIZE_CLASSES["sm"]))
    
    return " ".join(filter(None, classes))


def map_alert_classes(
    variant: Optional[str] = None,
    tone: Optional[str] = None,
) -> str:
    """
    Map design tokens to Tailwind classes for alerts/toasts.
    
    Args:
        variant: Visual variant (elevated/outlined/ghost/subtle)
        tone: Semantic tone (neutral/primary/success/warning/danger)
    
    Returns:
        Space-separated Tailwind class string
    """
    classes: List[str] = ["rounded-lg p-4"]
    
    # Variant
    variant_key = variant or "subtle"
    classes.append(ALERT_VARIANT_CLASSES.get(variant_key, ALERT_VARIANT_CLASSES["subtle"]))
    
    # Tone
    tone_key = tone or "neutral"
    classes.append(ALERT_TONE_CLASSES.get(tone_key, ALERT_TONE_CLASSES["neutral"]))
    
    return " ".join(filter(None, classes))


def map_density_classes(
    density: Optional[str] = None,
    component_type: str = "default",
) -> str:
    """
    Map density token to spacing classes.
    
    Args:
        density: Density value (comfortable/compact)
        component_type: Type of component (default/row)
    
    Returns:
        Space-separated Tailwind class string
    """
    density_key = density or "comfortable"
    
    if component_type == "row":
        return DENSITY_ROW_HEIGHT_CLASSES.get(density_key, DENSITY_ROW_HEIGHT_CLASSES["comfortable"])
    
    return DENSITY_SPACING_CLASSES.get(density_key, DENSITY_SPACING_CLASSES["comfortable"])


def map_table_classes(
    variant: Optional[str] = None,
    tone: Optional[str] = None,
    size: Optional[str] = None,
    density: Optional[str] = None,
) -> str:
    """
    Map design tokens to Tailwind classes for tables.
    
    Args:
        variant: Visual variant (elevated/outlined/ghost/subtle)
        tone: Semantic tone (neutral/primary/success/warning/danger)
        size: Size scale (xs/sm/md/lg/xl)
        density: Density (comfortable/compact)
    
    Returns:
        Space-separated Tailwind class string
    """
    classes: List[str] = ["w-full border-collapse"]
    
    # Base table styles
    classes.append("text-left")
    
    # Variant
    variant_key = variant or "elevated"
    if variant_key == "elevated":
        classes.append("bg-white dark:bg-gray-800 shadow-sm rounded-lg overflow-hidden")
    elif variant_key == "outlined":
        classes.append("border border-gray-200 dark:border-gray-700 rounded-lg")
    elif variant_key == "ghost":
        classes.append("border-0")
    elif variant_key == "subtle":
        classes.append("bg-gray-50 dark:bg-gray-900 border border-gray-100 dark:border-gray-800")
    
    # Tone (affects header color)
    tone_key = tone or "neutral"
    if tone_key == "neutral":
        classes.append("[&_thead]:bg-gray-100 [&_thead]:dark:bg-gray-700")
    elif tone_key == "primary":
        classes.append("[&_thead]:bg-blue-50 [&_thead]:dark:bg-blue-900/20")
    elif tone_key == "success":
        classes.append("[&_thead]:bg-green-50 [&_thead]:dark:bg-green-900/20")
    elif tone_key == "warning":
        classes.append("[&_thead]:bg-orange-50 [&_thead]:dark:bg-orange-900/20")
    elif tone_key == "danger":
        classes.append("[&_thead]:bg-red-50 [&_thead]:dark:bg-red-900/20")
    
    # Size (text size and padding)
    size_key = size or "md"
    if size_key == "xs":
        classes.append("[&_th]:px-2 [&_th]:py-1 [&_td]:px-2 [&_td]:py-1 [&_th]:text-xs [&_td]:text-xs")
    elif size_key == "sm":
        classes.append("[&_th]:px-3 [&_th]:py-1.5 [&_td]:px-3 [&_td]:py-1.5 [&_th]:text-sm [&_td]:text-sm")
    elif size_key == "md":
        classes.append("[&_th]:px-4 [&_th]:py-2 [&_td]:px-4 [&_td]:py-2 [&_th]:text-base [&_td]:text-base")
    elif size_key == "lg":
        classes.append("[&_th]:px-6 [&_th]:py-3 [&_td]:px-6 [&_td]:py-3 [&_th]:text-lg [&_td]:text-lg")
    elif size_key == "xl":
        classes.append("[&_th]:px-8 [&_th]:py-4 [&_td]:px-8 [&_td]:py-4 [&_th]:text-xl [&_td]:text-xl")
    
    # Density (row height)
    density_key = density or "comfortable"
    if density_key == "compact":
        classes.append("[&_tbody_tr]:h-8")
    else:
        classes.append("[&_tbody_tr]:h-12")
    
    # Row styling
    classes.append("[&_tbody_tr]:border-b [&_tbody_tr]:border-gray-200 [&_tbody_tr]:dark:border-gray-700")
    classes.append("[&_tbody_tr:hover]:bg-gray-50 [&_tbody_tr:hover]:dark:bg-gray-800/50")
    
    return " ".join(filter(None, classes))


def get_theme_class_name(theme: Optional[str] = None) -> str:
    """
    Get the HTML class name for theme mode.
    
    Args:
        theme: Theme mode (light/dark/system)
    
    Returns:
        Class name to apply to root element
    """
    theme_key = theme or "system"
    
    if theme_key == "dark":
        return "dark"
    elif theme_key == "light":
        return ""  # No class for light mode (default)
    else:
        # System theme - use JS to detect and apply
        return "system-theme"


def get_color_scheme_css_var(color_scheme: Optional[str] = None) -> Dict[str, str]:
    """
    Get CSS custom properties for color scheme.
    
    Args:
        color_scheme: Brand color (blue/green/violet/rose/etc.)
    
    Returns:
        Dict of CSS variable names to values
    """
    scheme = color_scheme or "blue"
    
    # Map to Tailwind color scales
    color_maps = {
        "blue": {"primary": "#3b82f6", "primary-dark": "#2563eb"},
        "green": {"primary": "#10b981", "primary-dark": "#059669"},
        "violet": {"primary": "#8b5cf6", "primary-dark": "#7c3aed"},
        "rose": {"primary": "#f43f5e", "primary-dark": "#e11d48"},
        "orange": {"primary": "#f97316", "primary-dark": "#ea580c"},
        "teal": {"primary": "#14b8a6", "primary-dark": "#0d9488"},
        "indigo": {"primary": "#6366f1", "primary-dark": "#4f46e5"},
        "slate": {"primary": "#64748b", "primary-dark": "#475569"},
    }
    
    return color_maps.get(scheme, color_maps["blue"])


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Mapping functions
    "map_button_classes",
    "map_input_classes",
    "map_card_classes",
    "map_table_classes",
    "map_badge_classes",
    "map_alert_classes",
    "map_density_classes",
    "get_theme_class_name",
    "get_color_scheme_css_var",
    # Constants (for reference)
    "BUTTON_VARIANT_CLASSES",
    "BUTTON_TONE_CLASSES",
    "BUTTON_SIZE_CLASSES",
    "INPUT_VARIANT_CLASSES",
    "INPUT_TONE_CLASSES",
    "INPUT_SIZE_CLASSES",
    "CARD_VARIANT_CLASSES",
    "BADGE_VARIANT_CLASSES",
    "ALERT_VARIANT_CLASSES",
    "DENSITY_SPACING_CLASSES",
]
