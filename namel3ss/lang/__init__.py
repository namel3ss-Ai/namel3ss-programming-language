"""Namel3ss language specification helpers."""

from .version import LANGUAGE_VERSION, SUPPORTED_LANGUAGE_VERSIONS
from .keywords import (
    TOP_LEVEL_KEYWORDS,
    PAGE_STATEMENT_KEYWORDS,
    CONTROL_FLOW_KEYWORDS,
    COMPONENT_KEYWORDS,
    EFFECT_KEYWORDS,
    MULTI_WORD_PATTERNS,
    KeywordContext,
    KEYWORD_CONTEXTS,
    KEYWORD_TYPOS,
    suggest_keyword,
    valid_keywords_for_context,
    is_valid_keyword,
    get_keyword_description,
    format_keyword_list,
)

__all__ = [
    # Version info
    "LANGUAGE_VERSION",
    "SUPPORTED_LANGUAGE_VERSIONS",
    # Keyword sets
    "TOP_LEVEL_KEYWORDS",
    "PAGE_STATEMENT_KEYWORDS",
    "CONTROL_FLOW_KEYWORDS",
    "COMPONENT_KEYWORDS",
    "EFFECT_KEYWORDS",
    "MULTI_WORD_PATTERNS",
    # Keyword data structures
    "KeywordContext",
    "KEYWORD_CONTEXTS",
    "KEYWORD_TYPOS",
    # Keyword validation helpers
    "suggest_keyword",
    "valid_keywords_for_context",
    "is_valid_keyword",
    "get_keyword_description",
    "format_keyword_list",
]
