"""
N3 Language Keywords and Grammar Constants.

This module defines the complete keyword hierarchy for the N3 language,
providing a single source of truth for keyword validation and error messages.

**Design Goals:**
- Central registry for all N3 keywords
- Context-aware validation (top-level vs page vs control-flow)
- Typo detection and correction suggestions
- Helpful error messages with keyword lists

**Usage:**
    from namel3ss.lang import (
        TOP_LEVEL_KEYWORDS,
        PAGE_STATEMENT_KEYWORDS,
        suggest_keyword,
        valid_keywords_for_context,
    )
    
    if keyword not in TOP_LEVEL_KEYWORDS:
        suggestion = suggest_keyword(keyword, 'top-level')
        raise error(f"Unknown keyword '{keyword}'. Did you mean '{suggestion}'?")
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set
from dataclasses import dataclass
import difflib


# ============================================================================
# Top-Level Keywords (program level, indent=0)
# ============================================================================

TOP_LEVEL_KEYWORDS: FrozenSet[str] = frozenset({
    # Core program structure
    'module',
    'import',
    'language_version',
    
    # Application
    'app',
    
    # Data structures
    'dataset',
    'frame',
    'model',
    
    # AI/ML
    'ai',  # Used in 'ai model'
    'prompt',
    'connector',
    'memory',
    'training',
    'tuning',
    'experiment',
    
    # Logic programming
    'knowledge',
    'query',
    
    # Evaluation
    'evaluator',
    'metric',
    'guardrail',
    'eval_suite',
    
    # Templates and chains
    'define',  # Used in 'define template', 'define chain'
    
    # Pages
    'page',
    
    # Insights
    'insight',
    
    # Theme
    'theme',
    
    # CRUD
    'enable',  # Used in 'enable crud'
})


# ============================================================================
# Block Keywords (inside pages, control flow, etc.)
# ============================================================================

PAGE_STATEMENT_KEYWORDS: FrozenSet[str] = frozenset({
    'set',        # Variable assignment
    'show',       # Used in 'show text', 'show table', etc.
    'if',         # Conditional
    'elif',       # Else-if
    'else',       # Else
    'for',        # Loop
    'while',      # Loop
    'break',      # Loop control
    'continue',   # Loop control
    'action',     # Action declaration
    'predict',    # ML prediction
})

CONTROL_FLOW_KEYWORDS: FrozenSet[str] = frozenset({
    'if',
    'elif',
    'else',
    'for',
    'while',
    'break',
    'continue',
})


# ============================================================================
# Component Keywords (show text, show table, etc.)
# ============================================================================

COMPONENT_KEYWORDS: FrozenSet[str] = frozenset({
    'text',
    'table',
    'chart',
    'form',
    'button',
    'input',
    'select',
    'checkbox',
    'radio',
    'textarea',
})


# ============================================================================
# Effect Keywords (function purity annotations)
# ============================================================================

EFFECT_KEYWORDS: FrozenSet[str] = frozenset({
    'pure',
    'io',
    'read',
    'write',
    'stateful',
})


# ============================================================================
# Multi-word Keyword Patterns
# ============================================================================

MULTI_WORD_PATTERNS: Dict[str, List[str]] = {
    'ai model': ['ai', 'model'],
    'define template': ['define', 'template'],
    'define chain': ['define', 'chain'],
    'enable crud': ['enable', 'crud'],
    'show text': ['show', 'text'],
    'show table': ['show', 'table'],
    'show chart': ['show', 'chart'],
    'show form': ['show', 'form'],
    'auto refresh': ['auto', 'refresh'],
    'language_version': ['language_version'],
}


# ============================================================================
# Keyword Context Rules
# ============================================================================

@dataclass
class KeywordContext:
    """
    Defines where a keyword is valid.
    
    Attributes:
        keyword: The keyword string
        valid_contexts: Set of contexts where this keyword is allowed
        description: Human-readable description of the keyword's purpose
    """
    keyword: str
    valid_contexts: Set[str]  # e.g., {'top-level', 'page', 'control-flow'}
    description: str


# Complete mapping of keywords to their valid contexts
KEYWORD_CONTEXTS: Dict[str, KeywordContext] = {
    # Top-level program structure
    'module': KeywordContext('module', {'top-level'}, 'Module declaration'),
    'import': KeywordContext('import', {'top-level'}, 'Import statement'),
    'language_version': KeywordContext('language_version', {'top-level'}, 'Language version directive'),
    'app': KeywordContext('app', {'top-level'}, 'Application declaration'),
    
    # Data structures
    'dataset': KeywordContext('dataset', {'top-level'}, 'Dataset definition'),
    'frame': KeywordContext('frame', {'top-level'}, 'Data frame definition'),
    'model': KeywordContext('model', {'top-level'}, 'Model declaration'),
    
    # AI/ML
    'ai': KeywordContext('ai', {'top-level'}, 'AI model prefix (ai model)'),
    'prompt': KeywordContext('prompt', {'top-level'}, 'Prompt template'),
    'connector': KeywordContext('connector', {'top-level'}, 'External connector'),
    'memory': KeywordContext('memory', {'top-level'}, 'Conversational memory'),
    'training': KeywordContext('training', {'top-level'}, 'Training job'),
    'tuning': KeywordContext('tuning', {'top-level'}, 'Hyperparameter tuning'),
    'experiment': KeywordContext('experiment', {'top-level'}, 'ML experiment'),
    
    # Logic programming
    'knowledge': KeywordContext('knowledge', {'top-level'}, 'Knowledge base module'),
    'query': KeywordContext('query', {'top-level'}, 'Logic query definition'),
    
    # Evaluation
    'evaluator': KeywordContext('evaluator', {'top-level'}, 'Model evaluator'),
    'metric': KeywordContext('metric', {'top-level'}, 'Evaluation metric'),
    'guardrail': KeywordContext('guardrail', {'top-level'}, 'Safety guardrail'),
    'eval_suite': KeywordContext('eval_suite', {'top-level'}, 'Evaluation suite'),
    
    # Templates
    'define': KeywordContext('define', {'top-level'}, 'Template or chain definition'),
    
    # Pages
    'page': KeywordContext('page', {'top-level'}, 'Page definition'),
    
    # Insights
    'insight': KeywordContext('insight', {'top-level'}, 'Insight definition'),
    
    # Theme
    'theme': KeywordContext('theme', {'top-level'}, 'Theme configuration'),
    
    # CRUD
    'enable': KeywordContext('enable', {'top-level'}, 'Enable feature (e.g., crud)'),
    
    # Page statements
    'set': KeywordContext('set', {'page', 'control-flow'}, 'Variable assignment'),
    'show': KeywordContext('show', {'page', 'control-flow'}, 'Display component'),
    'action': KeywordContext('action', {'page', 'control-flow'}, 'Action definition'),
    'predict': KeywordContext('predict', {'page', 'control-flow'}, 'ML prediction'),
    
    # Control flow
    'if': KeywordContext('if', {'page', 'control-flow'}, 'Conditional statement'),
    'elif': KeywordContext('elif', {'page', 'control-flow'}, 'Else-if branch'),
    'else': KeywordContext('else', {'page', 'control-flow'}, 'Else branch'),
    'for': KeywordContext('for', {'page', 'control-flow'}, 'Loop statement'),
    'while': KeywordContext('while', {'page', 'control-flow'}, 'While loop'),
    'break': KeywordContext('break', {'page', 'control-flow'}, 'Break loop'),
    'continue': KeywordContext('continue', {'page', 'control-flow'}, 'Continue loop'),
}


# ============================================================================
# Common Typos and Suggestions
# ============================================================================

KEYWORD_TYPOS: Dict[str, str] = {
    # Common misspellings
    'modle': 'model',
    'modl': 'model',
    'mdoel': 'model',
    'imoprt': 'import',
    'improt': 'import',
    'ipmort': 'import',
    'pge': 'page',
    'pgae': 'page',
    'datasset': 'dataset',
    'datset': 'dataset',
    'frme': 'frame',
    'fram': 'frame',
    'insihgt': 'insight',
    'insigt': 'insight',
    'insite': 'insight',
    'promtp': 'prompt',
    'promt': 'prompt',
    'conector': 'connector',
    'conecter': 'connector',
    'connetor': 'connector',
    'tabel': 'table',
    'tabl': 'table',
    'cahrt': 'chart',
    'chrt': 'chart',
    'formt': 'format',
    'formm': 'format',
    'memroy': 'memory',
    'memoy': 'memory',
    'experient': 'experiment',
    'experimnet': 'experiment',
    'trainig': 'training',
    'taining': 'training',
    'tunning': 'tuning',
    'tuninig': 'tuning',
    'guarrail': 'guardrail',
    'gaurdail': 'guardrail',
    'evaluater': 'evaluator',
    'evalator': 'evaluator',
    'knoledge': 'knowledge',
    'knowlege': 'knowledge',
    
    # Case variations (N3 is case-sensitive for keywords)
    'Module': 'module',
    'Import': 'import',
    'App': 'app',
    'Page': 'page',
    'Model': 'model',
    'Dataset': 'dataset',
    'Frame': 'frame',
    'Insight': 'insight',
    'Prompt': 'prompt',
    'Connector': 'connector',
    'Memory': 'memory',
    'Training': 'training',
    'Tuning': 'tuning',
    'Experiment': 'experiment',
    'Knowledge': 'knowledge',
    'Query': 'query',
    'Evaluator': 'evaluator',
    'Metric': 'metric',
    'Guardrail': 'guardrail',
    
    # Common command confusions from other languages
    'def': 'define',
    'define_template': 'define template',
    'define_chain': 'define chain',
    'enable_crud': 'enable crud',
    'show_text': 'show text',
    'show_table': 'show table',
    'ai_model': 'ai model',
}


# ============================================================================
# Keyword Validation Helpers
# ============================================================================

def suggest_keyword(unknown: str, context: str = 'top-level') -> Optional[str]:
    """
    Suggest the most likely correct keyword for an unknown token.
    
    Uses a combination of:
    1. Direct typo mapping lookup
    2. Fuzzy string matching (Levenshtein distance)
    3. Context-aware filtering
    
    Args:
        unknown: The unknown keyword string
        context: Where the keyword appeared ('top-level', 'page', 'control-flow', 'any')
        
    Returns:
        Suggested keyword or None if no good match found
        
    Examples:
        >>> suggest_keyword('modle', 'top-level')
        'model'
        
        >>> suggest_keyword('breik', 'page')
        'break'
        
        >>> suggest_keyword('xyz123', 'top-level')
        None
    """
    # Check direct typo mapping first (exact matches)
    if unknown in KEYWORD_TYPOS:
        return KEYWORD_TYPOS[unknown]
    
    # Get candidate keywords for the context
    candidates = _get_candidate_keywords(context)
    
    # Use fuzzy matching to find close matches
    close_matches = difflib.get_close_matches(unknown, candidates, n=1, cutoff=0.6)
    
    if close_matches:
        return close_matches[0]
    
    # If no match in specific context, try all keywords
    if context != 'any':
        all_keywords = _get_candidate_keywords('any')
        close_matches = difflib.get_close_matches(unknown, all_keywords, n=1, cutoff=0.7)
        if close_matches:
            return close_matches[0]
    
    return None


def _get_candidate_keywords(context: str) -> List[str]:
    """
    Get list of candidate keywords for fuzzy matching.
    
    Args:
        context: The parsing context
        
    Returns:
        List of keywords valid in this context
    """
    if context == 'any':
        # All keywords
        return list(KEYWORD_CONTEXTS.keys())
    
    # Filter by context
    candidates = []
    for kw, ctx in KEYWORD_CONTEXTS.items():
        if context in ctx.valid_contexts:
            candidates.append(kw)
    
    return candidates


def valid_keywords_for_context(context: str) -> List[str]:
    """
    Get sorted list of valid keywords for a given context.
    
    Args:
        context: The parsing context ('top-level', 'page', 'control-flow', 'any')
        
    Returns:
        Sorted list of valid keywords
        
    Examples:
        >>> keywords = valid_keywords_for_context('top-level')
        >>> 'app' in keywords
        True
        >>> 'set' in keywords
        False
        
        >>> keywords = valid_keywords_for_context('page')
        >>> 'set' in keywords
        True
        >>> 'show' in keywords
        True
    """
    if context == 'any':
        return sorted(KEYWORD_CONTEXTS.keys())
    
    keywords = []
    for kw, ctx in KEYWORD_CONTEXTS.items():
        if context in ctx.valid_contexts:
            keywords.append(kw)
    
    return sorted(keywords)


def is_valid_keyword(keyword: str, context: str = 'any') -> bool:
    """
    Check if keyword is valid in given context.
    
    Args:
        keyword: The keyword to validate
        context: The parsing context, or 'any' to check all contexts
        
    Returns:
        True if keyword is valid in context
        
    Examples:
        >>> is_valid_keyword('app', 'top-level')
        True
        
        >>> is_valid_keyword('set', 'top-level')
        False
        
        >>> is_valid_keyword('set', 'page')
        True
        
        >>> is_valid_keyword('unknown_keyword', 'any')
        False
    """
    if context == 'any':
        return keyword in KEYWORD_CONTEXTS
    
    ctx = KEYWORD_CONTEXTS.get(keyword)
    return ctx is not None and context in ctx.valid_contexts


def get_keyword_description(keyword: str) -> Optional[str]:
    """
    Get human-readable description of a keyword.
    
    Args:
        keyword: The keyword to describe
        
    Returns:
        Description string or None if keyword not found
        
    Examples:
        >>> get_keyword_description('app')
        'Application declaration'
        
        >>> get_keyword_description('set')
        'Variable assignment'
    """
    ctx = KEYWORD_CONTEXTS.get(keyword)
    return ctx.description if ctx else None


def format_keyword_list(keywords: List[str], max_items: int = 10) -> str:
    """
    Format a list of keywords for display in error messages.
    
    Args:
        keywords: List of keywords to format
        max_items: Maximum number to show before truncating
        
    Returns:
        Formatted string like "app, page, model, ... (15 total)"
        
    Examples:
        >>> format_keyword_list(['app', 'page', 'model'])
        'app, page, model'
        
        >>> format_keyword_list(list(TOP_LEVEL_KEYWORDS), max_items=5)
        'app, connector, crud, dataset, define, ... (23 total)'
    """
    if len(keywords) <= max_items:
        return ', '.join(keywords)
    
    shown = ', '.join(keywords[:max_items])
    total = len(keywords)
    return f"{shown}, ... ({total} total)"


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Keyword sets
    'TOP_LEVEL_KEYWORDS',
    'PAGE_STATEMENT_KEYWORDS',
    'CONTROL_FLOW_KEYWORDS',
    'COMPONENT_KEYWORDS',
    'EFFECT_KEYWORDS',
    'MULTI_WORD_PATTERNS',
    
    # Data structures
    'KeywordContext',
    'KEYWORD_CONTEXTS',
    'KEYWORD_TYPOS',
    
    # Helper functions
    'suggest_keyword',
    'valid_keywords_for_context',
    'is_valid_keyword',
    'get_keyword_description',
    'format_keyword_list',
]
