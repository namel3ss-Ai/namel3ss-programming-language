# Parser Infrastructure Refactoring - Design Document

**Status:** Design Phase  
**Author:** Phase 4 - Production-Ready Parser Infrastructure  
**Date:** 2024  

---

## 1. Executive Summary

This document specifies a **world-class, production-ready** refactoring of the N3 language parser infrastructure focusing on:

1. **Consistent indentation model** - Centralized, robust indentation handling with clear error messages
2. **Keyword hierarchy** - Central registry of all N3 keywords with context-aware validation
3. **Enhanced error UX** - Friendly, high-signal error messages with actionable hints

**Critical Requirements:**
- Production-ready (no shortcuts, no demos)
- Backwards-compatible where reasonable
- Well-tested with comprehensive unit and integration tests
- Clear, maintainable code with proper documentation

---

## 2. Current State Analysis

### 2.1 Parser Architecture

**Base Parser:** `namel3ss/parser/base.py` (858 lines)
- `ParserBase` class with shared utilities
- Basic indentation: `_indent(line)` returns space count
- Error creation: `_error()` generates N3SyntaxError
- Coercion helpers: `_parse_bool()`, `_coerce_scalar()`, `_coerce_int()`

**Specialized Parsers:** 18 modules inheriting from ParserBase
- `program.py` - Top-level program structure, imports, module declarations
- `pages.py` - Page definitions with components
- `ai.py` - Connectors, templates, chains (1600 lines)
- `control_flow.py` - if/for/while blocks
- `models.py` - ML model declarations
- `logic.py` - Logic programming constructs
- `insights.py`, `frames.py`, `datasets.py`, `crud.py`, `experiments.py`, `eval.py`, `components.py`, `actions.py`, `symbolic.py`, `expressions.py`

### 2.2 Current Indentation Handling

**Basic Implementation in ParserBase:**
```python
def _indent(self, line: str) -> int:
    """Compute the indentation level (leading spaces) for *line*."""
    return len(line) - len(line.lstrip(' '))
```

**Limitations:**
- ❌ Only counts spaces, no tab detection
- ❌ No mixed indentation validation (tabs + spaces)
- ❌ No consistency checking within blocks
- ❌ No helpful error messages for indentation issues
- ❌ Cannot distinguish between 2-space and 4-space indentation deterministically

**Ad-hoc Patterns Found:**
```python
# Pattern 1: Basic indent check
indent = self._indent(line)
if indent <= base_indent:
    break

# Pattern 2: Block indent tracking
block_indent: Optional[int] = None
if block_indent is None:
    block_indent = indent
elif indent < block_indent:
    break

# Pattern 3: Expected indent validation (scattered)
if indent != 0:
    raise self._error("Top level statements must not be indented", line_no, line)
```

### 2.3 Current Keyword Handling

**Constants in base.py:**
```python
_EFFECT_KEYWORDS = frozenset({'pure', 'io', 'read', 'write', 'stateful'})
_BOOL_NORMALISATIONS = {...}
_LIKE_TOKEN_MAP = {...}
```

**Ad-hoc Keyword Checks:**
```python
# In program.py (lines 70+)
if stripped.startswith('module '):
    ...
elif stripped.startswith('import '):
    ...
elif stripped.startswith('app '):
    ...
elif stripped.startswith('evaluator '):
    ...
# ... 30+ more keyword checks

# In pages.py
if lowered.startswith('reactive:'):
    ...
elif lowered.startswith('auto refresh'):
    ...
elif lowered.startswith('layout:'):
    ...

# In control_flow.py
if stripped.startswith('set '):
    ...
elif stripped.startswith('if '):
    ...
elif stripped.startswith('for '):
    ...
```

**Limitations:**
- ❌ No central keyword registry
- ❌ Keyword validation scattered across 18+ files
- ❌ No context-aware keyword suggestions
- ❌ No typo detection/correction hints
- ❌ Inconsistent error messages

### 2.4 Current Error Handling

**Basic Error Creation:**
```python
def _error(self, message: str, line_no: Optional[int] = None, 
           line: Optional[str] = None) -> N3SyntaxError:
    path = self.file_path or "<input>"
    hint = None  # Rarely set in practice
    return N3SyntaxError(message=message, path=path, line_number=line_no,
                         code=line, hint=hint)
```

**Current Error Examples:**
```
N3SyntaxError: Top level statements must not be indented
  at line 42 in app.ai

N3SyntaxError: Expected ':' after if condition
  at line 15 in page.ai
```

**Desired Improvements:**
```
N3SyntaxError: Inconsistent indentation detected
  at line 42 in app.ai
  → This line uses 3 spaces, but the block started with 4 spaces at line 40
  Hint: Use consistent indentation throughout each block (either 2 or 4 spaces)

N3SyntaxError: Unknown keyword 'modle'
  at line 10 in app.ai
  → Did you mean 'model'?
  Hint: Valid top-level keywords: app, model, page, dataset, frame, insight...
```

---

## 3. Design Specification

### 3.1 Centralized Indentation System

#### 3.1.1 IndentationInfo Data Structure

**Location:** `namel3ss/parser/base.py`

```python
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class IndentationInfo:
    """
    Detailed information about a line's indentation.
    
    Attributes:
        spaces: Number of leading spaces
        tabs: Number of leading tabs
        mixed: True if line has both tabs and spaces in leading whitespace
        indent_style: Detected indentation style ('spaces', 'tabs', 'mixed', 'none')
        effective_level: Effective indentation level for comparison
    """
    spaces: int
    tabs: int
    mixed: bool
    indent_style: Literal['spaces', 'tabs', 'mixed', 'none']
    effective_level: int
```

#### 3.1.2 Core Indentation Methods

**Location:** `namel3ss/parser/base.py` - extend `ParserBase` class

```python
def _compute_indent_details(self, line: str) -> IndentationInfo:
    """
    Analyze line indentation in detail.
    
    Returns IndentationInfo with spaces, tabs, mixed flag, and style.
    This replaces the simple _indent() method with robust analysis.
    """
    # Implementation: Count leading spaces and tabs separately
    # Detect mixed indentation
    # Determine indent_style
    # Calculate effective_level (spaces + tabs*4 for comparison)

def _expect_indent_greater_than(
    self,
    line: str,
    base_indent: int,
    line_no: int,
    context: str = "block"
) -> IndentationInfo:
    """
    Validate that line is indented more than base_indent.
    
    Args:
        line: The line to check
        base_indent: The parent indentation level
        line_no: Line number for error reporting
        context: Human-readable context (e.g., "if block", "page body")
    
    Returns:
        IndentationInfo for the line
        
    Raises:
        N3SyntaxError: If line is not properly indented with helpful hint
    """
    # Implementation: Check indent > base_indent
    # Generate helpful error message with context if validation fails
    # Include hint about expected indentation

def _validate_block_indent(
    self,
    line: str,
    expected_indent: int,
    line_no: int,
    block_start_line: int,
    context: str = "block"
) -> IndentationInfo:
    """
    Validate that line matches expected block indentation.
    
    Args:
        line: The line to check
        expected_indent: Expected indentation level
        line_no: Current line number
        block_start_line: Line where block started (for error messages)
        context: Human-readable context
        
    Returns:
        IndentationInfo for the line
        
    Raises:
        N3SyntaxError: If indentation doesn't match with helpful hint
    """
    # Implementation: Check indent == expected_indent
    # Generate error showing block start line and expected vs actual
    # Include hint about fixing indentation

def _detect_indentation_issues(self, lines: List[str]) -> Optional[str]:
    """
    Scan lines for common indentation problems.
    
    Detects:
    - Mixed tabs and spaces across file
    - Inconsistent indentation increments (mixing 2-space and 4-space)
    - Lines with trailing whitespace in indentation
    
    Returns:
        Warning message string if issues found, None otherwise
        
    Note: This is called during initialization for early detection
    """
    # Implementation: Scan all lines, detect patterns
    # Build warning message with specific issues found
    # Return None if no issues

def _indent(self, line: str) -> int:
    """
    Backward-compatible indentation level computation.
    
    Returns effective indentation level (spaces count).
    Preserved for compatibility with existing code.
    
    NOTE: New code should use _compute_indent_details() for robust handling.
    """
    return self._compute_indent_details(line).effective_level
```

#### 3.1.3 Indentation Usage Patterns

**Pattern 1: Expecting indented block**
```python
# Old code:
while self.pos < len(self.lines):
    nxt = self._peek()
    indent = self._indent(nxt)
    if indent <= base_indent:
        break
    # ... parse statement

# New code:
while self.pos < len(self.lines):
    nxt = self._peek()
    try:
        info = self._expect_indent_greater_than(nxt, base_indent, self.pos + 1, "page body")
    except N3SyntaxError:
        break  # End of block
    # ... parse statement
```

**Pattern 2: Tracking consistent block indentation**
```python
# Old code:
block_indent: Optional[int] = None
while self.pos < len(self.lines):
    nxt = self._peek()
    indent = self._indent(nxt)
    if block_indent is None:
        block_indent = indent
    elif indent < block_indent:
        break

# New code:
block_indent: Optional[int] = None
block_start_line: Optional[int] = None
while self.pos < len(self.lines):
    nxt = self._peek()
    current_line = self.pos + 1
    
    if block_indent is None:
        info = self._expect_indent_greater_than(nxt, base_indent, current_line, "if body")
        block_indent = info.effective_level
        block_start_line = current_line
    else:
        info = self._validate_block_indent(nxt, block_indent, current_line, 
                                           block_start_line, "if body")
```

### 3.2 Central Keyword Registry

#### 3.2.1 Keyword Data Structures

**Location:** `namel3ss/lang.py` (new file)

```python
"""
N3 Language Keywords and Grammar Constants.

This module defines the complete keyword hierarchy for the N3 language,
providing a single source of truth for keyword validation and error messages.
"""

from typing import Dict, FrozenSet, List, Optional, Set
from dataclasses import dataclass

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
    'language_version': ['language_version'],  # Single token but version-like
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
        description: Human-readable description
    """
    keyword: str
    valid_contexts: Set[str]  # e.g., {'top-level', 'page', 'control-flow'}
    description: str

KEYWORD_CONTEXTS: Dict[str, KeywordContext] = {
    'module': KeywordContext('module', {'top-level'}, 'Module declaration'),
    'import': KeywordContext('import', {'top-level'}, 'Import statement'),
    'app': KeywordContext('app', {'top-level'}, 'Application declaration'),
    'page': KeywordContext('page', {'top-level'}, 'Page definition'),
    'set': KeywordContext('set', {'page', 'control-flow'}, 'Variable assignment'),
    'if': KeywordContext('if', {'page', 'control-flow'}, 'Conditional statement'),
    'for': KeywordContext('for', {'page', 'control-flow'}, 'Loop statement'),
    # ... (complete mapping)
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
    'insihgt': 'insight',
    'insigt': 'insight',
    'promtp': 'prompt',
    'promt': 'prompt',
    'conector': 'connector',
    'conecter': 'connector',
    'tabel': 'table',
    'tabl': 'table',
    'cahrt': 'chart',
    'chrt': 'chart',
    'formt': 'format',
    'formm': 'format',
    # Case variations (N3 is case-sensitive for keywords)
    'Module': 'module',
    'Import': 'import',
    'App': 'app',
    'Page': 'page',
    'Model': 'model',
    'Dataset': 'dataset',
}

# ============================================================================
# Keyword Validation Helpers
# ============================================================================

def suggest_keyword(unknown: str, context: str = 'top-level') -> Optional[str]:
    """
    Suggest the most likely correct keyword for an unknown token.
    
    Args:
        unknown: The unknown keyword
        context: Where the keyword appeared ('top-level', 'page', etc.)
        
    Returns:
        Suggested keyword or None
    """
    # Check direct typo mapping
    if unknown in KEYWORD_TYPOS:
        return KEYWORD_TYPOS[unknown]
    
    # Use fuzzy matching (Levenshtein distance) for context-appropriate keywords
    # Return best match if distance <= 2
    # Implementation: Use difflib or custom fuzzy matcher
    pass  # Implementation in actual code

def valid_keywords_for_context(context: str) -> List[str]:
    """
    Get list of valid keywords for a given context.
    
    Args:
        context: The parsing context ('top-level', 'page', 'control-flow')
        
    Returns:
        Sorted list of valid keywords
    """
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
    """
    if context == 'any':
        return keyword in KEYWORD_CONTEXTS
    
    ctx = KEYWORD_CONTEXTS.get(keyword)
    return ctx is not None and context in ctx.valid_contexts
```

#### 3.2.2 Keyword Validation Methods

**Location:** `namel3ss/parser/base.py` - extend `ParserBase`

```python
from namel3ss.lang import (
    suggest_keyword,
    valid_keywords_for_context,
    is_valid_keyword,
    TOP_LEVEL_KEYWORDS,
    PAGE_STATEMENT_KEYWORDS,
)

def _validate_top_level_keyword(
    self,
    line: str,
    line_no: int,
    keyword: str
) -> None:
    """
    Validate that keyword is allowed at top level.
    
    Args:
        line: The full line text
        line_no: Line number
        keyword: The keyword to validate
        
    Raises:
        N3SyntaxError: If keyword is invalid with suggestion
    """
    if not is_valid_keyword(keyword, 'top-level'):
        suggestion = suggest_keyword(keyword, 'top-level')
        valid_list = ', '.join(valid_keywords_for_context('top-level'))
        
        message = f"Unknown top-level keyword '{keyword}'"
        hint = None
        
        if suggestion:
            message += f"\n  → Did you mean '{suggestion}'?"
            hint = f"Valid top-level keywords: {valid_list}"
        else:
            hint = f"Valid top-level keywords are: {valid_list}"
        
        raise self._error(message, line_no, line, hint=hint)

def _validate_page_statement_keyword(
    self,
    line: str,
    line_no: int,
    keyword: str
) -> None:
    """
    Validate that keyword is allowed in page body.
    
    Similar to _validate_top_level_keyword but for page context.
    """
    # Similar implementation for page context

def _suggest_keyword_correction(
    self,
    unknown: str,
    context: str = 'any'
) -> Optional[str]:
    """
    Suggest correction for unknown keyword.
    
    Args:
        unknown: The unknown keyword
        context: Parsing context
        
    Returns:
        Suggested keyword or None
    """
    return suggest_keyword(unknown, context)
```

### 3.3 Enhanced Error Messages and Coercion

#### 3.3.1 Improved Coercion with Context

**Location:** `namel3ss/parser/base.py`

```python
def _coerce_scalar_with_context(
    self,
    raw: Any,
    field_name: str,
    expected_type: Optional[str] = None,
    line_no: Optional[int] = None,
    line: Optional[str] = None
) -> Any:
    """
    Coerce scalar value with contextual error messages.
    
    Args:
        raw: The raw value to coerce
        field_name: Name of the field being parsed (for errors)
        expected_type: Expected type hint ('int', 'float', 'bool', 'string')
        line_no: Line number for error reporting
        line: Line text for error reporting
        
    Returns:
        Coerced value
        
    Raises:
        N3SyntaxError: If coercion fails with helpful context
        
    Examples:
        # Good:
        value = self._coerce_scalar_with_context(raw, "page_size", "int", line_no, line)
        
        # Error message:
        N3SyntaxError: Invalid value for page_size
          at line 42 in config.ai
          → Expected an integer, got "abc"
          Hint: page_size must be a positive integer
    """
    try:
        result = self._coerce_scalar(raw)
        
        # Validate type if specified
        if expected_type == 'int' and not isinstance(result, int):
            raise ValueError(f"Expected integer, got {type(result).__name__}")
        elif expected_type == 'float' and not isinstance(result, (int, float)):
            raise ValueError(f"Expected number, got {type(result).__name__}")
        elif expected_type == 'bool' and not isinstance(result, bool):
            raise ValueError(f"Expected boolean, got {type(result).__name__}")
        
        return result
        
    except (ValueError, TypeError) as exc:
        message = f"Invalid value for {field_name}"
        if expected_type:
            message += f"\n  → Expected {expected_type}, got {repr(raw)}"
        
        hint = self._coercion_hint(field_name, expected_type)
        
        raise self._error(message, line_no, line, hint=hint)

def _coercion_hint(self, field_name: str, expected_type: Optional[str]) -> Optional[str]:
    """
    Generate helpful hint for coercion errors.
    
    Args:
        field_name: Name of field being coerced
        expected_type: Expected type
        
    Returns:
        Hint string or None
    """
    # Field-specific hints
    hints = {
        'page_size': 'page_size must be a positive integer (e.g., 10, 20, 50)',
        'temperature': 'temperature must be a number between 0.0 and 2.0',
        'max_tokens': 'max_tokens must be a positive integer',
        'ttl_seconds': 'ttl_seconds must be a positive integer',
        'width': 'width must be a positive integer (pixels or percentage)',
        'height': 'height must be a positive integer (pixels or percentage)',
    }
    
    if field_name in hints:
        return hints[field_name]
    
    # Generic type hints
    if expected_type == 'int':
        return 'Must be a whole number (e.g., 1, 42, 100)'
    elif expected_type == 'float':
        return 'Must be a number (e.g., 1.5, 3.14, 0.5)'
    elif expected_type == 'bool':
        return 'Must be true/false, yes/no, or 1/0'
    
    return None
```

#### 3.3.2 Enhanced Error Creation

**Location:** `namel3ss/parser/base.py`

```python
def _error(
    self,
    message: str,
    line_no: Optional[int] = None,
    line: Optional[str] = None,
    hint: Optional[str] = None,
    suggestion: Optional[str] = None
) -> N3SyntaxError:
    """
    Create a comprehensive N3SyntaxError with hints.
    
    Args:
        message: Primary error message
        line_no: Line number (1-indexed)
        line: The problematic line text
        hint: Helpful hint for fixing the error
        suggestion: Suggested correction
        
    Returns:
        N3SyntaxError instance
        
    Examples:
        # Simple error:
        raise self._error("Expected ':' after if condition", line_no, line)
        
        # Error with hint:
        raise self._error(
            "Inconsistent indentation detected",
            line_no,
            line,
            hint="Use consistent indentation throughout each block (2 or 4 spaces)"
        )
        
        # Error with suggestion:
        raise self._error(
            f"Unknown keyword '{kw}'",
            line_no,
            line,
            suggestion=f"Did you mean '{corrected}'?",
            hint="Valid keywords: app, page, model, dataset..."
        )
    """
    path = self.file_path or "<input>"
    
    # Build comprehensive error message
    full_message = message
    if suggestion:
        full_message += f"\n  {suggestion}"
    
    return N3SyntaxError(
        message=full_message,
        path=path,
        line_number=line_no,
        code=line,
        hint=hint
    )
```

---

## 4. Implementation Strategy

### 4.1 Phase 1: Foundation (Tasks 2-3)

**Goal:** Implement core infrastructure without breaking existing code

1. **Create `namel3ss/lang.py`**
   - Define all keyword constants
   - Implement keyword validation helpers
   - Add typo detection and suggestions
   - Write comprehensive docstrings

2. **Extend `namel3ss/parser/base.py`**
   - Add `IndentationInfo` dataclass
   - Implement `_compute_indent_details()`
   - Implement `_expect_indent_greater_than()`
   - Implement `_validate_block_indent()`
   - Implement `_detect_indentation_issues()`
   - Keep `_indent()` for backward compatibility
   - Add keyword validation methods
   - Enhance `_error()` with hint support
   - Add `_coerce_scalar_with_context()`

3. **Write Unit Tests**
   - `tests/test_parser_indentation.py` - Test all indentation methods
   - `tests/test_keyword_registry.py` - Test keyword validation
   - Target: 100% coverage of new code

### 4.2 Phase 2: Parser Migration (Tasks 4-11)

**Goal:** Refactor all parser modules to use new infrastructure

**Order of Migration:**
1. `control_flow.py` - Simplest, well-defined control flow
2. `pages.py` - Uses control flow, single-purpose
3. `program.py` - Top-level orchestration
4. `ai.py` - Complex but isolated
5. `models.py`, `insights.py`, `datasets.py`, `frames.py` - Data structures
6. `crud.py`, `experiments.py`, `eval.py` - Specialized features
7. `logic.py` - Logic programming
8. `actions.py`, `components.py`, `symbolic.py`, `expressions.py` - Utilities

**Migration Pattern for Each Module:**
```python
# BEFORE:
def _parse_something(self, line: str, line_no: int, base_indent: int):
    while self.pos < len(self.lines):
        nxt = self._peek()
        indent = self._indent(nxt)
        if indent <= base_indent:
            break
        # ... parse

# AFTER:
def _parse_something(self, line: str, line_no: int, base_indent: int):
    block_indent: Optional[int] = None
    block_start_line: Optional[int] = None
    
    while self.pos < len(self.lines):
        nxt = self._peek()
        current_line = self.pos + 1
        
        # First line of block: expect indent > base
        if block_indent is None:
            try:
                info = self._expect_indent_greater_than(
                    nxt, base_indent, current_line, "something block"
                )
                block_indent = info.effective_level
                block_start_line = current_line
            except N3SyntaxError:
                break  # End of block
        else:
            # Subsequent lines: validate consistency
            try:
                info = self._validate_block_indent(
                    nxt, block_indent, current_line, block_start_line, "something block"
                )
            except N3SyntaxError as exc:
                if self._indent(nxt) < block_indent:
                    break  # Dedent, end of block
                raise  # Inconsistent indentation, propagate error
        
        # ... parse statement
```

### 4.3 Phase 3: Testing and Validation (Tasks 12-15)

**Goal:** Ensure production quality and no regressions

1. **Write Comprehensive Tests**
   - Unit tests for indentation (tabs, spaces, mixed, errors)
   - Unit tests for keyword validation (typos, context, suggestions)
   - Integration tests with real N3 programs
   - Edge case tests (empty files, all comments, etc.)

2. **Run Existing Test Suite**
   - Execute all existing parser tests
   - Fix any regressions
   - Ensure 100% pass rate

3. **Manual Testing**
   - Test with example N3 programs from `examples/`
   - Verify error messages are helpful
   - Test with intentionally malformed input

### 4.4 Phase 4: Documentation and Polish (Tasks 16-18)

**Goal:** Production-ready deliverable

1. **Documentation**
   - Update `README.md` with parser improvements
   - Create `docs/PARSER_DESIGN.md` with architecture
   - Document error message patterns
   - Add examples of fixing common errors

2. **Code Quality**
   - Add comprehensive docstrings
   - Ensure type hints on all methods
   - Run linters (mypy, flake8, black)
   - Code review checklist

3. **Finalize and Commit**
   - Update `CHANGELOG.md`
   - Update `IMPLEMENTATION_SUMMARY.md`
   - Commit with descriptive message
   - Push to GitHub

---

## 5. Success Criteria

### 5.1 Functional Requirements

- ✅ **Indentation Detection:** Correctly identifies tabs, spaces, and mixed indentation
- ✅ **Consistency Validation:** Detects inconsistent indentation within blocks
- ✅ **Helpful Errors:** Provides actionable error messages with line references
- ✅ **Keyword Validation:** Central registry validates all N3 keywords
- ✅ **Typo Suggestions:** Suggests corrections for misspelled keywords
- ✅ **Context-Aware:** Errors show expected vs actual with context

### 5.2 Quality Requirements

- ✅ **Backwards Compatible:** Existing N3 programs parse correctly
- ✅ **Well-Tested:** >90% code coverage on new code
- ✅ **No Regressions:** All existing tests pass
- ✅ **Production-Ready:** No shortcuts, no demo code
- ✅ **Maintainable:** Clear architecture, comprehensive docs

### 5.3 Error Message Quality Examples

**Before:**
```
N3SyntaxError: Expected ':' after if condition
  at line 15
```

**After:**
```
N3SyntaxError: Missing ':' after if condition
  at line 15 in page.ai
  →  if ctx.user.is_admin
  Hint: All control flow statements (if, for, while) must end with ':'
  
  Expected:
    if ctx.user.is_admin:
```

**Before:**
```
N3SyntaxError: Unknown effect 'pure2'
```

**After:**
```
N3SyntaxError: Unknown effect keyword 'pure2'
  at line 42 in model.ai
  → Did you mean 'pure'?
  Hint: Valid effect keywords: pure, io, read, write, stateful
```

---

## 6. Risk Mitigation

### 6.1 Breaking Changes

**Risk:** Refactoring might break existing parser behavior

**Mitigation:**
- Maintain `_indent()` method for backward compatibility
- Incremental migration with extensive testing after each module
- Test against all example files in `examples/`
- Keep old methods as deprecated fallbacks initially

### 6.2 Performance Impact

**Risk:** More robust parsing might slow down compilation

**Mitigation:**
- Indentation detection is O(n) per line (same as before)
- Keyword lookups use frozenset (O(1))
- Profile parser performance before/after
- Optimize hot paths if needed

### 6.3 Complexity Growth

**Risk:** Central registry might become unwieldy as language grows

**Mitigation:**
- Modular keyword registry design
- Separate files for different keyword categories if needed
- Clear documentation on adding new keywords
- Validation scripts to ensure consistency

---

## 7. Future Enhancements

### 7.1 Configuration-Based Indentation

Allow users to specify indentation preference:
```n3
language_version "1.0.0"
indent_style: spaces
indent_size: 4
```

### 7.2 Auto-Formatting

Provide auto-formatter for N3 code:
```bash
n3 format app.ai  # Fix indentation, normalize spacing
```

### 7.3 IDE Integration

Export indentation and keyword data for:
- Syntax highlighting
- Auto-completion
- Real-time error checking
- Quick fixes in VS Code

### 7.4 Linter Mode

Provide warnings for style issues:
- Mixed indentation across files
- Inconsistent spacing around operators
- Suggested improvements

---

## 8. Appendix

### 8.1 Current Indentation Patterns (Survey Results)

**Files with `_indent()` usage:** 18+ parser modules

**Common patterns:**
```python
# Pattern A: Basic boundary check (60% of usage)
indent = self._indent(line)
if indent <= base_indent:
    break

# Pattern B: Block tracking (30% of usage)
block_indent: Optional[int] = None
if block_indent is None:
    block_indent = indent
elif indent < block_indent:
    break

# Pattern C: Strict validation (10% of usage)
if indent != expected_indent:
    raise self._error("Inconsistent indentation", ...)
```

### 8.2 All Parser Modules

1. `base.py` - ParserBase foundation (858 lines)
2. `program.py` - Top-level program parsing (338 lines)
3. `pages.py` - Page declarations (~190 lines)
4. `ai.py` - AI constructs (1600 lines)
5. `control_flow.py` - If/for/while (~160 lines)
6. `models.py` - ML model declarations (445 lines)
7. `logic.py` - Logic programming (453 lines)
8. `insights.py` - Insight parsing
9. `frames.py` - Data frame parsing
10. `datasets.py` - Dataset declarations
11. `crud.py` - CRUD operations
12. `experiments.py` - Experiment tracking
13. `eval.py` - Evaluation suites
14. `components.py` - UI components
15. `actions.py` - Action parsing
16. `symbolic.py` - Symbolic computation
17. `expressions.py` - Expression parsing

### 8.3 Keywords by Category (Complete)

**Top-Level (indent=0):**
- Program structure: `module`, `import`, `language_version`
- App: `app`
- Data: `dataset`, `frame`, `model`
- AI/ML: `ai`, `prompt`, `connector`, `memory`, `training`, `tuning`, `experiment`
- Logic: `knowledge`, `query`
- Evaluation: `evaluator`, `metric`, `guardrail`, `eval_suite`
- Templates: `define`
- Pages: `page`
- Insights: `insight`
- Styling: `theme`
- CRUD: `enable`

**Page Statements:**
- Variables: `set`
- Control flow: `if`, `elif`, `else`, `for`, `while`, `break`, `continue`
- Components: `show` (+ `text`, `table`, `chart`, `form`)
- Actions: `action`, `predict`

**Modifiers/Attributes:**
- Effects: `pure`, `io`, `read`, `write`, `stateful`
- Page types: `reactive`, `static`
- Sources: `from`, `dataset`, `table`, `frame`
- Layout: `layout`, `at`, `kind`

---

**END OF DESIGN DOCUMENT**
