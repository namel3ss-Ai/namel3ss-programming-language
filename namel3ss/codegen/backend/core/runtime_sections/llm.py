"""Backward compatibility wrapper for LLM runtime code generation.

This module maintains backward compatibility by re-exporting the
refactored LLM_SECTION from the llm package.

The LLM section has been refactored into a modular package structure:
    namel3ss/codegen/backend/core/runtime_sections/llm/
        imports.py - Import statements (27 lines)
        utilities.py - Utility functions (174 lines)
        http_client.py - HTTP client with retries (93 lines)
        response_parser.py - Response extraction (75 lines)
        tools.py - Tool/plugin system (178 lines)
        workflow.py - Workflow execution (681 lines)
        prompts.py - Prompt handling (273 lines)
        connectors.py - LLM connectors (416 lines)
        structured.py - Structured prompts (445 lines)
        main.py - Entry points (64 lines)
        __init__.py - Package composition (28 lines)

Original file: 2,181 lines, 40 functions, monolithic template string
New structure: 11 modules, ~2,454 lines total with module headers
Reduction in wrapper: 2,181 lines to 24 lines (99% reduction)
"""

from .llm import LLM_SECTION

__all__ = ['LLM_SECTION']
