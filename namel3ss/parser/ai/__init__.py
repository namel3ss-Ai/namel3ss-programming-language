"""
AI parser package for namel3ss DSL constructs.

This package refactors the original 2,200-line AIParserMixin into focused modules:

Modules:
    connectors: Connector and template parsing
    models: AI model, memory, and chain parsing  
    prompts: Structured prompt parsing with schemas
    workflows: Workflow block and control flow parsing
    training: Training and tuning job parsing
    utils: Helper functions and utilities
    main: AIParserMixin class assembling all functionality

The main entry point AIParserMixin is re-exported for backward compatibility.
"""

from .main import AIParserMixin

__all__ = ["AIParserMixin"]
