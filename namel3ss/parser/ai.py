"""
Backward compatibility wrapper for AIParserMixin.

The original 2,202-line AIParserMixin has been refactored into a modular
package structure for better maintainability. This file maintains backward
compatibility by re-exporting the AIParserMixin from the new location.

Original Structure (ai.py):
    - 2,202 lines
    - 36 methods in single AIParserMixin class
    - Monolithic implementation

New Structure (ai/ package):
    - models.py (460 lines): Connectors, templates, chains, memory, AI models, detection (10 methods)
    - chains.py (120 lines): Multi-step workflow chain definitions (1 method)
    - prompts.py (195 lines): Structured prompt definitions (1 method)
    - schemas.py (650 lines): Input/output schema parsing (11 methods)
    - training.py (400 lines): Training and tuning job specifications (5 methods)
    - workflows.py (307 lines): Workflow control flow (7 methods)
    - utils.py (45 lines): Shared utility functions (3 functions)
    - main.py (60 lines): AIParserMixin composition class

Total: ~2,237 lines across 8 focused modules (includes documentation)
Wrapper: 35 lines (98.4% reduction from original)

All original functionality is preserved. Import paths remain unchanged:
    from namel3ss.parser.ai import AIParserMixin  # Still works!
"""

from namel3ss.parser.ai.main import AIParserMixin

__all__ = ['AIParserMixin']
