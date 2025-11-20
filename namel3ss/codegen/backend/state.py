"""
BACKWARD COMPATIBILITY WRAPPER

This module has been refactored into a modular package structure.
Original: 2,270 lines, 93 functions, 3 classes
Result: 17 focused modules + this 43-line wrapper

New structure: namel3ss/codegen/backend/state/
    classes.py - Core dataclasses (3 classes)
    expressions.py - Expression encoding (14 functions)
    utils.py - Helper utilities (12 functions)
    datasets.py - Dataset encoding (8 functions)
    frames.py - Frame/table encoding (9 functions)
    models.py - ML model encoding (10 functions)
    ai.py - AI resource encoding (14 functions)
    insights.py - Insight encoding (8 functions)
    evaluation.py - Evaluator/metric encoding (5 functions)
    training.py - Training/tuning job encoding (5 functions)
    experiments.py - Experiment encoding (2 functions)
    agents.py - Agent encoding (2 functions)
    pages.py - Page encoding (3 functions)
    statements.py - Statement encoding (2 functions)
    actions.py - Action operation encoding (1 function)
    crud.py - CRUD resource encoding (1 function)
    logic.py - Logic programming encoding (5 functions)
    variables.py - Variable assignment encoding (1 function)
    main.py - Main orchestration (build_backend_state function)

For new code, import directly from the package:
    from namel3ss.codegen.backend.state import build_backend_state, BackendState

This wrapper maintains backward compatibility for existing imports:
    from namel3ss.codegen.backend.state import build_backend_state, BackendState
"""

from .state.classes import BackendState, PageComponent, PageSpec
from .state.main import build_backend_state
from .state.utils import _component_to_serializable

__all__ = [
    "build_backend_state",
    "BackendState",
    "PageComponent",
    "PageSpec",
    "_component_to_serializable",
]
