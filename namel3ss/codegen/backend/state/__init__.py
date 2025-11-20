"""
Backend state encoding package - Modular AST-to-backend translation.

This package refactors the original 2,270-line state.py into focused modules
organized by domain for better maintainability and navigation.

Module Organization:
    datasets.py - Dataset encoding (8 functions)
    frames.py - Frame/table encoding (9 functions)
    models.py - ML model encoding (10 functions)
    insights.py - Insight/analytics encoding (8 functions)
    ai.py - AI resource encoding (10+ functions)
    experiments.py - Experiment encoding (2 functions)
    training.py - Training/tuning job encoding (3 functions)
    agents.py - Agent and graph encoding (2 functions)
    rag.py - RAG index and pipeline encoding (2 functions)
    evaluation.py - Evaluator, metric, guardrail encoding (5 functions)
    pages.py - Page and layout encoding (3 functions)
    statements.py - Statement encoding (2 functions)
    actions.py - Action operation encoding (1 function)
    crud.py - CRUD resource encoding (1 function)
    expressions.py - Expression encoding utilities (14 functions)
    utils.py - Shared utility functions (12 functions)
    main.py - Main build_backend_state orchestration

Usage:
    from namel3ss.codegen.backend.state import build_backend_state, BackendState
"""

from .classes import BackendState, PageSpec, PageComponent
from .main import build_backend_state
from .utils import _component_to_serializable
from .agents import _encode_agent, _encode_graph

__all__ = [
    'build_backend_state',
    'BackendState',
    'PageSpec',
    'PageComponent',
    '_component_to_serializable',
    '_encode_agent',
    '_encode_graph',
]
