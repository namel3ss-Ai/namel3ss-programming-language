"""Frame execution runtime package - Modular frame processing.

This package refactors the original 1,760-line frames.py into focused modules
organized by functionality for better maintainability and navigation.

Module Organization:
    constants.py - Constants, type aliases, exceptions (95 lines)
    utilities.py - Core utility functions (119 lines)
    backend_base.py - Abstract base class (76 lines)
    backend_polars.py - Polars implementation (~200 lines)
    backend_pandas.py - Pandas implementation (~160 lines)
    backend_python.py - Python fallback (~410 lines)
    normalizers.py - Column/expression normalization (~115 lines)
    loaders.py - File and SQL loading (~130 lines)
    frame.py - N3Frame class and projection (~200 lines)
    pipeline.py - Pipeline engine and execution (~310 lines)

Usage:
    from namel3ss.codegen.backend.core.runtime.frames import (
        N3Frame,
        project_frame_rows,
        load_frame_file_source,
        load_frame_sql_source,
        execute_frame_pipeline_plan,
        build_pipeline_frame_spec,
    )
"""

from .constants import (
    DEFAULT_FRAME_LIMIT,
    MAX_FRAME_LIMIT,
    FrameSourceLoadError,
    FramePipelineExecutionError,
)
from .frame import N3Frame, project_frame_rows
from .loaders import load_frame_file_source, load_frame_sql_source
from .pipeline import execute_frame_pipeline_plan, build_pipeline_frame_spec

__all__ = [
    "N3Frame",
    "project_frame_rows",
    "DEFAULT_FRAME_LIMIT",
    "MAX_FRAME_LIMIT",
    "FrameSourceLoadError",
    "load_frame_file_source",
    "load_frame_sql_source",
    "FramePipelineExecutionError",
    "execute_frame_pipeline_plan",
    "build_pipeline_frame_spec",
]
