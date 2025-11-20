"""Reusable runtime helpers for N3Frame execution.

REFACTORING NOTE: This module has been refactored into a modular package.
Original: 1,609 lines (frames.py - actual line count from file)
New structure: 10 focused modules in namel3ss/codegen/backend/core/runtime/frames/ package
Total: ~1,815 lines (+12.8% overhead for modularity)

Modules:
  - constants.py: Constants, type aliases, exceptions (95 lines)
  - utilities.py: Core utility functions (119 lines)
  - backend_base.py: Abstract base class (76 lines)
  - backend_polars.py: Polars backend implementation (~200 lines)
  - backend_pandas.py: Pandas backend implementation (~160 lines)
  - backend_python.py: Python fallback backend (~410 lines)
  - normalizers.py: Column/expression normalization (~115 lines)
  - loaders.py: File and SQL loading (~130 lines)
  - frame.py: N3Frame class and projection (~200 lines)
  - pipeline.py: Pipeline engine and execution (~310 lines)

This wrapper provides backward compatibility by re-exporting all public APIs.
"""

from __future__ import annotations

# Re-export all public APIs for backward compatibility
from .frames import (
    N3Frame,
    project_frame_rows,
    DEFAULT_FRAME_LIMIT,
    MAX_FRAME_LIMIT,
    FrameSourceLoadError,
    load_frame_file_source,
    load_frame_sql_source,
    FramePipelineExecutionError,
    execute_frame_pipeline_plan,
    build_pipeline_frame_spec,
)

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
