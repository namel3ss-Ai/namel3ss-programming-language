"""Connector drivers - generates runtime code template.

REFACTORING NOTE: This module has been refactored into a modular package.
Original: 1,601 lines (monolithic template string)
New structure: 7 focused modules in connectors/ package  
Total: ~1,592 lines (minimal overhead)

Modules:
  - utilities.py: Common helper functions (~240 lines)
  - driver_sql.py: SQL database driver (~75 lines)
  - driver_rest.py: REST API driver (~496 lines)
  - driver_graphql.py: GraphQL driver (~170 lines)
  - driver_grpc.py: gRPC driver (~219 lines)
  - driver_streaming.py: Streaming/WebSocket driver (~370 lines)
  - transformers.py: Row transformation utilities (~22 lines)

This wrapper reconstructs the CONNECTORS_SECTION template string for backward compatibility.
"""

from __future__ import annotations

from textwrap import dedent
from pathlib import Path

# Read all connector modules and combine them into the template string
_connector_dir = Path(__file__).parent / 'connectors'

def _read_module_code(filename: str) -> str:
    """Read a module file and extract the code (skip docstring and imports)."""
    module_path = _connector_dir / filename
    with open(module_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip docstring, imports, and __all__ at the end
    code_lines = []
    in_docstring = False
    skip_imports = True
    
    for line in lines:
        stripped = line.strip()
        
        # Skip module docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                continue
            else:
                in_docstring = False
                continue
        
        if in_docstring:
            continue
            
        # Skip imports section
        if skip_imports:
            if stripped.startswith(('from ', 'import ', 'try:', 'except')) or not stripped:
                continue
            else:
                skip_imports = False
        
        # Skip __all__ at end
        if stripped.startswith('__all__'):
            break
            
        code_lines.append(line)
    
    return ''.join(code_lines)

# Build the complete CONNECTORS_SECTION by combining all modules
_utilities_code = _read_module_code('utilities.py')
_sql_driver_code = _read_module_code('driver_sql.py')
_rest_driver_code = _read_module_code('driver_rest.py')
_graphql_driver_code = _read_module_code('driver_graphql.py')
_grpc_driver_code = _read_module_code('driver_grpc.py')
_streaming_driver_code = _read_module_code('driver_streaming.py')
_transformers_code = _read_module_code('transformers.py')

CONNECTORS_SECTION = dedent(
    '''

import importlib
import inspect
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    httpx = None  # type: ignore

from sqlalchemy.ext.asyncio import AsyncSession

''' + _utilities_code + '\n\n' + 
_sql_driver_code + '\n\n' +
_rest_driver_code + '\n\n' +
_graphql_driver_code + '\n\n' +
_grpc_driver_code + '\n\n' +
_streaming_driver_code + '\n\n' +
_transformers_code +
'''
'''
).rstrip()

__all__ = ['CONNECTORS_SECTION']
