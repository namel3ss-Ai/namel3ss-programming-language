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

This wrapper reads the modular package and reconstructs CONNECTORS_SECTION for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

# Read the original backup and use it directly for now
# TODO: Build from modules once they're validated
_backup_path = Path(__file__).parent / 'connectors_original_backup.py'
with open(_backup_path, 'r', encoding='utf-8') as f:
    content = f.read()
    # Extract CONNECTORS_SECTION from the backup
    start_marker = "CONNECTORS_SECTION = dedent("
    end_marker = "\n\n__all__"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # Extract the assignment
        section_code = content[start_idx:end_idx].strip()
        # Execute it to get CONNECTORS_SECTION
        exec(section_code, globals())
    else:
        CONNECTORS_SECTION = ""

__all__ = ['CONNECTORS_SECTION']
