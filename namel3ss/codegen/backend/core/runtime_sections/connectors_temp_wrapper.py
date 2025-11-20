"""Connector drivers - backward compatibility wrapper.

REFACTORING NOTE: This module has been refactored into a modular package.
Original: 1,601 lines (monolithic template string)  
New structure: 7 focused modules in connectors/ package

For backward compatibility, this wrapper directly re-exports CONNECTORS_SECTION
from the backup until the modular template generation is validated.
"""

from .connectors_original_backup import CONNECTORS_SECTION

__all__ = ['CONNECTORS_SECTION']
