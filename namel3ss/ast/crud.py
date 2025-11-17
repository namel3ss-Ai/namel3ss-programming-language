"""AST nodes for CRUD resource declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CrudResource:
    """Describe a CRUD-enabled resource bound to a table or dataset."""

    name: str
    source_type: str
    source_name: str
    primary_key: str = "id"
    select_fields: List[str] = field(default_factory=list)
    mutable_fields: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(
        default_factory=lambda: ["list", "retrieve", "create", "update", "delete"],
    )
    tenant_column: Optional[str] = None
    default_limit: int = 100
    max_limit: int = 500
    read_only: bool = False
    label: Optional[str] = None


__all__ = ["CrudResource"]
