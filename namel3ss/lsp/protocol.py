"""Shared protocol helpers for the Namel3ss language server."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from lsprotocol.types import Position, Range


class SymbolType(Enum):
    """High level categories understood by the language server."""

    APP = auto()
    DATASET = auto()
    FRAME = auto()
    PAGE = auto()
    MODEL = auto()
    PROMPT = auto()
    TEMPLATE = auto()
    CHAIN = auto()
    EXPERIMENT = auto()
    EVALUATOR = auto()
    METRIC = auto()
    GUARDRAIL = auto()
    VARIABLE = auto()
    PAGE_VARIABLE = auto()
    COLUMN = auto()


@dataclass(slots=True)
class SymbolLocation:
    """Represents a URI + range pair."""

    uri: str
    range: Range


@dataclass(slots=True)
class IndexedSymbol:
    """Symbol metadata tracked by the workspace index."""

    name: str
    type: SymbolType
    location: SymbolLocation
    detail: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def matches(self, query: str) -> bool:
        """Return True if *query* is contained in the symbol name."""

        return query.lower() in self.name.lower()


@dataclass(slots=True)
class ColumnInfo:
    """Schema summary used by completion and hover handlers."""

    name: str
    dtype: Optional[str] = None
    nullable: Optional[bool] = None

    def format_signature(self) -> str:
        dtype = self.dtype or "any"
        null_suffix = "?" if self.nullable else ""
        return f"{self.name}: {dtype}{null_suffix}"


@dataclass(slots=True)
class CompletionContext:
    """Light-weight description of the logical context for completion."""

    token: Optional[str]
    prefix: str
    scope_symbol: Optional[str] = None
    scope_type: Optional[SymbolType] = None


__all__ = [
    "SymbolType",
    "SymbolLocation",
    "IndexedSymbol",
    "ColumnInfo",
    "CompletionContext",
    "Position",
    "Range",
]
