"""Program/module level AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .modules import Import
from .comments import Comment


@dataclass
class Module:
    """Parsed module containing top-level statements."""

    name: Optional[str] = None
    language_version: Optional[str] = None
    path: str = ""
    imports: List[Import] = field(default_factory=list)
    body: List[Any] = field(default_factory=list)
    has_explicit_app: bool = False
    comments: List[Comment] = field(default_factory=list)


@dataclass
class Program:
    """Collection of modules participating in a compilation unit."""

    modules: List[Module] = field(default_factory=list)


__all__ = ["Module", "Program"]
