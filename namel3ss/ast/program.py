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

    def __getattr__(self, name: str) -> Any:
        """
        Backward-compatible attribute forwarding.

        Many legacy callers expect parser.parse() to return the App directly.
        When a single App is present as the first body element, forward
        attribute access to it.
        """
        if self.body:
            target = self.body[0]
            if hasattr(target, name):
                return getattr(target, name)
        raise AttributeError(f"Module has no attribute '{name}'")


@dataclass
class Program:
    """Collection of modules participating in a compilation unit."""

    modules: List[Module] = field(default_factory=list)


__all__ = ["Module", "Program"]
