from __future__ import annotations

from typing import Optional

from namel3ss.ast import App, Module
from .base import N3SyntaxError
from .program import ProgramParserMixin


class Parser(ProgramParserMixin):
    """Public parser entry point for Namel3ss programs."""

    def __init__(self, source: str, *, module_name: Optional[str] = None, path: str = ""):
        super().__init__(source, module_name=module_name, path=path)

    def parse(self) -> Module:
        return super().parse()

    def parse_app(self) -> App:
        module = self.parse()
        if not module.body:
            raise N3SyntaxError("Module does not contain any declarations", 0, "")
        root = module.body[0]
        if not isinstance(root, App):
            raise ValueError("Module body does not contain an app definition")
        if not module.has_explicit_app:
            raise ValueError("Module does not declare an app definition")
        return root


__all__ = ["Parser", "N3SyntaxError"]
