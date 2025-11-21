from __future__ import annotations

from typing import Optional

from namel3ss.ast import App, Module
from namel3ss.lang.parser import parse_module as new_parse_module
from namel3ss.lang.parser import N3SyntaxError


class Parser:
    """Public parser entry point for Namel3ss programs."""

    def __init__(self, source: str, *, module_name: Optional[str] = None, path: str = ""):
        self._source = source
        self._module_name = module_name
        self.source_path = path

    def parse(self) -> Module:
        """Parse source into Module AST using the unified parser."""
        return new_parse_module(self._source, path=self.source_path, module_name=self._module_name)

    def parse_app(self) -> App:
        """Parse source and extract the App node."""
        module = self.parse()
        if not module.body:
            raise N3SyntaxError(
                "Module does not contain any declarations",
                path=self.source_path or None,
                line=0,
                code="SYNTAX_MISSING_APP",
            )
        root = module.body[0]
        if not isinstance(root, App):
            raise ValueError("Module body does not contain an app definition")
        if not module.has_explicit_app:
            raise ValueError("Module does not declare an app definition")
        return root


__all__ = ["Parser", "N3SyntaxError"]
