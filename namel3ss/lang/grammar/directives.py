"""Directive parsing methods (module, import, language_version)."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .helpers import _Line
    from namel3ss.ast.modules import ImportedName


class DirectiveParserMixin:
    """Mixin providing directive parsing methods for grammar parser."""

    def _parse_module_declaration(self, line: _Line) -> None:
        """Parse module declaration: module <dotted.name>"""
        from .constants import MODULE_DECL_RE
        
        match = MODULE_DECL_RE.match(line.text.strip())
        if not match:
            raise self._error("Expected: module <name>", line)
        if self._module_name is not None:
            raise self._error("Only one module declaration is allowed", line)
        if self._imports:
            raise self._error("Module declaration must appear before imports", line)
        if self._app is not None:
            raise self._error("Module declaration must appear before app declaration", line)
        self._module_name = match.group(1)
        self._advance()

    def _parse_import(self, line: _Line) -> None:
        """Parse import statement: import module.name [as alias] [: name1, name2]"""
        from .constants import IMPORT_TARGET_RE
        from namel3ss.ast.modules import Import, ImportedName
        
        remainder = line.text.strip()[len('import ') :].strip()
        if not remainder:
            raise self._error("Expected module path after 'import'", line)
        module_part = remainder
        names_part: Optional[str] = None
        colon_index = remainder.find(':')
        if colon_index != -1:
            module_part = remainder[:colon_index].strip()
            names_part = remainder[colon_index + 1 :].strip()
            if not names_part:
                raise self._error("Expected imported names after ':'", line)
        target_match = IMPORT_TARGET_RE.match(module_part)
        if not target_match:
            raise self._error("Invalid module import target", line)
        alias = target_match.group(2)
        module_name = target_match.group(1)
        names: Optional[List[ImportedName]] = None
        if names_part is not None:
            names = self._parse_import_names(names_part, line)
            if alias:
                raise self._error("Module alias is not allowed when selecting specific names", line)
        self._imports.append(Import(module=module_name, names=names, alias=alias))
        self._advance()

    def _parse_import_names(self, segment: str, line: _Line) -> List[ImportedName]:
        """Parse comma-separated list of imported names with optional aliases."""
        from .constants import IMPORT_TARGET_RE
        from namel3ss.ast.modules import ImportedName
        
        entries = [piece.strip() for piece in segment.split(',') if piece.strip()]
        results: List[ImportedName] = []
        for entry in entries:
            match = IMPORT_TARGET_RE.match(entry)
            if not match or match.group(1).count('.'):
                raise self._error("Invalid imported name", line)
            results.append(ImportedName(name=match.group(1), alias=match.group(2)))
        if not results:
            raise self._error("Import list cannot be empty", line)
        return results

    def _parse_language_version(self, line: _Line) -> None:
        """Parse language_version directive: language_version "1.0.0" """
        from .constants import LANGUAGE_VERSION_RE
        
        match = LANGUAGE_VERSION_RE.match(line.text.strip())
        if not match:
            raise self._error('Expected: language_version "<semver>"', line)
        if self._language_version is not None:
            raise self._error('language_version directive may only appear once', line)
        self._language_version = match.group(1)
        self._advance()


__all__ = ['DirectiveParserMixin']
