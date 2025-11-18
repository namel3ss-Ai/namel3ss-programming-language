"""Grammar-driven parser entry points for Namel3ss `.n3` programs.

This module introduces a lightweight EBNF description of the supported
surface syntax and a reference implementation that produces the existing
AST structures used throughout the runtime.  The grammar intentionally
captures the high-level layout of a module while delegating certain
expression details to the established expression parser mixin.

```
module         ::= directive* declaration*
directive      ::= module_decl | import_stmt | language_version
module_decl    ::= "module" dotted_name NEWLINE
import_stmt    ::= "import" dotted_name ("as" NAME)? (":" import_target ("," import_target)*)? NEWLINE
import_target  ::= NAME ("as" NAME)?
language_version ::= "language_version" STRING NEWLINE

declaration    ::= app_decl | theme_block | dataset_def | frame_def | page_block
                 | model_block | ai_model_block | prompt_block | memory_block
                 | template_block | chain_block | experiment_block | crud_block
                 | evaluator_block | metric_block | guardrail_block

app_decl       ::= "app" STRING ("connects to" NAME STRING)? "."?
theme_block    ::= "theme" ":" INDENT theme_entry+ DEDENT

dataset_def    ::= "dataset" STRING "from" source_ref ":" INDENT dataset_stmt+ DEDENT
dataset_stmt   ::= "filter by" ":" expression
                 | "group by" ":" name_list
                 | "order by" ":" name_list
                 | "transform" NAME ":" expression

frame_def      ::= "frame" STRING ("from" source_ref)? ":" INDENT frame_stmt+ DEDENT
frame_stmt     ::= "columns" ":" column_list
                 | "description" ":" STRING

page_block     ::= "page" STRING "at" STRING ":" INDENT page_stmt+ DEDENT
page_stmt      ::= show_stmt | form_stmt | action_stmt | control_flow_stmt
control_flow_stmt ::= if_stmt | for_stmt
if_stmt        ::= "if" expression ":" INDENT page_stmt+ DEDENT (elif_stmt)* (else_stmt)?
for_stmt       ::= "for" NAME "in" ("dataset"|"table"|"frame") NAME ":" INDENT page_stmt+ DEDENT
show_stmt      ::= "show" ("text"|"table"|"chart") ...
```

Only a subset of the listed productions are implemented at this stage;
the focus of Phase 1 is to support realistic modules that declare an app,
datasets, frames, pages, and page-level control-flow.  The grammar text
serves both as documentation and as a roadmap for future incremental
coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from namel3ss.ast import (
    App,
    Dataset,
    FilterOp,
    Frame,
    FrameColumn,
    GroupByOp,
    Page,
    PageStatement,
    ShowChart,
    ShowTable,
    ShowText,
    Theme,
)
from namel3ss.ast.pages import ElifBlock, ForLoop, IfBlock
from namel3ss.ast.program import Module
from namel3ss.ast.modules import Import, ImportedName
from namel3ss.parser.base import N3SyntaxError, ParserBase
from namel3ss.parser.expressions import ExpressionParserMixin


_DATASET_HEADER_RE = re.compile(r'^dataset\s+"([^"]+)"\s+from\s+(\w+)\s+([A-Za-z0-9_"\.]+)\s*:\s*$')
_FRAME_HEADER_RE = re.compile(r'^frame\s+"([^"]+)"(?:\s+from\s+(\w+)\s+([A-Za-z0-9_"\.]+))?\s*:\s*$')
_PAGE_HEADER_RE = re.compile(r'^page\s+"([^"]+)"\s+at\s+"([^"]+)"\s*:\s*$')
_APP_HEADER_RE = re.compile(
    r'^app\s+"([^"]+)"(?:\s+connects\s+to\s+[A-Za-z_][A-Za-z0-9_]*\s+"([^"]+)")?\s*\.?$'
)
_MODULE_DECL_RE = re.compile(r'^module\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*$')
_LANGUAGE_VERSION_RE = re.compile(r'^language_version\s+"([0-9]+\.[0-9]+\.[0-9]+)"\s*\.?$')
_IMPORT_TARGET_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?\s*$')


class GrammarUnsupportedError(Exception):
    """Raised when the grammar-backed parser encounters an unsupported construct."""


class _ExpressionHelper(ExpressionParserMixin):
    """Adapter around the legacy expression parser for reuse in the grammar parser."""

    def __init__(self) -> None:
        # The base parser expects an initial source buffer; we keep it empty
        # and populate `lines`/`pos` before every parse call.
        ParserBase.__init__(self, "", path="")

    def parse(self, text: str, *, line_no: int, line: str) -> object:
        self.lines = [line]
        self.pos = line_no
        return self._parse_expression(text)


def parse_module(source: str, path: str = "", module_name: Optional[str] = None) -> Module:
    """Parse *source* using the grammar-backed parser and return a Module AST."""

    parser = _GrammarModuleParser(source=source, path=path, module_name=module_name)
    try:
        return parser.parse()
    except GrammarUnsupportedError:
        from namel3ss.parser.program import LegacyProgramParser

        legacy = LegacyProgramParser(source, module_name=module_name, path=path)
        return legacy.parse()


@dataclass
class _Line:
    text: str
    number: int


class _GrammarModuleParser:
    """Recursive-descent parser that follows the EBNF outlined above."""

    def __init__(self, source: str, *, path: str = "", module_name: Optional[str] = None) -> None:
        self._source = source
        self._lines = source.splitlines()
        self._path = path
        self._cursor = 0
        self._app: Optional[App] = None
        self._explicit_app_declared = False
        self._module_name: Optional[str] = None
        self._module_name_override = module_name
        self._language_version: Optional[str] = None
        self._imports: List[Import] = []
        self._extra_nodes: List[object] = []
        self._expression_helper = _ExpressionHelper()
        self._directives_locked = False

    # ------------------------------------------------------------------
    # High-level driver
    # ------------------------------------------------------------------
    def parse(self) -> Module:
        while self._cursor < len(self._lines):
            line = self._peek()
            if line is None:
                break
            stripped = line.text.strip()
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            indent = self._indent(line.text)
            if indent != 0:
                raise self._error("Top level statements must not be indented", line)
            if stripped.startswith('module '):
                if self._directives_locked:
                    raise self._error("Module declaration must appear before other statements", line)
                self._parse_module_declaration(line)
                continue
            if stripped.startswith('import '):
                if self._directives_locked:
                    raise self._error("Import statements must appear before other declarations", line)
                self._parse_import(line)
                continue
            if stripped.lower().startswith('language_version'):
                if self._directives_locked:
                    raise self._error("language_version must appear before other declarations", line)
                if self._language_version is not None:
                    raise self._error('language_version directive may only appear once', line)
                self._parse_language_version(line)
                continue
            self._directives_locked = True
            if stripped.startswith('app '):
                self._parse_app(line)
                continue
            if stripped.startswith('theme'):
                self._parse_theme(line)
                continue
            if stripped.startswith('dataset '):
                self._parse_dataset(line)
                continue
            if stripped.startswith('frame '):
                self._parse_frame(line)
                continue
            if stripped.startswith('page '):
                self._parse_page(line)
                continue
            self._unsupported(line, f"top-level statement '{stripped.split()[0]}'")

        body: List[object] = []
        if self._app is not None:
            body.append(self._app)
        body.extend(self._extra_nodes)
        return Module(
            name=self._module_name or self._module_name_override,
            language_version=self._language_version,
            path=self._path,
            imports=list(self._imports),
            body=body,
            has_explicit_app=self._explicit_app_declared,
        )

    # ------------------------------------------------------------------
    # Directive parsing
    # ------------------------------------------------------------------
    def _parse_module_declaration(self, line: _Line) -> None:
        match = _MODULE_DECL_RE.match(line.text.strip())
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
        target_match = _IMPORT_TARGET_RE.match(module_part)
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
        entries = [piece.strip() for piece in segment.split(',') if piece.strip()]
        results: List[ImportedName] = []
        for entry in entries:
            match = _IMPORT_TARGET_RE.match(entry)
            if not match or match.group(1).count('.'):
                raise self._error("Invalid imported name", line)
            results.append(ImportedName(name=match.group(1), alias=match.group(2)))
        if not results:
            raise self._error("Import list cannot be empty", line)
        return results

    def _parse_language_version(self, line: _Line) -> None:
        match = _LANGUAGE_VERSION_RE.match(line.text.strip())
        if not match:
            raise self._error('Expected: language_version "<semver>"', line)
        if self._language_version is not None:
            raise self._error('language_version directive may only appear once', line)
        self._language_version = match.group(1)
        self._advance()

    # ------------------------------------------------------------------
    # Top-level declarations
    # ------------------------------------------------------------------
    def _parse_app(self, line: _Line) -> None:
        match = _APP_HEADER_RE.match(line.text.strip())
        if not match:
            raise self._error('Expected: app "Name" [connects to postgres "ALIAS"].', line)
        name = match.group(1)
        database = match.group(2)
        if self._app is not None and self._explicit_app_declared:
            raise self._error('Only one app declaration is allowed', line)
        self._app = App(name=name, database=database)
        self._explicit_app_declared = True
        self._advance()

    def _parse_theme(self, line: _Line) -> None:
        base_indent = self._indent(line.text)
        if not line.text.rstrip().endswith(':'):
            raise self._error("Theme declaration must end with ':'", line)
        self._advance()
        entries = self._parse_kv_block(base_indent)
        self._ensure_app(line)
        theme = self._app.theme if self._app and self._app.theme else Theme()
        theme.values.update(entries)
        if self._app:
            self._app.theme = theme

    def _parse_dataset(self, line: _Line) -> None:
        match = _DATASET_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "dataset declaration")
        name = match.group(1)
        source_type = match.group(2)
        raw_source = match.group(3)
        source = raw_source.strip('"')
        base_indent = self._indent(line.text)
        self._advance()
        operations = []
        while True:
            next_line = self._peek()
            if next_line is None:
                break
            stripped = next_line.text.strip()
            indent = self._indent(next_line.text)
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if stripped.startswith('filter by:'):
                expr_text = stripped.split(':', 1)[1].strip()
                expr = self._expression_helper.parse(expr_text, line_no=next_line.number, line=next_line.text)
                operations.append(FilterOp(condition=expr))
                self._advance()
                continue
            if stripped.startswith('group by:'):
                columns_text = stripped.split(':', 1)[1]
                columns = [col.strip() for col in columns_text.split(',') if col.strip()]
                operations.append(GroupByOp(columns=columns))
                self._advance()
                continue
            self._unsupported(next_line, "dataset clause")
        dataset = Dataset(name=name, source_type=source_type, source=source, operations=operations)
        self._ensure_app(line)
        if self._app:
            self._app.datasets.append(dataset)

    def _parse_frame(self, line: _Line) -> None:
        match = _FRAME_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "frame declaration")
        name = match.group(1)
        source_type = match.group(2) or 'dataset'
        source = match.group(3).strip('"') if match.group(3) else None
        base_indent = self._indent(line.text)
        self._advance()
        columns: List[FrameColumn] = []
        description: Optional[str] = None
        while True:
            next_line = self._peek()
            if next_line is None:
                break
            stripped = next_line.text.strip()
            indent = self._indent(next_line.text)
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= base_indent:
                break
            if stripped.startswith('columns:'):
                column_text = stripped.split(':', 1)[1]
                column_names = [col.strip() for col in column_text.split(',') if col.strip()]
                columns.extend(FrameColumn(name=col) for col in column_names)
                self._advance()
                continue
            if stripped.startswith('description:'):
                description = stripped.split(':', 1)[1].strip().strip('"')
                self._advance()
                continue
            self._unsupported(next_line, "frame clause")
        frame = Frame(name=name, source_type=source_type, source=source, description=description, columns=columns)
        self._ensure_app(line)
        if self._app:
            self._app.frames.append(frame)

    def _parse_page(self, line: _Line) -> None:
        match = _PAGE_HEADER_RE.match(line.text.strip())
        if not match:
            self._unsupported(line, "page declaration")
        page_name = match.group(1)
        route = match.group(2)
        base_indent = self._indent(line.text)
        self._advance()
        statements = self._parse_page_statements(base_indent)
        page = Page(name=page_name, route=route, statements=statements)
        self._ensure_app(line)
        if self._app:
            self._app.pages.append(page)

    # ------------------------------------------------------------------
    # Page statements and control flow
    # ------------------------------------------------------------------
    def _parse_page_statements(self, parent_indent: int) -> List[PageStatement]:
        statements: List[PageStatement] = []
        while True:
            line = self._peek()
            if line is None:
                break
            stripped = line.text.strip()
            indent = self._indent(line.text)
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            if stripped.startswith('show text '):
                statements.append(self._parse_show_text(line))
                continue
            if stripped.startswith('show table '):
                statements.append(self._parse_show_table(line))
                continue
            if stripped.startswith('show chart '):
                statements.append(self._parse_show_chart(line))
                continue
            if stripped.startswith('if '):
                statements.append(self._parse_if_block(line, indent))
                continue
            if stripped.startswith('for '):
                statements.append(self._parse_for_loop(line, indent))
                continue
            self._unsupported(line, "page statement")
        return statements

    def _parse_show_text(self, line: _Line) -> ShowText:
        match = re.match(r'^\s*show\s+text\s+"([^"]+)"\s*$', line.text)
        if not match:
            raise self._error('Expected: show text "message"', line)
        self._advance()
        return ShowText(text=match.group(1))

    def _parse_show_table(self, line: _Line) -> ShowTable:
        match = re.match(
            r'^\s*show\s+table\s+"([^"]+)"\s+from\s+(dataset|table|frame)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$',
            line.text,
        )
        if not match:
            self._unsupported(line, "show table statement")
        self._advance()
        return ShowTable(title=match.group(1), source_type=match.group(2), source=match.group(3))

    def _parse_show_chart(self, line: _Line) -> ShowChart:
        match = re.match(
            r'^\s*show\s+chart\s+"([^"]+)"\s+from\s+(dataset|table)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$',
            line.text,
        )
        if not match:
            self._unsupported(line, "show chart statement")
        self._advance()
        return ShowChart(heading=match.group(1), source_type=match.group(2), source=match.group(3))

    def _parse_if_block(self, line: _Line, indent: int) -> IfBlock:
        condition_text = line.text.strip()
        if not condition_text.endswith(':'):
            raise self._error("if statement must end with ':'", line)
        condition_src = condition_text[len('if') : -1].strip()
        condition = self._expression_helper.parse(condition_src, line_no=line.number, line=line.text)
        self._advance()
        body = self._parse_page_statements(indent)
        elifs: List[ElifBlock] = []
        else_body: Optional[List[PageStatement]] = None
        while True:
            next_line = self._peek()
            if next_line is None:
                break
            stripped = next_line.text.strip()
            next_indent = self._indent(next_line.text)
            if not stripped:
                self._advance()
                continue
            if next_indent != indent:
                break
            if stripped.startswith('elif '):
                if not stripped.endswith(':'):
                    raise self._error("elif statement must end with ':'", next_line)
                expr_text = stripped[len('elif') : -1].strip()
                condition_expr = self._expression_helper.parse(expr_text, line_no=next_line.number, line=next_line.text)
                self._advance()
                elif_body = self._parse_page_statements(indent)
                elifs.append(ElifBlock(condition=condition_expr, body=elif_body))
                continue
            if stripped.startswith('else:'):
                self._advance()
                else_body = self._parse_page_statements(indent)
                break
            break
        return IfBlock(condition=condition, body=body, elifs=elifs, else_body=else_body)

    def _parse_for_loop(self, line: _Line, indent: int) -> ForLoop:
        match = re.match(r'^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(dataset|table|frame)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$', line.text)
        if not match:
            raise self._error('Expected: for name in dataset foo:', line)
        loop_var = match.group(1)
        source_kind = match.group(2)
        source_name = match.group(3)
        self._advance()
        body = self._parse_page_statements(indent)
        return ForLoop(loop_var=loop_var, source_kind=source_kind, source_name=source_name, body=body)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _parse_kv_block(self, parent_indent: int) -> dict[str, str]:
        entries: dict[str, str] = {}
        while True:
            line = self._peek()
            if line is None:
                break
            stripped = line.text.strip()
            indent = self._indent(line.text)
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            if indent <= parent_indent:
                break
            match = re.match(r'^([A-Za-z0-9_\-\s]+):\s*(.+)$', stripped)
            if not match:
                raise self._error("Expected 'key: value' inside block", line)
            key = match.group(1).strip()
            value = match.group(2).strip()
            entries[key] = value
            self._advance()
        return entries

    def _ensure_app(self, line: _Line) -> None:
        if self._app is None:
            fallback_name = self._module_name or self._module_name_override
            fallback = fallback_name.split('.')[-1] if fallback_name else 'app'
            self._app = App(name=fallback)

    def _peek(self) -> Optional[_Line]:
        while self._cursor < len(self._lines):
            line = self._lines[self._cursor]
            number = self._cursor + 1
            return _Line(text=line, number=number)
        return None

    def _advance(self) -> None:
        self._cursor += 1

    @staticmethod
    def _indent(text: str) -> int:
        return len(text) - len(text.lstrip(' '))

    def _error(self, message: str, line: _Line) -> N3SyntaxError:
        return N3SyntaxError(
            f"Syntax error: {message}",
            path=self._path or None,
            line=line.number,
            code="SYNTAX_GRAMMAR",
            hint=line.text.strip() or None,
        )

    def _unsupported(self, line: _Line, feature: str) -> None:
        location = f"{self._path}:{line.number}" if self._path else f"line {line.number}"
        raise GrammarUnsupportedError(f"Unsupported {feature} near {location}")
