"""Document level state tracking for the Namel3ss language server."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from lsprotocol.types import Diagnostic, DiagnosticSeverity, Position, Range
from pygls.uris import to_fs_path

from namel3ss.ast import App, Dataset, Frame, Page, VariableAssignment
from namel3ss.ast.datasets import DatasetSchemaField
from namel3ss.ast.frames import FrameColumn
from namel3ss.ast.pages import ForLoop, IfBlock, PageStatement, WhileLoop
from namel3ss.errors import N3Error, N3SyntaxError, N3TypeError
from namel3ss.parser import Parser, enable_parser_cache, disable_parser_cache
from namel3ss.types import check_module

from .protocol import ColumnInfo, CompletionContext, IndexedSymbol, SymbolLocation, SymbolType

_KEYWORD_PATTERNS = {
    SymbolType.APP: "app",
    SymbolType.DATASET: "dataset",
    SymbolType.FRAME: "frame",
    SymbolType.PAGE: "page",
    SymbolType.MODEL: "model",
    SymbolType.PROMPT: "prompt",
    SymbolType.TEMPLATE: "template",
    SymbolType.CHAIN: "chain",
    SymbolType.EXPERIMENT: "experiment",
    SymbolType.EVALUATOR: "evaluator",
    SymbolType.METRIC: "metric",
    SymbolType.GUARDRAIL: "guardrail",
}

_COLUMN_TRIGGER = re.compile(r"([A-Za-z_][\w]*)\.$")


def _default_symbol_bucket() -> Dict[str, IndexedSymbol]:
    return {}


@dataclass
class DocumentIndex:
    """Holds the derived symbols for a parsed document."""

    by_type: Dict[SymbolType, Dict[str, IndexedSymbol]] = field(
        default_factory=lambda: {symbol_type: _default_symbol_bucket() for symbol_type in SymbolType}
    )
    columns: Dict[str, List[ColumnInfo]] = field(default_factory=dict)
    page_variables: Dict[str, Dict[str, IndexedSymbol]] = field(default_factory=dict)

    def all_symbols(self) -> Iterable[IndexedSymbol]:
        for mapping in self.by_type.values():
            for symbol in mapping.values():
                yield symbol

    def upsert(self, symbol: IndexedSymbol) -> None:
        self.by_type.setdefault(symbol.type, {})[symbol.name] = symbol

    def lookup(self, name: str) -> Optional[IndexedSymbol]:
        lowered = name.lower()
        for mapping in self.by_type.values():
            for symbol in mapping.values():
                if symbol.name.lower() == lowered:
                    return symbol
        return None


@dataclass
class DocumentState:
    """Tracks the parsed structure and cached facts for an open text document."""

    uri: str
    text: str
    version: int
    root_path: Optional[Path] = None
    path: Path = field(init=False)
    lines: List[str] = field(init=False)
    _line_offsets: List[int] = field(default_factory=list)
    module: Optional[object] = None
    app: Optional[App] = None
    diagnostics: List[Diagnostic] = field(default_factory=list)
    symbols: DocumentIndex = field(default_factory=DocumentIndex)
    _syntax_error: Optional[N3SyntaxError] = None
    _type_error: Optional[N3TypeError] = None
    _page_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = self._resolve_path()
        self._set_text(self.text)
        self.rebuild()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, text: str, version: int) -> List[Diagnostic]:
        self._set_text(text)
        self.version = version
        self.rebuild()
        return self.diagnostics

    def rebuild(self) -> None:
        self._syntax_error = None
        self._type_error = None
        self.module = None
        self.app = None
        self.symbols = DocumentIndex()
        self._page_ranges.clear()
        parser = Parser(self.text, path=str(self.path))
        try:
            module = parser.parse()
        except N3SyntaxError as exc:
            self._syntax_error = exc
            self.diagnostics = [self._diagnostic_from_error(exc)]
            return
        module.path = module.path or str(self.path)
        self.module = module
        self.app = self._extract_app(module)
        self._build_symbol_index()
        diagnostics: List[Diagnostic] = []
        try:
            check_module(module)
        except N3TypeError as exc:
            self._type_error = exc
            diagnostics.append(self._diagnostic_from_error(exc))
        self.diagnostics = diagnostics

    def diagnostics_for_publish(self) -> List[Diagnostic]:
        return list(self.diagnostics)

    def completion_context(self, position: Position) -> CompletionContext:
        prefix = self._current_line_prefix(position)
        token = self.word_at(position)
        scope = self.scope_for_position(position)
        scope_symbol = scope[0] if scope else None
        scope_type = scope[1] if scope else None
        match = _COLUMN_TRIGGER.search(prefix)
        if match:
            name = match.group(1)
            symbol = self.symbols.lookup(name)
            if symbol is not None:
                return CompletionContext(token=token, prefix=prefix, scope_symbol=symbol.name, scope_type=symbol.type)
        return CompletionContext(token=token, prefix=prefix, scope_symbol=scope_symbol, scope_type=scope_type)

    def scope_for_position(self, position: Position) -> Optional[Tuple[str, SymbolType]]:
        for page_name, (start, end) in self._page_ranges.items():
            if start <= position.line <= end:
                return page_name, SymbolType.PAGE
        return None

    def word_at(self, position: Position) -> str:
        if position.line >= len(self.lines):
            return ""
        line = self.lines[position.line]
        if position.character > len(line):
            return ""
        left = line[: position.character]
        right = line[position.character :]
        left_match = re.findall(r"[A-Za-z0-9_]+", left)
        right_match = re.findall(r"^[A-Za-z0-9_]+", right)
        left_part = left_match[-1] if left_match else ""
        right_part = right_match[0] if right_match else ""
        return left_part + right_part

    def hover_text(self, name: str) -> Optional[str]:
        symbol = self.symbols.lookup(name)
        if symbol is None:
            return None
        lines: List[str] = [f"**{symbol.name}**"]
        if symbol.detail:
            lines.append(symbol.detail)
        columns = self.symbols.columns.get(symbol.name)
        if columns:
            lines.append("")
            lines.append("Schema:")
            for column in columns[:20]:
                lines.append(column.format_signature())
            if len(columns) > 20:
                lines.append("…")
        return "\n".join(lines)

    def definition_for(self, name: str) -> Optional[SymbolLocation]:
        symbol = self.symbols.lookup(name)
        return symbol.location if symbol else None

    def column_owner(self, name: str) -> Optional[str]:
        symbol = self.symbols.lookup(name)
        if symbol and symbol.type in {SymbolType.DATASET, SymbolType.FRAME}:
            return symbol.name
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_path(self) -> Path:
        try:
            return Path(to_fs_path(self.uri))
        except ValueError:
            return Path(self.uri)

    def _set_text(self, text: str) -> None:
        self.text = text
        self.lines = text.splitlines()
        if not self.lines:
            self.lines = [""]
        self._recompute_line_offsets()

    def offset_at(self, position: Position) -> int:
        line_index = min(max(position.line, 0), len(self.lines) - 1)
        start_offset = 0
        if self._line_offsets and line_index < len(self._line_offsets):
            start_offset = self._line_offsets[line_index]
        else:
            for idx in range(line_index):
                start_offset += len(self.lines[idx]) + 1
        column = min(max(position.character, 0), len(self.lines[line_index]))
        return start_offset + column

    def _extract_app(self, module: object) -> Optional[App]:
        if not hasattr(module, "body"):
            return None
        for node in getattr(module, "body", []):
            if isinstance(node, App):
                return node
        return None

    def _diagnostic_from_error(self, error: N3Error) -> Diagnostic:
        line = max((error.line or 1) - 1, 0)
        column = max((error.column or 1) - 1, 0)
        start = Position(line=line, character=column)
        end = Position(line=line, character=column + 1)
        return Diagnostic(
            range=Range(start=start, end=end),
            message=error.message,
            severity=DiagnosticSeverity.Error,
            source="namel3ss",
            code=error.code,
        )

    def _build_symbol_index(self) -> None:
        self.symbols = DocumentIndex()
        self._page_ranges = {}
        if self.app is None:
            return
        self._maybe_register(SymbolType.APP, self.app.name, detail="Application")
        self._index_datasets(self.app.datasets)
        self._index_frames(self.app.frames)
        self._index_pages(self.app.pages)
        if getattr(self.app, "models", None):
            for model in self.app.models:
                self._maybe_register(SymbolType.MODEL, model.name, detail=f"Model {model.model_type}")
        if getattr(self.app, "prompts", None):
            for prompt in self.app.prompts:
                detail = f"Prompt inputs: {len(prompt.input_fields)}"
                self._maybe_register(SymbolType.PROMPT, prompt.name, detail=detail)
        if getattr(self.app, "templates", None):
            for template in self.app.templates:
                self._maybe_register(SymbolType.TEMPLATE, template.name, detail="Template")
        if getattr(self.app, "chains", None):
            for chain in self.app.chains:
                self._maybe_register(SymbolType.CHAIN, chain.name, detail="Chain")
        if getattr(self.app, "experiments", None):
            for experiment in self.app.experiments:
                self._maybe_register(SymbolType.EXPERIMENT, experiment.name, detail="Experiment")
        if getattr(self.app, "evaluators", None):
            for evaluator in self.app.evaluators:
                self._maybe_register(SymbolType.EVALUATOR, evaluator.name, detail="Evaluator")
        if getattr(self.app, "metrics", None):
            for metric in self.app.metrics:
                self._maybe_register(SymbolType.METRIC, metric.name, detail="Metric")
        if getattr(self.app, "guardrails", None):
            for guardrail in self.app.guardrails:
                self._maybe_register(SymbolType.GUARDRAIL, guardrail.name, detail="Guardrail")
        if getattr(self.app, "variables", None):
            for assignment in self.app.variables:
                self._maybe_register(SymbolType.VARIABLE, assignment.name, detail="App variable")

    def _index_datasets(self, datasets: Sequence[Dataset]) -> None:
        for dataset in datasets:
            detail = f"Dataset from {dataset.source_type} {dataset.source}"
            self._maybe_register(SymbolType.DATASET, dataset.name, detail=detail)
            columns = self._dataset_columns(dataset)
            if columns:
                self.symbols.columns[dataset.name] = columns

    def _index_frames(self, frames: Sequence[Frame]) -> None:
        for frame in frames:
            source = f"{frame.source_type} {frame.source}" if frame.source else frame.source_type
            self._maybe_register(SymbolType.FRAME, frame.name, detail=f"Frame from {source}")
            columns = [ColumnInfo(name=column.name, dtype=column.dtype, nullable=column.nullable) for column in frame.columns]
            if columns:
                self.symbols.columns[frame.name] = columns

    def _index_pages(self, pages: Sequence[Page]) -> None:
        for page in pages:
            detail = f"Route {page.route}"
            location = self._maybe_register(SymbolType.PAGE, page.name, detail=detail)
            if location is not None:
                line_range = self._block_extent(location.range.start.line)
                self._page_ranges[page.name] = line_range
            variables = self._collect_page_variables(page)
            if variables:
                bucket = self.symbols.page_variables.setdefault(page.name, {})
                for variable_name in variables:
                    bucket[variable_name] = self._make_symbol(
                        name=variable_name,
                        symbol_type=SymbolType.PAGE_VARIABLE,
                        detail=f"Variable on page {page.name}",
                    )
                    self.symbols.upsert(bucket[variable_name])

    def _collect_page_variables(self, page: Page) -> Set[str]:
        found: Set[str] = set()
        for statement in page.statements:
            found.update(self._collect_statement_variables(statement))
        return found

    def _collect_statement_variables(self, statement: PageStatement) -> Set[str]:
        results: Set[str] = set()
        if isinstance(statement, VariableAssignment):
            results.add(statement.name)
            return results
        if isinstance(statement, IfBlock):
            for inner in statement.body:
                results.update(self._collect_statement_variables(inner))
            for branch in statement.elifs:
                for inner in branch.body:
                    results.update(self._collect_statement_variables(inner))
            if statement.else_body:
                for inner in statement.else_body:
                    results.update(self._collect_statement_variables(inner))
        if isinstance(statement, ForLoop):
            results.add(statement.loop_var)
            for inner in statement.body:
                results.update(self._collect_statement_variables(inner))
        if isinstance(statement, WhileLoop):
            for inner in statement.body:
                results.update(self._collect_statement_variables(inner))
        return results

    def _dataset_columns(self, dataset: Dataset) -> List[ColumnInfo]:
        columns: List[ColumnInfo] = []
        for field in getattr(dataset, "schema", []) or []:
            if isinstance(field, DatasetSchemaField):
                columns.append(ColumnInfo(name=field.name, dtype=field.dtype, nullable=field.nullable))
        return columns

    def _maybe_register(self, symbol_type: SymbolType, name: str, *, detail: Optional[str] = None) -> Optional[SymbolLocation]:
        location = self._locate_symbol(symbol_type, name)
        symbol = self._make_symbol(name=name, symbol_type=symbol_type, detail=detail, location=location)
        self.symbols.upsert(symbol)
        return symbol.location

    def _make_symbol(
        self,
        *,
        name: str,
        symbol_type: SymbolType,
        detail: Optional[str] = None,
        location: Optional[SymbolLocation] = None,
    ) -> IndexedSymbol:
        if location is None:
            start = Position(line=0, character=0)
            location = SymbolLocation(uri=self.uri, range=Range(start=start, end=start))
        return IndexedSymbol(name=name, type=symbol_type, location=location, detail=detail)

    def _locate_symbol(self, symbol_type: SymbolType, name: str) -> Optional[SymbolLocation]:
        keyword = _KEYWORD_PATTERNS.get(symbol_type)
        if not keyword:
            return None
        target = f"{keyword} \"{name}\""
        range_ = self._find_range_for_text(target)
        if range_ is None:
            fallback = f"{keyword} {name}"
            range_ = self._find_range_for_text(fallback)
        if range_ is None:
            return None
        return SymbolLocation(uri=self.uri, range=range_)

    def _find_range_for_text(self, snippet: str) -> Optional[Range]:
        lowered_snippet = snippet.lower()
        for idx, line in enumerate(self.lines):
            lowered_line = line.lower()
            column = lowered_line.find(lowered_snippet)
            if column != -1:
                start = Position(line=idx, character=column)
                end = Position(line=idx, character=column + len(snippet))
                return Range(start=start, end=end)
        return None

    def _block_extent(self, header_line: int) -> Tuple[int, int]:
        base_indent = self._indent(self.lines[header_line])
        end_line = header_line
        for idx in range(header_line + 1, len(self.lines)):
            text = self.lines[idx]
            stripped = text.strip()
            if not stripped or stripped.startswith('#'):
                continue
            indent = self._indent(text)
            if indent <= base_indent:
                break
            end_line = idx
        return header_line, end_line

    def _indent(self, line: str) -> int:
        return len(line) - len(line.lstrip(' '))

    def _current_line_prefix(self, position: Position) -> str:
        if position.line >= len(self.lines):
            return ""
        line = self.lines[position.line]
        return line[: position.character]

    def _recompute_line_offsets(self) -> None:
        offsets: List[int] = [0]
        text = self.text
        idx = 0
        length = len(text)
        while idx < length:
            char = text[idx]
            if char == '\r':
                next_idx = idx + 1
                if next_idx < length and text[next_idx] == '\n':
                    offsets.append(next_idx + 1)
                    idx = next_idx + 1
                else:
                    offsets.append(idx + 1)
                    idx += 1
            elif char == '\n':
                offsets.append(idx + 1)
                idx += 1
            else:
                idx += 1
        if len(offsets) < len(self.lines):
            offsets.extend([len(text)] * (len(self.lines) - len(offsets)))
        self._line_offsets = offsets


__all__ = ["DocumentState", "DocumentIndex"]
