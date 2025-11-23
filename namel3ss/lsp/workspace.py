"""Workspace level indexing utilities for the Namel3ss language server."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from lsprotocol.types import (
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    Diagnostic,
    DocumentFormattingParams,
    DocumentSymbol,
    Hover,
    HoverParams,
    Location,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
    SymbolInformation,
    SymbolKind,
    TextDocumentIdentifier,
    TextDocumentItem,
    TextDocumentContentChangeEvent,
    TextDocumentPositionParams,
    TextEdit,
    WorkspaceSymbolParams,
)
from pygls.uris import from_fs_path, to_fs_path

from .protocol import ColumnInfo, CompletionContext, IndexedSymbol, SymbolType
from .state import DocumentIndex, DocumentState

_SYMBOL_KIND_MAP = {
    SymbolType.APP: SymbolKind.Module,
    SymbolType.DATASET: SymbolKind.Object,
    SymbolType.FRAME: SymbolKind.Struct,
    SymbolType.PAGE: SymbolKind.Class,
    SymbolType.MODEL: SymbolKind.Interface,
    SymbolType.PROMPT: SymbolKind.Function,
    SymbolType.TEMPLATE: SymbolKind.String,
    SymbolType.CHAIN: SymbolKind.Event,
    SymbolType.EXPERIMENT: SymbolKind.Namespace,
    SymbolType.EVALUATOR: SymbolKind.Property,
    SymbolType.METRIC: SymbolKind.Property,
    SymbolType.GUARDRAIL: SymbolKind.Property,
    SymbolType.VARIABLE: SymbolKind.Variable,
    SymbolType.PAGE_VARIABLE: SymbolKind.Variable,
}

_SYMBOL_COMPLETION_KIND = {
    SymbolType.APP: CompletionItemKind.Module,
    SymbolType.DATASET: CompletionItemKind.Struct,
    SymbolType.FRAME: CompletionItemKind.Struct,
    SymbolType.PAGE: CompletionItemKind.Class,
    SymbolType.MODEL: CompletionItemKind.Interface,
    SymbolType.PROMPT: CompletionItemKind.Function,
    SymbolType.TEMPLATE: CompletionItemKind.Snippet,
    SymbolType.CHAIN: CompletionItemKind.Function,
    SymbolType.EXPERIMENT: CompletionItemKind.Class,
    SymbolType.EVALUATOR: CompletionItemKind.Property,
    SymbolType.METRIC: CompletionItemKind.Property,
    SymbolType.GUARDRAIL: CompletionItemKind.Property,
    SymbolType.VARIABLE: CompletionItemKind.Variable,
    SymbolType.PAGE_VARIABLE: CompletionItemKind.Variable,
}

_KEYWORD_COMPLETION_KIND = CompletionItemKind.Keyword

_KEYWORD_COMPLETIONS = (
    "app",
    "dataset",
    "frame",
    "page",
    "model",
    "prompt",
    "template",
    "chain",
    "experiment",
    "evaluator",
    "metric",
    "guardrail",
    "theme",
    "enable crud",
    "ai model",
    "memory",
    "connector",
    "page variable",
    "table",
    "chart",
    "form",
    "set",
    "for",
    "if",
)


@dataclass
class DocumentHandle:
    uri: str
    document: DocumentState


class WorkspaceIndex:
    """Loads and caches information for every .n3 file in the workspace."""

    def __init__(self, root_uri: Optional[str] = None) -> None:
        self.logger = logging.getLogger("namel3ss.lsp.workspace")
        self.root_uri = root_uri
        self.root_path = self._resolve_root(root_uri)
        self._open_documents: Dict[str, DocumentState] = {}
        self._snapshots: Dict[str, DocumentState] = {}
        self._symbols_by_uri: Dict[str, List[IndexedSymbol]] = {}

    def set_root(self, root_uri: Optional[str]) -> None:
        self.root_uri = root_uri
        self.root_path = self._resolve_root(root_uri)

    # ------------------------------------------------------------------
    # Document lifecycle
    # ------------------------------------------------------------------
    def did_open(self, item: TextDocumentItem) -> List[Diagnostic]:
        document = DocumentState(uri=item.uri, text=item.text, version=item.version, root_path=self.root_path)
        
        # Enhance document with improved diagnostics
        try:
            from .enhanced_diagnostics import enhance_document_diagnostics
            enhance_document_diagnostics(document)
        except ImportError:
            pass  # Fall back to basic diagnostics
        
        self._open_documents[item.uri] = document
        self._update_symbols(item.uri, document)
        return document.diagnostics_for_publish()

    def did_change(
        self,
        uri: str,
        version: int,
        changes: Sequence[TextDocumentContentChangeEvent],
    ) -> List[Diagnostic]:
        document = self._open_documents.get(uri)
        if document is None:
            initial_text = self._read_document_from_fs(uri)
            document = DocumentState(uri=uri, text=initial_text, version=version, root_path=self.root_path)
            
            # Enhance document with improved diagnostics
            try:
                from .enhanced_diagnostics import enhance_document_diagnostics
                enhance_document_diagnostics(document)
            except ImportError:
                pass  # Fall back to basic diagnostics
                
            self._open_documents[uri] = document
        next_text = self._apply_content_changes(document, changes)
        diagnostics = document.update(next_text, version)
        self._update_symbols(uri, document)
        return diagnostics

    def did_close(self, uri: str) -> None:
        document = self._open_documents.pop(uri, None)
        if document is not None:
            self._snapshots[uri] = document
        self._update_symbols(uri, document)

    def document(self, uri: str) -> Optional[DocumentState]:
        return self._open_documents.get(uri) or self._snapshots.get(uri)

    def refresh_index(self) -> None:
        if not self.root_path.exists():
            return
        for path in self.root_path.rglob("*.n3"):
            uri = from_fs_path(str(path))
            if uri in self._open_documents:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:  # pragma: no cover - filesystem errors
                self.logger.debug("Failed to read %s: %s", path, exc)
                continue
            snapshot = DocumentState(uri=uri, text=text, version=0, root_path=self.root_path)
            
            # Enhance snapshot with improved diagnostics
            try:
                from .enhanced_diagnostics import enhance_document_diagnostics
                enhance_document_diagnostics(snapshot)
            except ImportError:
                pass  # Fall back to basic diagnostics
            
            self._snapshots[uri] = snapshot
            self._update_symbols(uri, snapshot)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def diagnostics(self, uri: str) -> List[Diagnostic]:
        document = self.document(uri)
        if document is None:
            return []
        return document.diagnostics_for_publish()

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------
    def completion(self, params: TextDocumentPositionParams) -> CompletionList:
        document = self.document(params.text_document.uri)
        if document is None:
            return CompletionList(is_incomplete=False, items=[])
        context = document.completion_context(params.position)
        items: List[CompletionItem] = []
        items.extend(self._keyword_completions(context))
        items.extend(self._symbol_completions(document, context))
        items.extend(self._column_completions(document, context))
        return CompletionList(is_incomplete=False, items=items)

    def _keyword_completions(self, context: CompletionContext) -> List[CompletionItem]:
        prefix = context.token or ""
        results: List[CompletionItem] = []
        for keyword in _KEYWORD_COMPLETIONS:
            if not prefix or keyword.startswith(prefix):
                results.append(
                    CompletionItem(
                        label=keyword,
                        kind=_KEYWORD_COMPLETION_KIND,
                        detail="Keyword",
                        insert_text=keyword,
                    )
                )
        return results

    def _symbol_completions(self, document: DocumentState, context: CompletionContext) -> List[CompletionItem]:
        candidates: List[IndexedSymbol] = []
        scope_type = context.scope_type
        if scope_type == SymbolType.PAGE:
            page_name = context.scope_symbol
            page_vars = document.symbols.page_variables.get(page_name or "", {})
            candidates.extend(page_vars.values())
        desired_types = {
            SymbolType.DATASET,
            SymbolType.FRAME,
            SymbolType.MODEL,
            SymbolType.PROMPT,
            SymbolType.TEMPLATE,
            SymbolType.CHAIN,
            SymbolType.EXPERIMENT,
            SymbolType.EVALUATOR,
            SymbolType.METRIC,
            SymbolType.GUARDRAIL,
            SymbolType.VARIABLE,
            SymbolType.PAGE_VARIABLE,
        }
        candidates.extend(self._iter_symbols(desired_types))
        prefix = (context.token or "").lower()
        items: List[CompletionItem] = []
        seen: Set[str] = set()
        for symbol in candidates:
            if prefix and not symbol.name.lower().startswith(prefix):
                continue
            if symbol.name in seen:
                continue
            seen.add(symbol.name)
            items.append(
                CompletionItem(
                    label=symbol.name,
                    kind=_SYMBOL_COMPLETION_KIND.get(symbol.type, CompletionItemKind.Text),
                    detail=symbol.detail,
                    sort_text=f"1_{symbol.name.lower()}",
                )
            )
        return items

    def _column_completions(self, document: DocumentState, context: CompletionContext) -> List[CompletionItem]:
        owner = context.scope_symbol
        if not owner:
            return []
        columns = document.symbols.columns.get(owner)
        if columns is None:
            owner_symbol = self.resolve_symbol(owner, {SymbolType.DATASET, SymbolType.FRAME})
            if owner_symbol is not None:
                columns = document.symbols.columns.get(owner_symbol.name)
                if columns is None:
                    columns = self._columns_from_any(owner_symbol)
        if not columns:
            return []
        return [
            CompletionItem(
                label=column.name,
                kind=CompletionItemKind.Field,
                detail=column.format_signature(),
                sort_text=f"0_{column.name.lower()}",
            )
            for column in columns
        ]

    def _columns_from_any(self, symbol: IndexedSymbol) -> List[ColumnInfo]:
        document = self.document(symbol.location.uri)
        if document is None:
            return []
        return document.symbols.columns.get(symbol.name, [])

    # ------------------------------------------------------------------
    # Hover
    # ------------------------------------------------------------------
    def hover(self, params: HoverParams) -> Optional[Hover]:
        document = self.document(params.text_document.uri)
        if document is None:
            return None
        name = document.word_at(params.position)
        if not name:
            return None
        symbol = self.resolve_symbol(name)
        if symbol is None:
            return None
        hover_text = document.hover_text(symbol.name) or symbol.detail
        if not hover_text:
            return None
        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=hover_text), range=symbol.location.range)

    # ------------------------------------------------------------------
    # Definition
    # ------------------------------------------------------------------
    def definition(self, params: TextDocumentPositionParams) -> Optional[Location]:
        document = self.document(params.text_document.uri)
        if document is None:
            return None
        name = document.word_at(params.position)
        if not name:
            return None
        symbol = self.resolve_symbol(name)
        if symbol is None:
            return None
        return Location(uri=symbol.location.uri, range=symbol.location.range)

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------
    def document_symbols(self, identifier: TextDocumentIdentifier) -> List[DocumentSymbol]:
        document = self.document(identifier.uri)
        if document is None:
            return []
        symbols: List[DocumentSymbol] = []
        for symbol in document.symbols.all_symbols():
            kind = _SYMBOL_KIND_MAP.get(symbol.type, SymbolKind.Object)
            node = DocumentSymbol(
                name=symbol.name,
                detail=symbol.detail,
                kind=kind,
                range=symbol.location.range,
                selection_range=symbol.location.range,
                children=None,
            )
            symbols.append(node)
        return symbols

    def workspace_symbols(self, params: WorkspaceSymbolParams) -> List[SymbolInformation]:
        query = (params.query or "").lower()
        results: List[SymbolInformation] = []
        for symbol in self._iter_symbols():
            if query and query not in symbol.name.lower():
                continue
            kind = _SYMBOL_KIND_MAP.get(symbol.type, SymbolKind.Object)
            results.append(
                SymbolInformation(
                    name=symbol.name,
                    kind=kind,
                    location=Location(uri=symbol.location.uri, range=symbol.location.range),
                    container_name=None,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    def format_document(self, params: DocumentFormattingParams) -> List[TextEdit]:
        document = self.document(params.text_document.uri)
        if document is None:
            return []
        formatted_lines = [line.rstrip().replace("\t", "    ") for line in document.lines]
        formatted_text = "\n".join(formatted_lines)
        if document.text.endswith("\n"):
            formatted_text += "\n"
        total_range = Range(
            start=Position(line=0, character=0),
            end=Position(line=len(document.lines), character=0),
        )
        return [TextEdit(range=total_range, new_text=formatted_text)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def resolve_symbol(
        self,
        name: str,
        types: Optional[Sequence[SymbolType]] = None,
    ) -> Optional[IndexedSymbol]:
        lowered = name.lower()
        for symbol in self._iter_symbols(types):
            if symbol.name.lower() == lowered:
                return symbol
        return None

    def _iter_symbols(self, types: Optional[Sequence[SymbolType]] = None) -> Iterable[IndexedSymbol]:
        allowed = set(types) if types else None
        for symbols in self._symbols_by_uri.values():
            for symbol in symbols:
                if allowed and symbol.type not in allowed:
                    continue
                yield symbol

    def _update_symbols(self, uri: str, document: Optional[DocumentState]) -> None:
        if document is None:
            self._symbols_by_uri.pop(uri, None)
            return
        symbols = list(document.symbols.all_symbols())
        self._symbols_by_uri[uri] = symbols

    def _apply_content_changes(
        self,
        document: DocumentState,
        changes: Sequence[TextDocumentContentChangeEvent],
    ) -> str:
        text = document.text
        for change in changes:
            if change.range is None:
                text = change.text
                continue
            start = document.offset_at(change.range.start)
            end = document.offset_at(change.range.end)
            text = text[:start] + change.text + text[end:]
        return text

    def _read_document_from_fs(self, uri: str) -> str:
        try:
            path = Path(to_fs_path(uri))
        except ValueError:
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _resolve_root(self, root_uri: Optional[str]) -> Path:
        if root_uri:
            try:
                return Path(to_fs_path(root_uri))
            except ValueError:
                return Path(root_uri)
        return Path.cwd()


__all__ = ["WorkspaceIndex", "DocumentHandle"]
