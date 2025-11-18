from __future__ import annotations

from pathlib import Path

from lsprotocol.types import Position, TextDocumentIdentifier, TextDocumentPositionParams, WorkspaceSymbolParams

from namel3ss.lsp.workspace import WorkspaceIndex

from tests.lsp.conftest import open_document


def test_definition_locates_dataset_declaration(workspace: WorkspaceIndex) -> None:
    document = open_document(workspace, "dashboard.n3")
    line, column = _position_of_word(document.text, "orders")
    params = TextDocumentPositionParams(
        text_document=TextDocumentIdentifier(uri=document.uri),
        position=Position(line=line, character=column + 1),
    )
    location = workspace.definition(params)
    assert location is not None
    assert location.uri == document.uri


def test_workspace_symbol_search_includes_other_files(workspace: WorkspaceIndex) -> None:
    params = WorkspaceSymbolParams(query="customers")
    symbols = workspace.workspace_symbols(params)
    assert any(Path(symbol.location.uri).name == "metrics.n3" for symbol in symbols)


def _position_of_word(text: str, token: str) -> tuple[int, int]:
    for idx, line in enumerate(text.splitlines()):
        col = line.find(token)
        if col != -1:
            return idx, col
    raise AssertionError(f"Token '{token}' not found")
