from __future__ import annotations

from pathlib import Path

from lsprotocol.types import Position, TextDocumentIdentifier, TextDocumentPositionParams

from namel3ss.lsp.workspace import WorkspaceIndex

from tests.lsp.conftest import DATA_DIR, open_document


def test_dataset_completions_include_local_symbols(workspace: WorkspaceIndex) -> None:
    document = open_document(workspace, "dashboard.n3")
    line, character = _position_of("show table \"Recent\" from dataset ", document.text)
    params = TextDocumentPositionParams(
        text_document=TextDocumentIdentifier(uri=document.uri),
        position=Position(line=line, character=character),
    )
    completions = workspace.completion(params)
    labels = {item.label for item in completions.items}
    assert "orders" in labels


def test_column_completion_uses_frame_schema(workspace: WorkspaceIndex) -> None:
    document = open_document(workspace, "dashboard.n3")
    line, character = _position_of("orders_summary.", document.text)
    params = TextDocumentPositionParams(
        text_document=TextDocumentIdentifier(uri=document.uri),
        position=Position(line=line, character=character + len("orders_summary.")),
    )
    completions = workspace.completion(params)
    labels = {item.label for item in completions.items}
    assert "status" in labels
    assert "total" in labels


def _position_of(snippet: str, text: str) -> tuple[int, int]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        col = line.find(snippet)
        if col != -1:
            return idx, col + len(snippet)
    raise AssertionError(f"Snippet '{snippet}' not found in document")
