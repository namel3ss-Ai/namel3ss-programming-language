from __future__ import annotations

from lsprotocol.types import HoverParams, Position, TextDocumentIdentifier

from namel3ss.lsp.workspace import WorkspaceIndex

from tests.lsp.conftest import open_document


def test_hover_shows_dataset_summary(workspace: WorkspaceIndex) -> None:
    document = open_document(workspace, "dashboard.n3")
    line, column = _position_of_word(document.text, "orders")
    params = HoverParams(
        text_document=TextDocumentIdentifier(uri=document.uri),
        position=Position(line=line, character=column + 1),
    )
    hover = workspace.hover(params)
    assert hover is not None
    assert "Dataset from" in hover.contents.value


def test_hover_includes_frame_schema(workspace: WorkspaceIndex) -> None:
    document = open_document(workspace, "dashboard.n3")
    line, column = _position_of_word(document.text, "orders_summary")
    params = HoverParams(
        text_document=TextDocumentIdentifier(uri=document.uri),
        position=Position(line=line, character=column + len("orders_summary")),
    )
    hover = workspace.hover(params)
    assert hover is not None
    assert "Schema" in hover.contents.value
    assert "status" in hover.contents.value


def _position_of_word(text: str, token: str) -> tuple[int, int]:
    for idx, line in enumerate(text.splitlines()):
        col = line.find(token)
        if col != -1:
            return idx, col
    raise AssertionError(f"Token '{token}' not found")
