from __future__ import annotations

from namel3ss.lsp.workspace import WorkspaceIndex

from tests.lsp.conftest import DATA_DIR, open_document


def test_valid_program_produces_no_diagnostics(workspace: WorkspaceIndex) -> None:
    _, diags = open_document_with_diagnostics(workspace, "dashboard.n3")
    assert diags == []


def test_reports_syntax_errors(workspace: WorkspaceIndex) -> None:
    _, diags = open_document_with_diagnostics(workspace, "syntax_error.n3")
    assert diags, "Expected syntax diagnostics"
    assert "syntax" in diags[0].message.lower()


def test_reports_type_errors(workspace: WorkspaceIndex) -> None:
    _, diags = open_document_with_diagnostics(workspace, "type_error.n3")
    assert diags, "Expected type diagnostics"
    assert "unknown dataset" in diags[0].message.lower()


def open_document_with_diagnostics(workspace: WorkspaceIndex, filename: str):
    from lsprotocol.types import TextDocumentItem

    path = DATA_DIR / filename
    text = path.read_text(encoding="utf-8")
    item = TextDocumentItem(
        uri=path.resolve().as_uri(),
        language_id="namel3ss",
        version=1,
        text=text,
    )
    diagnostics = workspace.did_open(item)
    return item, diagnostics
