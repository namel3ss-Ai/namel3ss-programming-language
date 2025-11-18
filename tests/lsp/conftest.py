from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from lsprotocol.types import TextDocumentItem

from namel3ss.lsp.workspace import WorkspaceIndex

DATA_DIR = Path(__file__).parent / "data"


def _make_uri(path: Path) -> str:
    return path.resolve().as_uri()


@pytest.fixture()
def workspace() -> WorkspaceIndex:
    root_uri = _make_uri(DATA_DIR)
    ws = WorkspaceIndex(root_uri)
    ws.set_root(root_uri)
    ws.refresh_index()
    return ws


def open_document(workspace: WorkspaceIndex, filename: str, *, version: int = 1) -> TextDocumentItem:
    path = DATA_DIR / filename
    text = path.read_text(encoding="utf-8")
    item = TextDocumentItem(
        uri=_make_uri(path),
        language_id="namel3ss",
        version=version,
        text=text,
    )
    workspace.did_open(item)
    return item


__all__ = ["workspace", "open_document", "DATA_DIR"]
