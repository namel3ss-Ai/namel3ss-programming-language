"""Diagnostics and lifecycle handlers."""

from __future__ import annotations

from typing import Optional

from lsprotocol.types import (
    DidChangeTextDocumentParams,
    DidCloseTextDocumentParams,
    DidOpenTextDocumentParams,
    DidSaveTextDocumentParams,
)

from ..workspace import WorkspaceIndex


def register(server) -> None:
    workspace: WorkspaceIndex = server.workspace_index

    @server.feature("textDocument/didOpen")
    async def _did_open(ls, params: DidOpenTextDocumentParams) -> None:
        diagnostics = workspace.did_open(params.text_document)
        ls.publish_diagnostics(params.text_document.uri, diagnostics)

    @server.feature("textDocument/didChange")
    async def _did_change(ls, params: DidChangeTextDocumentParams) -> None:
        if not params.content_changes:
            return
        diagnostics = workspace.did_change(
            params.text_document.uri,
            params.text_document.version or 0,
            params.content_changes,
        )
        ls.publish_diagnostics(params.text_document.uri, diagnostics)

    @server.feature("textDocument/didClose")
    async def _did_close(ls, params: DidCloseTextDocumentParams) -> None:
        workspace.did_close(params.text_document.uri)
        ls.publish_diagnostics(params.text_document.uri, [])

    @server.feature("textDocument/didSave")
    async def _did_save(ls, params: DidSaveTextDocumentParams) -> None:
        workspace.refresh_index()
