"""Formatting handler."""

from __future__ import annotations

from lsprotocol.types import DocumentFormattingParams

def register(server) -> None:
    workspace = server.workspace_index

    @server.feature("textDocument/formatting")
    async def _format(ls, params: DocumentFormattingParams):
        return workspace.format_document(params)
