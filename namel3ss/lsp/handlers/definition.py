"""Definition handler."""

from __future__ import annotations

from lsprotocol.types import TextDocumentPositionParams

def register(server) -> None:
    workspace = server.workspace_index

    @server.feature("textDocument/definition")
    async def _definition(ls, params: TextDocumentPositionParams):
        return workspace.definition(params)
