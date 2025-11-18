"""Document/workspace symbol handlers."""

from __future__ import annotations

from lsprotocol.types import DocumentSymbolParams, WorkspaceSymbolParams

def register(server) -> None:
    workspace = server.workspace_index

    @server.feature("textDocument/documentSymbol")
    async def _doc_symbols(ls, params: DocumentSymbolParams):
        return workspace.document_symbols(params.text_document)

    @server.feature("workspace/symbol")
    async def _workspace_symbols(ls, params: WorkspaceSymbolParams):
        return workspace.workspace_symbols(params)
