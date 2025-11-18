"""Completion handlers."""

from __future__ import annotations

from lsprotocol.types import CompletionParams

def register(server) -> None:
    workspace = server.workspace_index

    @server.feature("textDocument/completion")
    async def _completion(ls, params: CompletionParams):
        return workspace.completion(params)
