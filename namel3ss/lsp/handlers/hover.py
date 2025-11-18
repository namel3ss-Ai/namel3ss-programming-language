"""Hover handler."""

from __future__ import annotations

from lsprotocol.types import HoverParams

def register(server) -> None:
    workspace = server.workspace_index

    @server.feature("textDocument/hover")
    async def _hover(ls, params: HoverParams):
        return workspace.hover(params)
