"""pygls based Language Server entrypoint."""

from __future__ import annotations

import os
from typing import Optional

from lsprotocol.types import InitializedParams
from pygls.server import LanguageServer

from namel3ss import __version__

from .handlers import register_all
from .workspace import WorkspaceIndex


class Namel3ssLanguageServer(LanguageServer):
    """Concrete LanguageServer with Namel3ss-specific state."""

    def __init__(self) -> None:
        super().__init__(name="namel3ss-lsp", version=__version__)
        self.workspace_index = WorkspaceIndex()
        register_all(self)
        self._register_lifecycle_handlers()

    def _register_lifecycle_handlers(self) -> None:
        workspace = self.workspace_index

        @self.feature("initialized")
        async def _on_initialized(ls: "Namel3ssLanguageServer", params: InitializedParams) -> None:  # noqa: ARG001
            workspace.set_root(ls.workspace.root_uri)
            workspace.refresh_index()
            ls.logger.info("Workspace index initialised at %s", workspace.root_path)


def create_server() -> Namel3ssLanguageServer:
    return Namel3ssLanguageServer()


def main() -> None:
    server = create_server()
    server.logger.info("Starting Namel3ss LSP (pid=%s)", os.getpid())
    server.start_io()


if __name__ == "__main__":  # pragma: no cover
    main()
