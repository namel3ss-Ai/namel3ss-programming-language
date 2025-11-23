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
        self._initialize_optimizations()
        self._initialize_enhanced_features()

    def _initialize_enhanced_features(self) -> None:
        """Enable enhanced IDE features like improved completions."""
        try:
            from .enhanced_completion import enhance_workspace_completions
            enhance_workspace_completions(self.workspace_index)
            self.logger.info("Enhanced completion provider enabled")
        except ImportError:
            self.logger.warning("Enhanced completions not available")
        
        try:
            from .code_actions import enhance_lsp_with_code_actions
            enhance_lsp_with_code_actions(self)
            self.logger.info("Code actions provider enabled")
        except ImportError:
            self.logger.warning("Code actions not available")
        
        try:
            from .symbol_navigation import enhance_lsp_with_navigation
            enhance_lsp_with_navigation(self)
            self.logger.info("Symbol navigation provider enabled")
        except ImportError:
            self.logger.warning("Symbol navigation not available")

    def _initialize_optimizations(self) -> None:
        """Enable parser optimizations for better IDE performance."""
        try:
            from namel3ss.parser import enable_parser_cache
            enable_parser_cache(max_entries=100)  # Larger cache for IDE usage
            self.logger.info("Parser caching enabled for IDE performance")
        except ImportError:
            self.logger.warning("Parser optimizations not available")

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
