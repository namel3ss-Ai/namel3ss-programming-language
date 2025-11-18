"""Handler registration helpers."""

from __future__ import annotations

from . import completion, definition, diagnostics, formatting, hover, symbols


def register_all(server) -> None:
    diagnostics.register(server)
    completion.register(server)
    hover.register(server)
    definition.register(server)
    symbols.register(server)
    formatting.register(server)


__all__ = ["register_all"]
