"""
React hooks for realtime functionality.

This module contains custom React hooks for handling realtime WebSocket
connections to namel3ss backend pages.
"""

import textwrap
from pathlib import Path

from .utils import write_file


def write_realtime_hook(lib_dir: Path) -> None:
    """Generate realtime.ts with useRealtimePage hook for WebSocket connections."""
    content = textwrap.dedent(
        """
        import { useEffect } from "react";
        import type { PageDefinition } from "./n3Client";

        export function useRealtimePage(definition: PageDefinition) {
          useEffect(() => {
            if (!definition.realtime || typeof window === "undefined") {
              return;
            }
            const slug = definition.slug;
            const protocol = window.location.protocol === "https:" ? "wss" : "ws";
            const host = window.location.host;
            const socket = new WebSocket(`${protocol}://${host}/ws/pages/${slug}`);
            socket.onmessage = (event) => {
              console.debug("namel3ss realtime", slug, event.data);
            };
            socket.onerror = (event) => {
              console.warn("namel3ss realtime error", event);
            };
            return () => {
              socket.close();
            };
          }, [definition.realtime, definition.slug]);
        }
        """
    ).strip() + "\n"
    write_file(lib_dir / "realtime.ts", content)
