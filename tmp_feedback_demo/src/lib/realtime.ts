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
