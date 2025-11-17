import { useEffect, useRef, useState } from "react";
import type { PageDefinition } from "./n3Client";

export interface RealtimeEvent {
  type?: string;
  slug: string;
  dataset?: string | null;
  payload?: unknown;
  meta?: Record<string, unknown>;
}

export interface UseRealtimePageOptions {
  fallbackIntervalSeconds?: number;
  onEvent?: (event: RealtimeEvent) => void;
  onFallbackTick?: () => void;
  onConnectionChange?: (connected: boolean) => void;
}

export interface UseRealtimePageState {
  connected: boolean;
  lastEvent: RealtimeEvent | null;
  lastError: string | null;
}

function getRuntime() {
  if (typeof window === "undefined") {
    return null;
  }
  const runtime = (window as any).N3Realtime;
  if (!runtime || typeof runtime.connectPage !== "function" || typeof runtime.disconnectPage !== "function") {
    return null;
  }
  return runtime;
}

function extractRealtimeEvent(event: Event, slug: string): RealtimeEvent | null {
  if (!(event instanceof CustomEvent)) {
    return null;
  }
  const detail = event.detail as { slug?: string; event?: RealtimeEvent } | undefined;
  if (!detail || (detail.slug && detail.slug !== slug)) {
    return null;
  }
  if (detail.event && typeof detail.event === "object") {
    const payload = detail.event;
    return {
      ...payload,
      slug: payload.slug ?? slug,
    };
  }
  return { slug, type: "message" };
}

export function useRealtimePage(definition: PageDefinition, options?: UseRealtimePageOptions): UseRealtimePageState {
  const [state, setState] = useState<UseRealtimePageState>({ connected: false, lastEvent: null, lastError: null });
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const pollingRef = useRef<number | null>(null);

  const startFallback = (interval: number) => {
    if (interval <= 0 || typeof window === "undefined") {
      return;
    }
    if (pollingRef.current !== null) {
      window.clearInterval(pollingRef.current);
    }
    pollingRef.current = window.setInterval(() => {
      optionsRef.current?.onFallbackTick?.();
    }, interval * 1000);
  };

  const stopFallback = () => {
    if (pollingRef.current !== null && typeof window !== "undefined") {
      window.clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  useEffect(() => {
    if (!definition.realtime || typeof document === "undefined") {
      setState({ connected: false, lastEvent: null, lastError: null });
      return;
    }

    const runtime = getRuntime();
    if (!runtime) {
      setState({ connected: false, lastEvent: null, lastError: "Realtime runtime unavailable." });
      const fallbackSeconds = optionsRef.current?.fallbackIntervalSeconds ?? 0;
      if (fallbackSeconds > 0) {
        startFallback(fallbackSeconds);
      }
      return;
    }

    let alive = true;
    const slug = definition.slug;
    const fallbackSeconds = optionsRef.current?.fallbackIntervalSeconds ?? 0;

    const setConnected = (connected: boolean) => {
      setState((prev) => ({ ...prev, connected }));
      optionsRef.current?.onConnectionChange?.(connected);
    };

    const handleConnected: EventListener = (event) => {
      if (!alive) {
        return;
      }
      const realtimeEvent = extractRealtimeEvent(event, slug);
      if (!realtimeEvent) {
        return;
      }
      stopFallback();
      setConnected(true);
      setState((prev) => ({ ...prev, lastError: null }));
    };

    const handleDisconnected: EventListener = (event) => {
      if (!alive || !extractRealtimeEvent(event, slug)) {
        return;
      }
      setConnected(false);
      setState((prev) => ({ ...prev, lastError: "Connection lost. Retrying..." }));
      if (fallbackSeconds > 0) {
        startFallback(fallbackSeconds);
      }
    };

    const handleMessage: EventListener = (event) => {
      if (!alive) {
        return;
      }
      const realtimeEvent = extractRealtimeEvent(event, slug);
      if (!realtimeEvent) {
        return;
      }
      stopFallback();
      setConnected(true);
      setState({ connected: true, lastEvent: realtimeEvent, lastError: null });
      optionsRef.current?.onEvent?.(realtimeEvent);
    };

    const listeners: Array<[string, EventListener]> = [
      ["n3:realtime:connected", handleConnected],
      ["n3:realtime:disconnected", handleDisconnected],
      ["n3:realtime:message", handleMessage],
      ["n3:realtime:snapshot", handleMessage],
      ["n3:realtime:hydration", handleMessage],
    ];

    listeners.forEach(([name, handler]) => document.addEventListener(name, handler));

    try {
      const connectOptions = fallbackSeconds > 0 ? { fallbackInterval: fallbackSeconds } : undefined;
      runtime.connectPage(slug, connectOptions);
    } catch (error) {
      console.warn("Failed to open realtime channel", error);
      setState((prev) => ({ ...prev, lastError: "Unable to open realtime channel." }));
      if (fallbackSeconds > 0) {
        startFallback(fallbackSeconds);
      }
    }

    return () => {
      alive = false;
      stopFallback();
      try {
        runtime.disconnectPage(slug);
      } catch (error) {
        console.warn("Failed to close realtime channel", error);
      }
      listeners.forEach(([name, handler]) => document.removeEventListener(name, handler));
    };
  }, [definition.realtime, definition.slug]);

  useEffect(() => stopFallback, []);

  return state;
}
