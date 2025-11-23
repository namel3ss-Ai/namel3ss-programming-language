/**
 * N3 Realtime Runtime
 *
 * Provides resilient WebSocket + fallback polling to keep pages in sync with
 * backend events. Integrates with widget core hydration and renderer.
 *
 * Exposed API on window.N3Realtime:
 *  - connectPage(slug, options)
 *  - disconnectPage(slug)
 *  - applySnapshot(slug, snapshot, meta?)
 *  - applyRealtimeUpdate(event)
 */
(() => {
  'use strict';

  /** @type {any} */
  const globalScope = typeof window !== 'undefined' ? window : globalThis;
  const realtimeNs = (globalScope.N3Realtime = globalScope.N3Realtime || {});
  const widgetsNs = globalScope.N3Widgets || {};
  const connections = new Map();

  const DEFAULT_BACKOFF = [500, 1000, 2000, 5000];

  function normalizePath(path) {
    if (!path) return '';
    return path.startsWith('/') ? path : `/${path}`;
  }

  function buildWsUrl(slug, options = {}) {
    if (options.wsUrl) return options.wsUrl;
    if (typeof window === 'undefined' || !window.location) return null;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const path = options.wsPath || `/ws/pages/${encodeURIComponent(slug)}`;
    return `${protocol}//${host}${normalizePath(path)}`;
  }

  function buildPageUrl(slug, options = {}) {
    const base = options.baseUrl ? options.baseUrl.replace(/\/$/, '') : '';
    const path = options.pageUrl || `/api/pages/${slug}`;
    return `${base}${normalizePath(path)}`;
  }

  function backoff(attempt) {
    return DEFAULT_BACKOFF[Math.min(attempt, DEFAULT_BACKOFF.length - 1)];
  }

  class RealtimeClient {
    constructor(slug, options = {}) {
      this.slug = slug;
      this.options = options;
      this.socket = null;
      this.retry = 0;
      this.fallbackTimer = null;
      this.closed = false;
    }

    connect() {
      const wsUrl = buildWsUrl(this.slug, this.options);
      if (!wsUrl) {
        this.startFallback();
        return;
      }
      try {
        this.socket = new WebSocket(wsUrl);
      } catch (err) {
        console.warn('[N3Realtime] WebSocket failed, using fallback', err);
        this.startFallback();
        return;
      }
      this.socket.onopen = () => {
        this.retry = 0;
        this.stopFallback();
      };
      this.socket.onmessage = (event) => {
        try {
          const payload = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;
          realtimeNs.applyRealtimeUpdate(payload);
        } catch (err) {
          console.warn('[N3Realtime] message parse failed', err);
        }
      };
      this.socket.onclose = () => {
        if (this.closed) return;
        this.scheduleReconnect();
      };
      this.socket.onerror = () => {
        this.socket && this.socket.close();
      };
    }

    scheduleReconnect() {
      this.retry += 1;
      const delay = backoff(this.retry);
      setTimeout(() => this.connect(), delay);
    }

    startFallback() {
      if (this.fallbackTimer) return;
      const intervalMs = (this.options.fallbackInterval || 30) * 1000;
      if (!intervalMs) return;
      this.fallbackTimer = setInterval(async () => {
        try {
          const res = await fetch(buildPageUrl(this.slug, this.options), {
            headers: { Accept: 'application/json' },
          });
          if (!res.ok) return;
          const data = await res.json();
          realtimeNs.applySnapshot(this.slug, data, { source: 'polling' });
        } catch (err) {
          console.warn('[N3Realtime] fallback polling failed', err);
        }
      }, intervalMs);
    }

    stopFallback() {
      if (this.fallbackTimer) {
        clearInterval(this.fallbackTimer);
        this.fallbackTimer = null;
      }
    }

    disconnect() {
      this.closed = true;
      this.stopFallback();
      if (this.socket && this.socket.readyState <= 1) {
        this.socket.close();
      }
      this.socket = null;
    }
  }

  function applyRealtimeUpdate(event) {
    if (!event) return;
    const slug = event.slug || event.page || event.channel;
    const payload = event.payload || event.data || event;
    if (slug && widgetsNs.hydratePage) {
      widgetsNs.hydratePage(slug, payload);
    }
  }

  function applySnapshot(slug, snapshot, meta) {
    if (!slug || !snapshot) return;
    if (widgetsNs.hydratePage) {
      widgetsNs.hydratePage(slug, snapshot);
    }
    if (widgetsNs.bus) {
      widgetsNs.bus.emit('realtime:snapshot', { slug, snapshot, meta });
    }
  }

  function connectPage(slug, options = {}) {
    if (!slug) return null;
    disconnectPage(slug);
    const client = new RealtimeClient(slug, options);
    connections.set(slug, client);
    client.connect();
    return client;
  }

  function disconnectPage(slug) {
    const existing = connections.get(slug);
    if (existing) {
      existing.disconnect();
      connections.delete(slug);
    }
  }

  Object.assign(realtimeNs, {
    connectPage,
    disconnectPage,
    applySnapshot,
    applyRealtimeUpdate,
  });
})();
