/**
 * N3 Widget Core Runtime
 *
 * Provides:
 *  - State management and event bus
 *  - HTTP helpers for JSON APIs
 *  - Error rendering utilities
 *  - Bootstrap/hydration entrypoints for generated frontends
 *
 * The API is exposed via global namespaces:
 *  - window.N3Widgets
 *  - window.N3Crud
 *
 * All functions are safe to consume from ES6 modules or classic script tags.
 */
(() => {
  'use strict';

  /** @type {any} */
  const globalScope = typeof window !== 'undefined' ? window : globalThis;
  const widgetsNs = (globalScope.N3Widgets = globalScope.N3Widgets || {});
  const crudNs = (globalScope.N3Crud = globalScope.N3Crud || {});
  const pageStores = new Map();
  const pendingBootstraps = [];

  /** Simple pub/sub bus */
  class EventBus {
    constructor() {
      this.handlers = new Map();
    }

    /**
     * @param {string} event
     * @param {(payload: any) => void} handler
     * @returns {() => void} unsubscribe
     */
    on(event, handler) {
      if (!event || typeof handler !== 'function') {
        return () => {};
      }
      const list = this.handlers.get(event) || [];
      list.push(handler);
      this.handlers.set(event, list);
      return () => this.off(event, handler);
    }

    /**
     * @param {string} event
     * @param {(payload: any) => void} handler
     */
    off(event, handler) {
      const list = this.handlers.get(event);
      if (!list) return;
      const next = list.filter((fn) => fn !== handler);
      this.handlers.set(event, next);
    }

    /**
     * @param {string} event
     * @param {any} payload
     */
    emit(event, payload) {
      const list = this.handlers.get(event);
      if (!list) return;
      list.forEach((fn) => {
        try {
          fn(payload);
        } catch (err) {
          console.error('[N3Widgets] handler failed for', event, err);
        }
      });
    }
  }

  /** Reactive store for a page */
  class StateStore {
    constructor(initial = {}) {
      this.state = { ...initial };
      this.subscribers = new Set();
    }

    getSnapshot() {
      return { ...this.state };
    }

    /**
     * @param {Record<string, any> | ((current: Record<string, any>) => Record<string, any>)} update
     */
    set(update) {
      const next =
        typeof update === 'function' ? update({ ...this.state }) : { ...this.state, ...(update || {}) };
      this.state = next;
      this.notify();
    }

    /**
     * Replace all state
     * @param {Record<string, any>} next
     */
    replace(next) {
      this.state = { ...(next || {}) };
      this.notify();
    }

    /**
     * @param {(state: Record<string, any>) => void} fn
     * @returns {() => void}
     */
    subscribe(fn) {
      if (typeof fn !== 'function') return () => {};
      this.subscribers.add(fn);
      fn({ ...this.state });
      return () => this.subscribers.delete(fn);
    }

    notify() {
      this.subscribers.forEach((fn) => {
        try {
          fn({ ...this.state });
        } catch (err) {
          console.error('[N3Widgets] subscriber failed', err);
        }
      });
    }
  }

  const bus = new EventBus();

  /**
   * @param {HeadersInit | undefined} base
   * @param {HeadersInit | undefined} extra
   * @returns {Record<string,string>}
   */
  function mergeHeaders(base, extra) {
    const normalized = {};
    const assign = (source) => {
      if (!source) return;
      if (Array.isArray(source)) {
        source.forEach((pair) => {
          if (Array.isArray(pair) && pair.length >= 2) normalized[String(pair[0])] = String(pair[1]);
        });
      } else if (typeof Headers !== 'undefined' && source instanceof Headers) {
        source.forEach((value, key) => (normalized[key] = value));
      } else if (typeof source === 'object') {
        Object.keys(source).forEach((key) => (normalized[key] = String(source[key])));
      }
    };
    assign(base);
    assign(extra);
    return normalized;
  }

  /**
   * Robust fetch helper with JSON convenience.
   * @param {string} url
   * @param {RequestInit & { json?: any, expected?: 'json' | 'text' }} options
   */
  async function httpRequest(url, options = {}) {
    const expected = options.expected || 'json';
    const headers = mergeHeaders(options.headers, {
      Accept: expected === 'json' ? 'application/json' : '*/*',
      ...(options.json !== undefined ? { 'Content-Type': 'application/json' } : {}),
    });

    const init = {
      method: options.method || (options.json !== undefined ? 'POST' : 'GET'),
      headers,
      body: options.json !== undefined ? JSON.stringify(options.json) : options.body,
      signal: options.signal,
      mode: options.mode,
      credentials: options.credentials || 'same-origin',
    };

    const response = await fetch(url, init);
    const contentType = response.headers?.get('content-type') || '';
    let payload = null;

    if (contentType.includes('application/json')) {
      payload = await response.json().catch(() => null);
    } else if (expected === 'text') {
      payload = await response.text().catch(() => null);
    }

    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}`); // eslint-disable-line no-undef
      error.status = response.status;
      error.payload = payload;
      throw error;
    }

    return { status: response.status, data: payload, headers: response.headers };
  }

  const SEVERITY_ORDER = { debug: 0, info: 1, warning: 2, error: 3 };

  /**
   * @param {any} entry
   */
  function normalizeError(entry) {
    if (!entry || typeof entry !== 'object') return null;
    const severity = (entry.severity || 'error').toString().toLowerCase();
    const normalizedSeverity = Object.prototype.hasOwnProperty.call(SEVERITY_ORDER, severity)
      ? severity
      : 'error';
    return {
      code: entry.code ? String(entry.code) : 'error',
      message: entry.message ? String(entry.message) : 'Runtime error encountered.',
      detail: entry.detail != null ? String(entry.detail) : null,
      scope: entry.scope != null ? String(entry.scope) : null,
      severity: normalizedSeverity,
    };
  }

  /**
   * Render a list of normalized errors into a container.
   * @param {HTMLElement|null} container
   * @param {any[]} errors
   */
  function renderErrors(container, errors) {
    if (!container) return;
    container.innerHTML = '';
    container.classList.add('n3-widget-errors');
    container.setAttribute('role', 'alert');

    if (!Array.isArray(errors) || !errors.length) {
      container.classList.add('n3-widget-errors--hidden');
      return;
    }

    const normalized = errors
      .map((e) => normalizeError(e))
      .filter(Boolean)
      .sort((a, b) => SEVERITY_ORDER[b.severity] - SEVERITY_ORDER[a.severity]);

    if (!normalized.length) {
      container.classList.add('n3-widget-errors--hidden');
      return;
    }

    container.classList.remove('n3-widget-errors--hidden');
    normalized.forEach((entry) => {
      const row = document.createElement('div');
      row.className = `n3-widget-error n3-widget-error--${entry.severity}`;

      const message = document.createElement('span');
      message.className = 'n3-widget-error__message';
      message.textContent = entry.message;
      row.appendChild(message);

      if (entry.detail) {
        const detail = document.createElement('span');
        detail.className = 'n3-widget-error__detail';
        detail.textContent = entry.detail;
        row.appendChild(detail);
      }

      container.appendChild(row);
    });
  }

  /**
   * Resolve nested property paths like "user.profile[0].name".
   * @param {string} path
   * @param {Record<string, any>} data
   */
  function resolvePath(path, data) {
    if (!path) return undefined;
    const cleaned = String(path).replace(/\[(\d+)\]/g, '.$1');
    return cleaned.split('.').reduce((target, key) => {
      if (target == null) return undefined;
      return Object.prototype.hasOwnProperty.call(target, key) ? target[key] : undefined;
    }, data);
  }

  /**
   * Replace {tokens} in a string with values from data.
   * @param {string} template
   * @param {Record<string, any>} data
   */
  function interpolate(template, data) {
    if (!template) return '';
    return String(template).replace(/\{([^{}]+)\}/g, (_match, token) => {
      const value = resolvePath(token.trim(), data);
      return value == null ? '' : String(value);
    });
  }

  function routeToPath(route) {
    if (!route || route === '/') return 'index.html';
    const cleaned = String(route).split('/').filter(Boolean).join('_') || 'index';
    return cleaned.endsWith('.html') ? cleaned : `${cleaned}.html`;
  }

  function getStore(slug) {
    if (!pageStores.has(slug)) {
      pageStores.set(slug, new StateStore());
    }
    return pageStores.get(slug);
  }

  /**
   * Hydrate a page snapshot and broadcast to the renderer.
   * @param {string} slug
   * @param {any} payload
   */
  function hydratePage(slug, payload) {
    const store = getStore(slug);
    const snapshot = payload && payload.data ? payload.data : payload;
    store.replace(snapshot || {});
    if (widgetsNs.__renderer && typeof widgetsNs.__renderer.hydratePage === 'function') {
      widgetsNs.__renderer.hydratePage(slug, payload || {}, { store, bus });
    }
    bus.emit('page:hydrated', { slug, payload });
  }

  /**
   * Bootstrap widget definitions produced by Namel3ss.
   * @param {Array<any>} widgetDefs
   */
  function bootstrap(widgetDefs) {
    if (!Array.isArray(widgetDefs)) return;
    const renderer = widgetsNs.__renderer;
    if (renderer && typeof renderer.renderWidgets === 'function') {
      renderer.renderWidgets(widgetDefs, { bus, getStore });
    } else {
      pendingBootstraps.push(widgetDefs);
    }
  }

  /**
   * Register renderer API provided by widget-rendering.js.
   * @param {{ renderWidgets?: Function, hydratePage?: Function }} rendererApi
   */
  function registerRenderer(rendererApi) {
    widgetsNs.__renderer = rendererApi;
    if (pendingBootstraps.length && rendererApi && typeof rendererApi.renderWidgets === 'function') {
      pendingBootstraps.splice(0).forEach((defs) => rendererApi.renderWidgets(defs, { bus, getStore }));
    }
  }

  function showToast(message, durationMs = 3000) {
    const toast = globalScope.document && globalScope.document.getElementById
      ? globalScope.document.getElementById('toast')
      : null;
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), durationMs);
  }

  // Expose CRUD helpers
  crudNs.fetchResource = (url, init) => httpRequest(url, { ...(init || {}), expected: 'json' });
  crudNs.submitJson = (url, payload, init) =>
    httpRequest(url, { ...(init || {}), json: payload, method: (init && init.method) || 'POST' });
  crudNs.mergePartial = (target, updates) => Object.assign({}, target || {}, updates || {});
  crudNs.applyPartial = (target, updates) => Object.assign(target || {}, updates || {});

  // Expose widget runtime
  Object.assign(widgetsNs, {
    EventBus,
    StateStore,
    bus,
    httpRequest,
    mergeHeaders,
    renderErrors,
    interpolate,
    resolvePath,
    routeToPath,
    hydratePage,
    bootstrap,
    registerRenderer,
    getStore,
    showToast,
  });
})();
