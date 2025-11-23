/**
 * N3 Widget Rendering Runtime
 *
 * Responsibilities:
 *  - Register and render widget components
 *  - Bind data and actions to DOM
 *  - Provide accessible, responsive defaults
 *
 * Public API (exposed on window.N3Widgets / window.N3Renderer):
 *  - registerComponent(type, renderer)
 *  - renderWidget(definition, context)
 *  - renderWidgets(definitions, context)
 *  - hydratePage(slug, payload, context)
 */
(() => {
  'use strict';

  /** @type {any} */
  const globalScope = typeof window !== 'undefined' ? window : globalThis;
  const widgetsNs = (globalScope.N3Widgets = globalScope.N3Widgets || {});
  const rendererNs = (globalScope.N3Renderer = globalScope.N3Renderer || {});
  const componentRegistry = new Map();

  /**
   * @typedef {Object} RenderContext
   * @property {any} bus
   * @property {(slug: string) => any} [getStore]
   * @property {HTMLElement} [root]
   */

  /**
   * Attach layout and style options to an element.
   * @param {HTMLElement} el
   * @param {Record<string, any>} layout
   */
  function applyLayout(el, layout = {}) {
    if (!el || !layout) return;
    if (layout.variant) el.classList.add(`n3-variant-${String(layout.variant).toLowerCase()}`);
    if (layout.align) el.classList.add(`n3-align-${String(layout.align).toLowerCase()}`);
    if (layout.emphasis) el.classList.add(`n3-emphasis-${String(layout.emphasis).toLowerCase()}`);
  }

  /**
   * Resolve target container for a widget definition.
   * @param {any} def
   */
  function resolveHost(def) {
    if (def.target && typeof document !== 'undefined') {
      return document.querySelector(`[data-n3-widget="${def.target}"]`) || document.getElementById(def.target);
    }
    if (typeof document !== 'undefined') {
      return document.getElementById(def.id) || document.querySelector(`[data-n3-widget="${def.id}"]`);
    }
    return null;
  }

  /**
   * @param {string} type
   * @param {(def: any, ctx: RenderContext) => HTMLElement | void} renderer
   */
  function registerComponent(type, renderer) {
    if (!type || typeof renderer !== 'function') return;
    componentRegistry.set(type, renderer);
  }

  /**
   * Render a single widget definition.
   * @param {any} def
   * @param {RenderContext} ctx
   */
  function renderWidget(def, ctx) {
    if (!def || !def.type) return null;
    const renderer = componentRegistry.get(def.type);
    if (!renderer) {
      console.warn(`[N3Widgets] No renderer for type "${def.type}"`);
      return null;
    }

    const host = resolveHost(def) || (typeof document !== 'undefined' ? document.body : null);
    if (!host) return null;

    // Clear host on first render for deterministic output
    if (def.replace !== false) {
      host.innerHTML = '';
    }

    const node = renderer(def, ctx);
    if (node) {
      applyLayout(node, def.layout || {});
      host.appendChild(node);
    }
    return node;
  }

  /**
   * Render multiple widget definitions.
   * @param {any[]} defs
   * @param {RenderContext} ctx
   */
  function renderWidgets(defs, ctx = {}) {
    if (!Array.isArray(defs)) return [];
    return defs.map((def) => renderWidget(def, ctx)).filter(Boolean);
  }

  /**
   * Hydrate page data by re-rendering widgets with latest payload.
   * @param {string} slug
   * @param {any} payload
   * @param {RenderContext} ctx
   */
  function hydratePage(slug, payload, ctx = {}) {
    const widgets = payload && payload.widgets ? payload.widgets : null;
    if (Array.isArray(widgets)) {
      renderWidgets(
        widgets.map((w) => ({ ...w, slug })),
        ctx,
      );
    }

    // Simple text bindings: data-n3-text-template attributes
    if (typeof document !== 'undefined' && payload && payload.data) {
      document.querySelectorAll('[data-n3-text-template]').forEach((el) => {
        const tpl = el.getAttribute('data-n3-text-template') || '';
        el.textContent = widgetsNs.interpolate ? widgetsNs.interpolate(tpl, payload.data) : tpl;
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Default components
  // ---------------------------------------------------------------------------

  registerComponent('text', (def) => {
    const el = document.createElement(def.variant || 'p');
    el.textContent = def.data?.text || '';
    if (def.data?.html === true && def.data?.text) {
      el.innerHTML = def.data.text;
    }
    return el;
  });

  registerComponent('heading', (def) => {
    const level = Math.min(Math.max(parseInt(def.data?.level || '2', 10) || 2, 1), 6);
    const el = document.createElement(`h${level}`);
    el.textContent = def.data?.text || '';
    return el;
  });

  registerComponent('button', (def, ctx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'n3-btn';
    btn.textContent = def.data?.label || 'Submit';
    btn.addEventListener('click', async () => {
      if (def.action && def.action.url) {
        try {
          await widgetsNs.httpRequest(def.action.url, {
            method: def.action.method || 'POST',
            json: def.action.payload || {},
          });
          widgetsNs.showToast && widgetsNs.showToast(def.action.success_message || 'Action completed');
          ctx.bus && ctx.bus.emit('action:completed', { id: def.id, action: def.action });
        } catch (err) {
          console.error('[N3Widgets] action failed', err);
          ctx.bus && ctx.bus.emit('action:error', { id: def.id, error: err });
        }
      }
    });
    return btn;
  });

  registerComponent('metric', (def) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'n3-metric';

    const label = document.createElement('div');
    label.className = 'n3-metric__label';
    label.textContent = def.data?.label || '';

    const value = document.createElement('div');
    value.className = 'n3-metric__value';
    value.textContent = def.data?.value != null ? String(def.data.value) : '';

    wrapper.appendChild(label);
    wrapper.appendChild(value);
    return wrapper;
  });

  registerComponent('table', (def) => {
    const table = document.createElement('table');
    table.className = 'n3-table';
    table.setAttribute('role', 'table');

    const captionText = def.data?.title || def.data?.caption;
    if (captionText) {
      const caption = document.createElement('caption');
      caption.textContent = captionText;
      table.appendChild(caption);
    }

    const columns = def.data?.columns || [];
    const rows = def.data?.rows || [];

    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach((col) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = col.label || col.name || col;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    rows.forEach((row) => {
      const tr = document.createElement('tr');
      columns.forEach((col) => {
        const key = col.key || col.name || col;
        const td = document.createElement('td');
        td.textContent =
          row && Object.prototype.hasOwnProperty.call(row, key) ? String(row[key]) : '\u2014';
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    return table;
  });

  registerComponent('form', (def, ctx) => {
    const form = document.createElement('form');
    form.noValidate = true;
    form.setAttribute('data-n3-form-id', def.id || '');

    const errors = document.createElement('div');
    errors.className = 'n3-form-errors n3-widget-errors n3-widget-errors--hidden';

    const fields = def.data?.fields || [];
    fields.forEach((field) => {
      const wrapper = document.createElement('label');
      wrapper.className = 'n3-field';
      const label = document.createElement('span');
      label.className = 'n3-field__label';
      label.textContent = field.label || field.name;
      const input =
        field.type === 'textarea'
          ? document.createElement('textarea')
          : document.createElement(field.type === 'number' ? 'input' : 'input');
      if (field.type === 'number') input.type = 'number';
      input.name = field.name;
      input.required = Boolean(field.required);
      input.placeholder = field.placeholder || '';
      if (field.maxLength) input.maxLength = field.maxLength;
      wrapper.appendChild(label);
      wrapper.appendChild(input);
      form.appendChild(wrapper);
    });

    const submit = document.createElement('button');
    submit.type = 'submit';
    submit.className = 'n3-btn';
    submit.textContent = (def.data && def.data.submit_label) || 'Submit';
    form.appendChild(submit);
    form.appendChild(errors);

    form.addEventListener('submit', async (evt) => {
      evt.preventDefault();
      const formData = new FormData(form);
      const payload = {};
      formData.forEach((value, key) => {
        payload[key] = value;
      });

      try {
        const endpoint = def.endpoint || def.action?.url;
        if (!endpoint) {
          throw new Error('Form endpoint not configured');
        }
        await widgetsNs.httpRequest(endpoint, {
          method: def.action?.method || 'POST',
          json: payload,
        });
        widgetsNs.showToast && widgetsNs.showToast(def.data?.success_message || 'Saved');
        errors.classList.add('n3-widget-errors--hidden');
        ctx.bus && ctx.bus.emit('form:submitted', { id: def.id, payload });
      } catch (err) {
        widgetsNs.renderErrors(errors, [err.payload || { message: err.message }]);
        ctx.bus && ctx.bus.emit('form:error', { id: def.id, error: err });
      }
    });

    return form;
  });

  // Register renderer API with core
  const rendererApi = { registerComponent, renderWidget, renderWidgets, hydratePage };
  if (typeof widgetsNs.registerRenderer === 'function') {
    widgetsNs.registerRenderer(rendererApi);
  }
  Object.assign(rendererNs, rendererApi);
})();
