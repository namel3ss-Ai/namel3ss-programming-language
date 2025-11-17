(function(global) {
    'use strict';

    var widgets = global.N3Widgets || (global.N3Widgets = {});

    function cloneObject(value) {
        if (!value || typeof value !== 'object') {
            return {};
        }
        try {
            return JSON.parse(JSON.stringify(value));
        } catch (err) {
            var copy = Array.isArray(value) ? [] : {};
            Object.keys(value).forEach(function(key) {
                var entry = value[key];
                if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
                    copy[key] = cloneObject(entry);
                } else if (Array.isArray(entry)) {
                    copy[key] = entry.slice();
                } else {
                    copy[key] = entry;
                }
            });
            return copy;
        }
    }

    function deepMerge(target, source) {
        if (!source || typeof source !== 'object') {
            return target;
        }
        Object.keys(source).forEach(function(key) {
            var incoming = source[key];
            if (incoming && typeof incoming === 'object' && !Array.isArray(incoming)) {
                if (!target[key] || typeof target[key] !== 'object' || Array.isArray(target[key])) {
                    target[key] = {};
                }
                deepMerge(target[key], incoming);
            } else {
                target[key] = incoming;
            }
        });
        return target;
    }

    widgets.applyLayout = function(element, layout) {
        if (!element || !layout) {
            return;
        }
        if (layout.variant) {
            element.classList.add('n3-widget--' + String(layout.variant).toLowerCase());
        }
        if (layout.align) {
            element.classList.add('n3-align-' + String(layout.align).toLowerCase());
        }
        if (layout.emphasis) {
            element.classList.add('n3-emphasis-' + String(layout.emphasis).toLowerCase());
        }
    };

    widgets.resolvePath = function(path, data) {
        if (!path) {
            return undefined;
        }
        var cleaned = String(path).trim();
        if (!cleaned) {
            return undefined;
        }
        var target = data;
        var segments = cleaned.replace(/\[(\d+)\]/g, '.$1').split('.');
        for (var i = 0; i < segments.length; i++) {
            var key = segments[i];
            if (!key) {
                continue;
            }
            if (target == null) {
                return undefined;
            }
            if (Object.prototype.hasOwnProperty.call(target, key)) {
                target = target[key];
                continue;
            }
            if (typeof target === 'object' && key in target) {
                target = target[key];
            } else {
                return undefined;
            }
        }
        return target;
    };

    widgets.interpolate = function(template, data) {
        if (!template) {
            return '';
        }
        return String(template).replace(/\{([^{}]+)\}/g, function(match, token) {
            var value = widgets.resolvePath(token, data);
            return value == null ? '' : String(value);
        });
    };

    widgets.showToast = function(message, duration) {
        var toast = document.getElementById('toast');
        if (!toast) {
            return;
        }
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(function() {
            toast.classList.remove('show');
        }, duration || 3000);
    };

    if (!global.showToast) {
        global.showToast = widgets.showToast;
    }

    widgets.populateMetrics = function(container, metrics) {
        if (!container) {
            return;
        }
        container.innerHTML = '';
        if (!Array.isArray(metrics) || !metrics.length) {
            var empty = document.createElement('div');
            empty.className = 'n3-insight-empty';
            empty.textContent = 'No metrics available.';
            container.appendChild(empty);
            return;
        }
        metrics.forEach(function(metric) {
            if (!metric) {
                return;
            }
            var card = document.createElement('div');
            card.className = 'n3-insight-metric';
            if (Array.isArray(metric.alerts) && metric.alerts.some(function(alert) { return alert && alert.triggered; })) {
                card.classList.add('n3-insight-metric--alert');
            }
            var label = document.createElement('div');
            label.className = 'n3-insight-metric__label';
            label.textContent = metric.label || metric.name || 'Metric';
            card.appendChild(label);

            var value = document.createElement('div');
            value.className = 'n3-insight-metric__value';
            if (metric.formatted) {
                value.textContent = metric.formatted;
            } else if (metric.value != null) {
                value.textContent = String(metric.value);
            } else {
                value.textContent = '—';
            }
            card.appendChild(value);

            var trendText = '';
            if (typeof metric.delta === 'number') {
                trendText += (metric.delta >= 0 ? '+' : '') + metric.delta.toFixed(2);
            }
            if (typeof metric.delta_pct === 'number') {
                if (trendText) {
                    trendText += ' (';
                }
                trendText += (metric.delta_pct >= 0 ? '+' : '') + metric.delta_pct.toFixed(1) + '%';
                if (trendText.indexOf('(') !== -1) {
                    trendText += ')';
                }
            }
            if (!trendText && metric.trend) {
                trendText = metric.trend;
            }
            if (!trendText && metric.status) {
                trendText = String(metric.status);
            }
            if (trendText) {
                var trend = document.createElement('div');
                trend.className = metric.trend ? 'n3-insight-metric__trend' : 'n3-insight-metric__status';
                trend.textContent = trendText;
                card.appendChild(trend);
            }

            container.appendChild(card);
        });
    };

    widgets.populateNarratives = function(container, narratives, templateData) {
        if (!container) {
            return;
        }
        container.innerHTML = '';
        if (!Array.isArray(narratives) || !narratives.length) {
            return;
        }
        narratives.forEach(function(narrative) {
            if (!narrative) {
                return;
            }
            var block = document.createElement('div');
            block.className = 'n3-insight-narrative';
            if (narrative.variant) {
                block.classList.add('n3-insight-narrative--' + String(narrative.variant).toLowerCase());
            }
            if (narrative.style && typeof narrative.style === 'object') {
                Object.keys(narrative.style).forEach(function(key) {
                    var cssKey = key.replace(/[A-Z]/g, function(match) {
                        return '-' + match.toLowerCase();
                    });
                    block.style.setProperty(cssKey, narrative.style[key]);
                });
            }
            var text = narrative.text;
            if (!text && narrative.template) {
                text = widgets.interpolate(narrative.template, templateData);
            }
            block.textContent = text || '';
            container.appendChild(block);
        });
    };

    widgets.renderChart = function(canvasId, config, layout, insightName) {
        var canvas = document.getElementById(canvasId);
        if (!canvas || typeof Chart === 'undefined') {
            return;
        }
        widgets.applyLayout(canvas, layout);
        var ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }
        if (canvas.__n3_chart__) {
            canvas.__n3_chart__.destroy();
        }
        var backgroundColor = null;
        if (config && typeof config === 'object') {
            if (typeof config.backgroundColor === 'string' && config.backgroundColor.trim()) {
                backgroundColor = config.backgroundColor;
            } else if (
                config.options &&
                typeof config.options === 'object' &&
                typeof config.options.backgroundColor === 'string' &&
                config.options.backgroundColor.trim()
            ) {
                backgroundColor = config.options.backgroundColor;
            }
        }
        var container = canvas.parentElement;
        if (backgroundColor) {
            canvas.style.backgroundColor = backgroundColor;
            if (container && container.classList && container.classList.contains('n3-chart-container')) {
                container.style.backgroundColor = backgroundColor;
            }
        } else {
            canvas.style.backgroundColor = '';
            if (container && container.classList && container.classList.contains('n3-chart-container')) {
                container.style.backgroundColor = '';
            }
        }
        var insightRef = insightName || (config && config.insight);
        if (insightRef) {
            canvas.setAttribute('data-n3-insight-ref', insightRef);
        }
        canvas.__n3_chart__ = new Chart(ctx, config);
    };

    widgets.renderTable = function(tableId, data, layout, insightName) {
        var table = document.getElementById(tableId);
        if (!table) {
            return;
        }
        widgets.applyLayout(table, layout);
        var columns = Array.isArray(data && data.columns) ? data.columns.slice() : [];
        var rows = Array.isArray(data && data.rows) ? data.rows : [];
        table.innerHTML = '';

        if (insightName) {
            table.setAttribute('data-n3-insight-ref', insightName);
        } else if (data && data.insight) {
            table.setAttribute('data-n3-insight-ref', data.insight);
        }
        if (!columns.length && rows.length && typeof rows[0] === 'object' && rows[0] !== null) {
            columns = Object.keys(rows[0]);
        }

        if (columns.length) {
            var thead = document.createElement('thead');
            var headerRow = document.createElement('tr');
            columns.forEach(function(column) {
                var th = document.createElement('th');
                th.textContent = column;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);
        }

        var tbody = document.createElement('tbody');
        rows.forEach(function(row) {
            var tr = document.createElement('tr');
            columns.forEach(function(column, index) {
                var td = document.createElement('td');
                if (row && typeof row === 'object' && row !== null && Object.prototype.hasOwnProperty.call(row, column)) {
                    td.textContent = row[column];
                } else if (Array.isArray(row)) {
                    td.textContent = row[index] != null ? row[index] : '';
                } else {
                    td.textContent = '';
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
    };

    widgets.renderInsight = function(containerId, spec) {
        var container = document.getElementById(containerId);
        if (!container) {
            return;
        }
        widgets.applyLayout(container, spec && spec.layout);
        var slug = spec && spec.slug;
        var endpoint = spec && spec.endpoint;
        if (!endpoint && slug) {
            endpoint = '/api/insights/' + slug;
        }
        if (!endpoint) {
            container.classList.add('n3-insight-card--error');
            container.setAttribute('data-error', 'Missing insight endpoint');
            return;
        }
        fetch(endpoint, { headers: { 'Accept': 'application/json' } })
            .then(function(response) {
                if (!response.ok) {
                    throw new Error('Failed to load insight (' + response.status + ')');
                }
                return response.json();
            })
            .then(function(payload) {
                var result = payload && payload.result ? payload.result : {};
                var metrics = Array.isArray(result.metrics) ? result.metrics : [];
                var metricMap = {};
                metrics.forEach(function(metric) {
                    if (metric && metric.name) {
                        metricMap[metric.name] = metric;
                    }
                });
                var templateData = Object.assign({}, result, {
                    metrics: metricMap,
                    variables: result.variables || result.expose || result.expose_as || {},
                    alerts: result.alerts_list || [],
                    selection: result.selection || [],
                    events: result.events || [],
                });
                widgets.populateMetrics(container.querySelector('[data-n3-insight="metrics"]'), metrics);
                widgets.populateNarratives(
                    container.querySelector('[data-n3-insight="narratives"]'),
                    Array.isArray(result.narratives) ? result.narratives : [],
                    templateData
                );
            })
            .catch(function(err) {
                container.classList.add('n3-insight-card--error');
                container.setAttribute('data-error', err && err.message ? err.message : 'Insight failed');
            });
    };

    function getRegistry() {
        if (!widgets.__registry__) {
            widgets.__registry__ = {};
        }
        return widgets.__registry__;
    }

    widgets.chartResponseToConfig = function(response, fallback) {
        if (!response) {
            return fallback || {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: (fallback && fallback.options) || {},
            };
        }
        if (response.type && response.data && Array.isArray(response.data.datasets)) {
            return response;
        }
        var chartType = response.chart_type || (fallback && fallback.type) || 'line';
        var labels = Array.isArray(response.labels) ? response.labels.slice() : [];
        if (!labels.length && fallback && fallback.data && Array.isArray(fallback.data.labels)) {
            labels = fallback.data.labels.slice();
        }
        var series = Array.isArray(response.series) ? response.series : [];
        var datasets = series.map(function(entry) {
            var next = Object.assign({}, entry || {});
            if (!Array.isArray(next.data)) {
                next.data = [];
            }
            if (!next.label) {
                next.label = response.title || 'Series';
            }
            return next;
        });
        if (!datasets.length && fallback && fallback.data && Array.isArray(fallback.data.datasets)) {
            datasets = fallback.data.datasets.map(function(item) {
                return Object.assign({}, item);
            });
        }
        if (!datasets.length) {
            datasets = [{ label: response.title || 'Series', data: [] }];
        }
        var options = {};
        if (fallback && fallback.options && typeof fallback.options === 'object') {
            options = cloneObject(fallback.options);
        }
        if (response.options && typeof response.options === 'object') {
            options = deepMerge(options, response.options);
        }
        var background = null;
        if (typeof response.backgroundColor === 'string' && response.backgroundColor.trim()) {
            background = response.backgroundColor;
        } else if (options && typeof options.backgroundColor === 'string' && options.backgroundColor.trim()) {
            background = options.backgroundColor;
        } else if (fallback && typeof fallback.backgroundColor === 'string' && fallback.backgroundColor.trim()) {
            background = fallback.backgroundColor;
        }
        var config = {
            type: chartType,
            data: {
                labels: labels,
                datasets: datasets,
            },
            options: options,
        };
        if (background) {
            config.backgroundColor = background;
        }
        return config;
    };

    widgets.registerComponent = function(def) {
        if (!def || typeof def.componentIndex !== 'number') {
            return;
        }
        var registry = getRegistry();
        var index = def.componentIndex;
        var entry = registry[index] || {};
        entry.type = def.type;
        entry.id = def.id;
        entry.layout = def.layout || {};
        entry.insight = def.insight || null;
        entry.initial = def;
        entry.lastSnapshot = entry.lastSnapshot || null;
        entry.previousSnapshot = entry.previousSnapshot || null;
        entry.render = function(payload) {
            if (entry.type === 'chart') {
                var config = widgets.chartResponseToConfig(payload, def.config || null);
                widgets.renderChart(entry.id, config, entry.layout, entry.insight);
            } else if (entry.type === 'table') {
                var tableData = payload && payload.rows ? payload : (payload || def.data || {});
                widgets.renderTable(entry.id, tableData, entry.layout, entry.insight);
            }
        };
        registry[index] = entry;
    };

    widgets.rememberSnapshot = function(index, snapshot) {
        var registry = getRegistry();
        var entry = registry[index];
        if (!entry) {
            return;
        }
        try {
            entry.lastSnapshot = snapshot == null ? snapshot : JSON.parse(JSON.stringify(snapshot));
        } catch (err) {
            entry.lastSnapshot = snapshot;
        }
    };

    widgets.updateComponent = function(index, payload, meta) {
        var registry = getRegistry();
        var entry = registry[index];
        if (!entry) {
            return;
        }
        if (entry.lastSnapshot != null) {
            try {
                entry.previousSnapshot = JSON.parse(JSON.stringify(entry.lastSnapshot));
            } catch (err) {
                entry.previousSnapshot = entry.lastSnapshot;
            }
        } else {
            entry.previousSnapshot = null;
        }
        entry.lastSnapshot = payload;
        entry.render(payload);
        if (meta) {
            entry.meta = meta;
        }
    };

    widgets.rollbackComponent = function(index) {
        var registry = getRegistry();
        var entry = registry[index];
        if (!entry) {
            return;
        }
        var snapshot = entry.previousSnapshot;
        if (!snapshot) {
            return;
        }
        entry.previousSnapshot = null;
        entry.lastSnapshot = snapshot;
        entry.render(snapshot);
    };

    widgets.applyRealtimeUpdate = function(event) {
        if (!event || typeof event !== 'object') {
            return;
        }
        var meta = event.meta || {};
        var componentIndex = meta.component_index;
        if (componentIndex === undefined || componentIndex === null) {
            componentIndex = event.component_index;
        }
        if (componentIndex !== undefined && componentIndex !== null && typeof componentIndex !== 'number') {
            var parsed = parseInt(componentIndex, 10);
            componentIndex = isNaN(parsed) ? null : parsed;
        }
        if (event.type === 'update') {
            if (typeof componentIndex === 'number') {
                widgets.updateComponent(componentIndex, event.payload || {}, meta);
            }
            return;
        }
        if (event.type === 'mutation') {
            var status = typeof meta.status === 'string' ? meta.status.toLowerCase() : '';
            if (status === 'rollback' && typeof componentIndex === 'number') {
                widgets.rollbackComponent(componentIndex);
                return;
            }
            if (typeof componentIndex === 'number' && (status === 'pending' || status === 'confirmed' || status === 'applied')) {
                widgets.updateComponent(componentIndex, event.payload || {}, meta);
            }
            return;
        }
        if (event.type === 'error') {
            if (global.console && typeof global.console.warn === 'function') {
                global.console.warn('Realtime error event', event);
            }
            return;
        }
        if (event.type === 'snapshot') {
            if (event.payload && typeof event.payload === 'object') {
                global.N3_PAGE_STATE = event.payload;
            }
        }
    };

    widgets.bootstrap = function(definitions) {
        if (!Array.isArray(definitions)) {
            return;
        }
        definitions.forEach(function(def) {
            if (!def || !def.id) {
                return;
            }
            var hasIndex = typeof def.componentIndex === 'number';
            if (hasIndex) {
                widgets.registerComponent(def);
            }
            if (def.type === 'chart') {
                if (def.insight && def.config && typeof def.config === 'object') {
                    def.config.insight = def.insight;
                }
                widgets.renderChart(def.id, def.config || {}, def.layout || {}, def.insight || null);
                if (hasIndex) {
                    widgets.rememberSnapshot(def.componentIndex, def.config || {});
                }
            } else if (def.type === 'table') {
                if (def.insight && def.data && typeof def.data === 'object') {
                    def.data.insight = def.insight;
                }
                widgets.renderTable(def.id, def.data || {}, def.layout || {}, def.insight || null);
                if (hasIndex) {
                    widgets.rememberSnapshot(def.componentIndex, def.data || {});
                }
            } else if (def.type === 'insight') {
                widgets.renderInsight(def.id, def);
            }
        });
    };
})(window);

(function(global) {
    'use strict';

    var realtime = global.N3Realtime || (global.N3Realtime = {});
    var connections = realtime.__connections || (realtime.__connections = {});
    var stateStore = realtime.__state || (realtime.__state = {});

    function toIntervalMs(value) {
        if (typeof value === 'number' && value > 0) {
            return value * 1000;
        }
        var parsed = parseInt(value, 10);
        if (!isNaN(parsed) && parsed > 0) {
            return parsed * 1000;
        }
        return null;
    }

    function dispatchEvent(name, detail) {
        if (typeof document === 'undefined' || typeof document.dispatchEvent !== 'function') {
            return;
        }
        try {
            var evt;
            if (typeof CustomEvent === 'function') {
                evt = new CustomEvent(name, { detail: detail });
            } else {
                evt = document.createEvent('CustomEvent');
                evt.initCustomEvent(name, true, true, detail);
            }
            document.dispatchEvent(evt);
        } catch (err) {
            console.warn('N3Realtime dispatch failed', err);
        }
    }

    function normalizePath(path) {
        if (typeof path !== 'string') {
            return '';
        }
        if (!path) {
            return '';
        }
        return path.charAt(0) === '/' ? path : '/' + path;
    }

    function buildWsUrl(slug, options) {
        if (!slug) {
            return null;
        }
        var explicit = options && options.wsUrl;
        if (explicit) {
            return explicit;
        }
        if (typeof window === 'undefined' || !window.location || !window.location.host) {
            return null;
        }
        var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        var host = window.location.host;
        var base = protocol + '//' + host;
        var path = options && options.wsPath ? options.wsPath : '/ws/pages/' + encodeURIComponent(slug);
        return base.replace(/[/]$/, '') + normalizePath(path);
    }

    function buildPageUrl(slug, options) {
        var path = options && options.pageUrl ? options.pageUrl : '/api/pages/' + slug;
        var base = options && options.baseUrl;
        if (base) {
            return base.replace(/[/]$/, '') + normalizePath(path);
        }
        return normalizePath(path);
    }

    function getConnection(slug) {
        var state = connections[slug];
        if (!state) {
            state = connections[slug] = {
                slug: slug,
                active: false,
                retries: 0,
                fallbackIntervalMs: null,
                reconnectTimer: null,
                fallbackTimer: null,
                websocket: null,
                options: {},
            };
        }
        return state;
    }

    function stopReconnect(state) {
        if (state.reconnectTimer) {
            clearTimeout(state.reconnectTimer);
            state.reconnectTimer = null;
        }
    }

    function stopFallback(state) {
        if (state.fallbackTimer) {
            clearInterval(state.fallbackTimer);
            state.fallbackTimer = null;
        }
    }

    function fetchSnapshot(slug, state, reason) {
        if (typeof fetch !== 'function') {
            return;
        }
        var baseUrl = buildPageUrl(slug, state.options);
        if (!baseUrl) {
            return;
        }
        var separator = baseUrl.indexOf('?') === -1 ? '?' : '&';
        var url = baseUrl + separator + '_ts=' + Date.now();
        fetch(url, { headers: { 'Accept': 'application/json' } })
            .then(function(response) {
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status);
                }
                return response.json();
            })
            .then(function(payload) {
                realtime.applyEvent(slug, {
                    type: 'snapshot',
                    slug: slug,
                    payload: payload,
                    meta: { source: reason || 'poll' },
                });
            })
            .catch(function(err) {
                console.warn('N3Realtime polling failed for ' + slug + ':', err);
            });
    }

    function startFallback(slug, state, reason) {
        stopFallback(state);
        if (!state.fallbackIntervalMs) {
            return;
        }
        fetchSnapshot(slug, state, reason || 'fallback-start');
        state.fallbackTimer = setInterval(function() {
            fetchSnapshot(slug, state, 'fallback-tick');
        }, state.fallbackIntervalMs);
    }

    function applyEvent(slug, event) {
        if (!event || typeof event !== 'object') {
            return;
        }
        if (!event.slug) {
            event.slug = slug;
        }
        if (event.type === 'snapshot' || event.type === 'hydration') {
            stateStore[slug] = event.payload || {};
        }
        if (global.N3Widgets && typeof global.N3Widgets.applyRealtimeUpdate === 'function') {
            try {
                global.N3Widgets.applyRealtimeUpdate(event);
            } catch (err) {
                console.error('N3Realtime failed to update widgets', err);
            }
        }
        dispatchEvent('n3:realtime:' + (event.type || 'message'), {
            slug: slug,
            event: event,
        });
    }

    function handleMessage(slug, state, raw) {
        var data = raw;
        if (typeof raw === 'string') {
            try {
                data = JSON.parse(raw);
            } catch (err) {
                console.warn('N3Realtime received non-JSON message', raw);
                return;
            }
        }
        applyEvent(slug, data);
    }

    function scheduleReconnect(slug, state) {
        stopReconnect(state);
        if (!state.active) {
            return;
        }
        state.retries += 1;
        // simple exponential backoff with an upper bound
        var delay = Math.min(30000, Math.pow(2, state.retries) * 250);
        state.reconnectTimer = setTimeout(function() {
            openWebSocket(slug, state);
        }, delay);
        if (state.fallbackIntervalMs) {
            startFallback(slug, state, 'reconnect-wait');
        }
    }

    function openWebSocket(slug, state) {
        stopReconnect(state);
        if (!state.active) {
            return;
        }
        var url = buildWsUrl(slug, state.options);
        if (!url) {
            if (state.fallbackIntervalMs) {
                startFallback(slug, state, 'no-websocket');
            }
            return;
        }
        try {
            var socket = new WebSocket(url);
            state.websocket = socket;
            socket.onopen = function() {
                state.retries = 0;
                stopFallback(state);
                dispatchEvent('n3:realtime:connected', { slug: slug });
            };
            socket.onmessage = function(evt) {
                handleMessage(slug, state, evt.data);
            };
            socket.onerror = function(err) {
                console.warn('N3Realtime websocket error for ' + slug + ':', err);
            };
            socket.onclose = function() {
                state.websocket = null;
                dispatchEvent('n3:realtime:disconnected', { slug: slug });
                if (state.active) {
                    scheduleReconnect(slug, state);
                } else {
                    stopFallback(state);
                }
            };
        } catch (err) {
            console.warn('N3Realtime failed to open websocket for ' + slug + ':', err);
            if (state.fallbackIntervalMs) {
                startFallback(slug, state, 'websocket-error');
            }
        }
    }

    realtime.connectPage = function(slug, options) {
        if (!slug) {
            return;
        }
        var state = getConnection(slug);
        state.active = true;
        state.options = options || {};
        state.fallbackIntervalMs = toIntervalMs(state.options.fallbackInterval);
        if (state.websocket && (state.websocket.readyState === 0 || state.websocket.readyState === 1)) {
            return;
        }
        var skipWebSocket = typeof window !== 'undefined' && window.location && window.location.protocol === 'file:';
        if (skipWebSocket) {
            startFallback(slug, state, 'file-protocol');
            return;
        }
        openWebSocket(slug, state);
        if (!state.websocket && state.fallbackIntervalMs) {
            startFallback(slug, state, 'websocket-unavailable');
        }
    };

    realtime.disconnectPage = function(slug) {
        var state = connections[slug];
        if (!state) {
            return;
        }
        state.active = false;
        stopReconnect(state);
        stopFallback(state);
        if (state.websocket && state.websocket.readyState <= 1) {
            try {
                state.websocket.close();
            } catch (err) {
                // ignore close errors
            }
        }
        state.websocket = null;
    };

    realtime.applyEvent = function(slug, event) {
        applyEvent(slug, event);
    };

    realtime.applySnapshot = function(slug, payload, meta) {
        applyEvent(slug, {
            type: 'snapshot',
            slug: slug,
            payload: payload || {},
            meta: meta || {},
        });
    };

    realtime.getState = function(slug) {
        return stateStore[slug];
    };
})(window);