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
        var crud = global.N3Crud;
        crud.fetchResource(url)
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
        } else if (event.payload && typeof event.payload === 'object') {
            var currentState = stateStore[slug] || {};
            var crud = global.N3Crud;
            if (crud && typeof crud.mergePartial === 'function') {
                stateStore[slug] = crud.mergePartial(currentState, event.payload, { copy: true });
            } else {
                stateStore[slug] = deepMerge(cloneObject(currentState), event.payload);
            }
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
                // ignore
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

})(typeof window !== 'undefined' ? window : this);
