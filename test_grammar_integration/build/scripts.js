(function(global) {
    'use strict';

    var widgets = global.N3Widgets || (global.N3Widgets = {});

    var SEVERITY_ORDER = { debug: 0, info: 1, warning: 2, error: 3 };
    var SEVERITY_LEVELS = ['debug', 'info', 'warning', 'error'];

    function normalizeSeverity(value) {
        var text = value == null ? '' : String(value).toLowerCase();
        return Object.prototype.hasOwnProperty.call(SEVERITY_ORDER, text) ? text : 'error';
    }

    function normalizeError(entry) {
        if (!entry || typeof entry !== 'object') {
            return null;
        }
        return {
            code: entry.code ? String(entry.code) : 'error',
            message: entry.message ? String(entry.message) : 'Runtime error encountered.',
            detail: entry.detail != null ? String(entry.detail) : null,
            scope: entry.scope != null ? String(entry.scope) : null,
            source: entry.source != null ? String(entry.source) : null,
            severity: normalizeSeverity(entry.severity),
        };
    }

    function highestSeverity(list) {
        var rank = -1;
        var severity = null;
        if (!Array.isArray(list)) {
            return severity;
        }
        list.forEach(function(entry) {
            if (!entry) {
                return;
            }
            var order = SEVERITY_ORDER[entry.severity] != null ? SEVERITY_ORDER[entry.severity] : SEVERITY_ORDER.error;
            if (order >= rank) {
                rank = order;
                severity = entry.severity;
            }
        });
        return severity;
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

    function extend(target) {
        for (var i = 1; i < arguments.length; i += 1) {
            var source = arguments[i];
            if (!source) {
                continue;
            }
            Object.keys(source).forEach(function(key) {
                target[key] = source[key];
            });
        }
        return target;
    }

    function mergeHeaders(existing, overrides) {
        var result = {};
        if (existing) {
            if (Array.isArray(existing)) {
                existing.forEach(function(entry) {
                    if (!entry || entry.length < 2) {
                        return;
                    }
                    result[String(entry[0])] = String(entry[1]);
                });
            } else if (typeof Headers !== 'undefined' && existing instanceof Headers) {
                existing.forEach(function(value, key) {
                    result[key] = value;
                });
            } else if (typeof existing === 'object') {
                Object.keys(existing).forEach(function(key) {
                    result[key] = existing[key];
                });
            }
        }
        if (overrides && typeof overrides === 'object') {
            Object.keys(overrides).forEach(function(key) {
                result[key] = overrides[key];
            });
        }
        return result;
    }

    function requestJson(url, init) {
        if (typeof fetch !== 'function') {
            return Promise.reject(new Error('Fetch API is not available'));
        }
        return fetch(url, init).then(function(response) {
            if (!response || !response.ok) {
                throw new Error('HTTP ' + (response && response.status));
            }
            var contentType = '';
            if (response && response.headers && typeof response.headers.get === 'function') {
                contentType = response.headers.get('content-type') || '';
            }
            if (contentType.indexOf('application/json') === -1) {
                return null;
            }
            return response.json();
        });
    }

    var crud = global.N3Crud || (global.N3Crud = {});

    crud.fetchResource = function(url, init) {
        var headers = mergeHeaders(init && init.headers, { 'Accept': 'application/json' });
        return requestJson(url, extend({}, init || {}, { headers: headers }));
    };

    crud.submitJson = function(url, payload, init) {
        var headers = mergeHeaders(init && init.headers, { 'Accept': 'application/json', 'Content-Type': 'application/json' });
        var body = init && typeof init.body !== 'undefined' ? init.body : JSON.stringify(payload || {});
        var method = (init && init.method) || 'POST';
        return requestJson(url, extend({}, init || {}, { method: method, headers: headers, body: body }));
    };

    crud.mergePartial = function(target, updates, options) {
        var shouldCopy = !options || options.copy !== false;
        var base = target && typeof target === 'object' ? (shouldCopy ? cloneObject(target) : target) : {};
        if (!updates || typeof updates !== 'object') {
            return base;
        }
        deepMerge(base, updates);
        return base;
    };

    crud.applyPartial = function(target, updates) {
        return crud.mergePartial(target, updates, { copy: false });
    };

    function forEachNode(collection, callback) {
        if (!collection || typeof collection.length === 'undefined' || typeof callback !== 'function') {
            return;
        }
        Array.prototype.forEach.call(collection, callback);
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

    widgets.routeToPath = function(route) {
        if (!route || route === '/') {
            return 'index.html';
        }
        var cleaned = String(route).trim();
        var segments = cleaned.split('/').filter(Boolean);
        cleaned = segments.join('_');
        if (!cleaned) {
            cleaned = 'index';
        }
        if (!/[.]html$/i.test(cleaned)) {
            cleaned += '.html';
        }
        return cleaned;
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

    widgets.renderErrors = function(container, errors) {
        if (!container) {
            return;
        }
        SEVERITY_LEVELS.forEach(function(level) {
            container.classList.remove('n3-widget-errors--severity-' + level);
        });
        container.innerHTML = '';
        if (!Array.isArray(errors) || !errors.length) {
            container.classList.add('n3-widget-errors--hidden');
            return;
        }
        var normalized = [];
        errors.forEach(function(entry) {
            var value = normalizeError(entry);
            if (value) {
                normalized.push(value);
            }
        });
        if (!normalized.length) {
            container.classList.add('n3-widget-errors--hidden');
            return;
        }
        container.classList.remove('n3-widget-errors--hidden');
        var severity = highestSeverity(normalized);
        if (severity) {
            container.classList.add('n3-widget-errors--severity-' + severity);
        }
        normalized.forEach(function(entry) {
            var wrapper = document.createElement('div');
            wrapper.className = 'n3-widget-error';

            var code = document.createElement('span');
            code.className = 'n3-widget-error__code';
            code.textContent = entry.code;
            wrapper.appendChild(code);

            var message = document.createElement('span');
            message.className = 'n3-widget-error__message';
            message.textContent = entry.message;
            wrapper.appendChild(message);

            if (entry.detail) {
                var detail = document.createElement('span');
                detail.className = 'n3-widget-error__detail';
                detail.textContent = entry.detail;
                wrapper.appendChild(detail);
            }

            container.appendChild(wrapper);
        });
    };

    var ErrorSurface = (function() {
        function escapeAttr(value) {
            return String(value).replace(/"/g, '\"');
        }

        function normalizeList(errors) {
            if (!Array.isArray(errors)) {
                return [];
            }
            var items = [];
            errors.forEach(function(entry) {
                var value = normalizeError(entry);
                if (value) {
                    items.push(value);
                }
            });
            return items;
        }

        function collectFieldNames(form) {
            var names = [];
            if (!form || typeof form.querySelectorAll !== 'function') {
                return names;
            }
            forEachNode(form.querySelectorAll('[data-n3-field]'), function(node) {
                var value = node.getAttribute('data-n3-field');
                if (value) {
                    names.push(value);
                }
            });
            return names;
        }

        function extractField(scope, knownFields) {
            if (scope == null) {
                return null;
            }
            var text = String(scope).trim();
            if (!text) {
                return null;
            }
            var lowered = text.toLowerCase();
            var prefixes = ['field:', 'field.', 'fields:', 'fields.', 'payload:', 'payload.', 'form:', 'form.field:', 'form.field.', 'form.payload:', 'form.payload.'];
            var matched = false;
            for (var i = 0; i < prefixes.length; i++) {
                var prefix = prefixes[i];
                if (lowered.indexOf(prefix) === 0) {
                    text = text.slice(prefix.length);
                    lowered = text.toLowerCase();
                    matched = true;
                    break;
                }
            }
            if (!matched && Array.isArray(knownFields)) {
                if (knownFields.indexOf(text) !== -1) {
                    matched = true;
                }
            }
            if (!matched) {
                return null;
            }
            var parts = text.split(/[:./]/).filter(function(part) { return part; });
            if (!parts.length) {
                return null;
            }
            return parts[parts.length - 1];
        }

        function clearField(form, fieldName) {
            if (!form || !fieldName) {
                return;
            }
            var fieldSelector = '[data-n3-field="' + escapeAttr(fieldName) + '"]';
            forEachNode(form.querySelectorAll(fieldSelector + ' input,' + fieldSelector + ' textarea,' + fieldSelector + ' select'), function(control) {
                control.classList.remove('n3-input-error');
            });
            var errorNode = form.querySelector('[data-n3-field-error="' + escapeAttr(fieldName) + '"]');
            if (errorNode) {
                errorNode.textContent = '';
                errorNode.classList.remove('n3-field-error--visible');
            }
        }

        function setFieldError(form, fieldName, entry) {
            if (!form || !fieldName || !entry) {
                return;
            }
            var message = entry.message || 'Invalid value.';
            if (entry.detail) {
                message = message + ' — ' + entry.detail;
            }
            var errorNode = form.querySelector('[data-n3-field-error="' + escapeAttr(fieldName) + '"]');
            if (errorNode) {
                errorNode.textContent = message;
                errorNode.classList.add('n3-field-error--visible');
            }
            var fieldSelector = '[data-n3-field="' + escapeAttr(fieldName) + '"]';
            forEachNode(form.querySelectorAll(fieldSelector + ' input,' + fieldSelector + ' textarea,' + fieldSelector + ' select'), function(control) {
                control.classList.add('n3-input-error');
            });
        }

        function partition(errors, knownFields) {
            var normalized = normalizeList(errors);
            var result = { general: [], fields: {} };
            normalized.forEach(function(entry) {
                var fieldName = extractField(entry.scope, knownFields);
                if (fieldName) {
                    if (!result.fields[fieldName]) {
                        result.fields[fieldName] = [];
                    }
                    result.fields[fieldName].push(entry);
                } else {
                    result.general.push(entry);
                }
            });
            return result;
        }

        function clearForm(form) {
            if (!form) {
                return;
            }
            var slot = form.querySelector('[data-n3-error-slot]');
            widgets.renderErrors(slot, []);
            var names = collectFieldNames(form);
            names.forEach(function(name) {
                clearField(form, name);
            });
        }

        function applyFormErrors(form, errors) {
            if (!form) {
                return;
            }
            var names = collectFieldNames(form);
            names.forEach(function(name) {
                clearField(form, name);
            });
            var split = partition(errors, names);
            var slot = form.querySelector('[data-n3-error-slot]');
            widgets.renderErrors(slot, split.general);
            Object.keys(split.fields).forEach(function(name) {
                var entries = split.fields[name];
                if (entries && entries.length) {
                    setFieldError(form, name, entries[0]);
                }
            });
        }

        function clearAction(button) {
            if (!button) {
                return;
            }
            var slot = null;
            if (typeof button.closest === 'function') {
                var wrapper = button.closest('[data-n3-action-wrapper]');
                if (wrapper) {
                    slot = wrapper.querySelector('[data-n3-error-slot]');
                }
            }
            widgets.renderErrors(slot, []);
        }

        function applyActionErrors(button, errors) {
            if (!button) {
                return;
            }
            var slot = null;
            if (typeof button.closest === 'function') {
                var wrapper = button.closest('[data-n3-action-wrapper]');
                if (wrapper) {
                    slot = wrapper.querySelector('[data-n3-error-slot]');
                }
            }
            widgets.renderErrors(slot, normalizeList(errors));
        }

        function applyPageErrors(errors) {
            var slot = document.querySelector('[data-n3-page-errors]');
            widgets.renderErrors(slot, normalizeList(errors));
        }

        return {
            normalizeList: normalizeList,
            clearField: clearField,
            clearForm: clearForm,
            applyFormErrors: applyFormErrors,
            clearAction: clearAction,
            applyActionErrors: applyActionErrors,
            applyPageErrors: applyPageErrors,
        };
    })();

    widgets.ErrorSurface = ErrorSurface;

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

    widgets.renderChart = function(canvasId, config, layout, insightName, errors) {
        var canvas = document.getElementById(canvasId);
        if (!canvas || typeof Chart === 'undefined') {
            return;
        }
        widgets.applyLayout(canvas, layout);
        var errorSlot = null;
        if (typeof canvas.closest === 'function') {
            var wrapper = canvas.closest('section');
            if (wrapper) {
                errorSlot = wrapper.querySelector('[data-n3-error-slot]');
            }
        }
        widgets.renderErrors(errorSlot, errors);
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

    widgets.renderTable = function(tableId, data, layout, insightName, errors) {
        var table = document.getElementById(tableId);
        if (!table) {
            return;
        }
        widgets.applyLayout(table, layout);
        var columns = Array.isArray(data && data.columns) ? data.columns.slice() : [];
        var rows = Array.isArray(data && data.rows) ? data.rows : [];
        table.innerHTML = '';

        var errorSlot = null;
        if (typeof table.closest === 'function') {
            var wrapper = table.closest('section');
            if (wrapper) {
                errorSlot = wrapper.querySelector('[data-n3-error-slot]');
            }
        }
        widgets.renderErrors(errorSlot, errors || (data && data.errors));

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
                widgets.renderChart(entry.id, config, entry.layout, entry.insight, payload && payload.errors);
            } else if (entry.type === 'table') {
                var tableData = payload && payload.rows ? payload : (payload || def.data || {});
                widgets.renderTable(
                    entry.id,
                    tableData,
                    entry.layout,
                    entry.insight,
                    payload && payload.errors
                );
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

    widgets.findDefinitionByIndex = function(index) {
        var defs = widgets.__definitions__ || [];
        for (var i = 0; i < defs.length; i++) {
            var def = defs[i];
            if (def && typeof def.componentIndex === 'number' && def.componentIndex === index) {
                return def;
            }
        }
        return null;
    };

    widgets.fetchComponentData = function(definition) {
        if (!definition || typeof definition.endpoint !== 'string' || !definition.endpoint) {
            return;
        }
        var endpoint = definition.endpoint;
        var index = typeof definition.componentIndex === 'number' ? definition.componentIndex : null;
        crud.fetchResource(endpoint)
            .then(function(data) {
                if (index !== null && data !== null) {
                    widgets.updateComponent(index, data, { endpoint: endpoint });
                }
            })
            .catch(function(err) {
                console.warn('Failed to fetch component payload', endpoint, err);
            });
    };

    widgets.refreshAllComponents = function() {
        var defs = widgets.__definitions__ || [];
        defs.forEach(function(def) {
            if (def && (def.type === 'chart' || def.type === 'table') && def.endpoint) {
                widgets.fetchComponentData(def);
            }
        });
    };

    widgets.handleInteractionResponse = function(response, definition) {
        var def = definition || {};
        if (!response) {
            if (def.successMessage) {
                widgets.showToast(def.successMessage);
            }
            return;
        }
        var pageErrors = Array.isArray(response.pageErrors)
            ? response.pageErrors.slice()
            : (Array.isArray(response.page_errors) ? response.page_errors.slice() : null);
        if (pageErrors !== null) {
            widgets.ErrorSurface.applyPageErrors(pageErrors);
        }
        var effects = [];
        if (Array.isArray(response.results)) {
            effects = response.results.slice();
        } else if (Array.isArray(response.effects)) {
            effects = response.effects.slice();
        }
        var shouldRefresh = false;
        if (effects.length) {
            effects.forEach(function(effect) {
                if (!effect) {
                    return;
                }
                var kind = String(effect.type || '').toLowerCase();
                if (kind === 'toast') {
                    widgets.showToast(effect.message || def.successMessage || 'Done');
                } else if (kind === 'navigate') {
                    var status = String(effect.status || 'ok').toLowerCase();
                    if (status === 'ok') {
                        var targetUrl = effect.url || null;
                        if (!targetUrl && effect.page_route) {
                            targetUrl = widgets.routeToPath(effect.page_route);
                        } else if (!targetUrl && effect.page_slug) {
                            targetUrl = widgets.routeToPath(effect.page_slug);
                        }
                        if (targetUrl) {
                            window.location.href = targetUrl;
                        }
                    }
                } else if (kind === 'update') {
                    if (String(effect.status || 'ok').toLowerCase() === 'ok') {
                        shouldRefresh = true;
                        if (!effect.silent) {
                            widgets.showToast(effect.message || 'Update completed');
                        }
                    }
                } else if (kind === 'python_call' || kind === 'connector_call' || kind === 'chain_run') {
                    widgets.showToast('Action completed');
                }
            });
        } else if (def.successMessage) {
            widgets.showToast(def.successMessage);
        }

        if (response.refresh && Array.isArray(response.refresh.components)) {
            response.refresh.components.forEach(function(idx) {
                var target = widgets.findDefinitionByIndex(idx);
                if (target) {
                    widgets.fetchComponentData(target);
                }
            });
        } else if (shouldRefresh) {
            widgets.refreshAllComponents();
        }
    };

    widgets.registerForm = function(definition) {
        if (!definition || !definition.id) {
            return;
        }
        var form = document.getElementById(definition.id);
        if (!form || form.__n3_registered__) {
            return;
        }
        form.__n3_registered__ = true;
        form.__n3_definition__ = definition;

        form.addEventListener('input', function(event) {
            var target = event && event.target;
            if (target && target.name) {
                widgets.ErrorSurface.clearField(form, target.name);
            }
        });

        form.addEventListener('submit', function(event) {
            event.preventDefault();
            var endpoint = definition.endpoint || form.getAttribute('data-n3-endpoint');
            if (!endpoint) {
                widgets.showToast('Form endpoint not configured');
                return;
            }
            var submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.classList.add('n3-button--pending');
            }
            form.classList.add('n3-form--submitting');
            widgets.ErrorSurface.clearForm(form);

            var formData = new FormData(form);
            var payload = {};
            formData.forEach(function(value, key) {
                if (Object.prototype.hasOwnProperty.call(payload, key)) {
                    if (!Array.isArray(payload[key])) {
                        payload[key] = [payload[key]];
                    }
                    payload[key].push(value);
                } else {
                    payload[key] = value;
                }
            });

            crud.submitJson(endpoint, payload)
                .then(function(data) {
                    if (data === null) {
                        widgets.ErrorSurface.applyFormErrors(form, []);
                        widgets.ErrorSurface.applyPageErrors([]);
                        if (definition.successMessage) {
                            widgets.showToast(definition.successMessage);
                        }
                        if (definition.resetOnSuccess !== false && typeof form.reset === 'function') {
                            form.reset();
                        }
                        return;
                    }
                    var errors = Array.isArray(data.errors) ? data.errors : [];
                    var pageErrors = Array.isArray(data.pageErrors)
                        ? data.pageErrors.slice()
                        : (Array.isArray(data.page_errors) ? data.page_errors.slice() : []);
                    widgets.ErrorSurface.applyFormErrors(form, errors);
                    widgets.ErrorSurface.applyPageErrors(pageErrors);
                    var status = data.status ? String(data.status).toLowerCase() : 'ok';
                    if (status === 'error' || errors.length) {
                        return;
                    }
                    widgets.handleInteractionResponse(data, definition);
                    if (definition.resetOnSuccess !== false && typeof form.reset === 'function') {
                        form.reset();
                    }
                })
                .catch(function(err) {
                    console.warn('Form submission failed', endpoint, err);
                    widgets.ErrorSurface.applyFormErrors(form, [{ code: 'network_error', message: 'Unable to submit form right now.', severity: 'error' }]);
                    widgets.showToast('Unable to submit form right now');
                })
                .finally(function() {
                    form.classList.remove('n3-form--submitting');
                    if (submitButton) {
                        submitButton.disabled = false;
                        submitButton.classList.remove('n3-button--pending');
                    }
                });
        });

        if (Array.isArray(definition.errors) && definition.errors.length) {
            widgets.ErrorSurface.applyFormErrors(form, definition.errors);
        }
    };

    widgets.registerAction = function(definition) {
        if (!definition || !definition.id) {
            return;
        }
        var button = document.getElementById(definition.id);
        if (!button || button.__n3_registered__) {
            return;
        }
        button.__n3_registered__ = true;
        button.__n3_definition__ = definition;

        button.addEventListener('click', function() {
            var endpoint = definition.endpoint || button.getAttribute('data-n3-endpoint');
            if (!endpoint) {
                widgets.showToast('Action endpoint not configured');
                return;
            }
            button.disabled = true;
            button.classList.add('n3-action--pending');
            widgets.ErrorSurface.clearAction(button);

            crud.submitJson(endpoint, {})
                .then(function(data) {
                    if (data === null) {
                        widgets.ErrorSurface.applyActionErrors(button, []);
                        widgets.ErrorSurface.applyPageErrors([]);
                        if (definition.successMessage) {
                            widgets.showToast(definition.successMessage);
                        }
                        return;
                    }
                    var errors = Array.isArray(data.errors) ? data.errors : [];
                    var pageErrors = Array.isArray(data.pageErrors)
                        ? data.pageErrors.slice()
                        : (Array.isArray(data.page_errors) ? data.page_errors.slice() : []);
                    widgets.ErrorSurface.applyActionErrors(button, errors);
                    widgets.ErrorSurface.applyPageErrors(pageErrors);
                    var status = data.status ? String(data.status).toLowerCase() : 'ok';
                    if (status === 'error' || errors.length) {
                        return;
                    }
                    widgets.handleInteractionResponse(data, definition);
                })
                .catch(function(err) {
                    console.warn('Action execution failed', endpoint, err);
                    widgets.ErrorSurface.applyActionErrors(button, [{ code: 'network_error', message: 'Unable to run action', severity: 'error' }]);
                    widgets.showToast('Unable to run action');
                })
                .finally(function() {
                    button.disabled = false;
                    button.classList.remove('n3-action--pending');
                });
        });

        if (Array.isArray(definition.errors) && definition.errors.length) {
            widgets.ErrorSurface.applyActionErrors(button, definition.errors);
        }
    };

    widgets.hydratePage = function(slug, payload) {
        if (!payload || typeof payload !== 'object') {
            return;
        }
        widgets.__pageSlug = slug;
        global.N3_PAGE_STATE = payload;
        if (payload.vars && typeof payload.vars === 'object') {
            global.N3_VARS = payload.vars;
        }
        var pageErrors = Array.isArray(payload.errors) ? payload.errors : [];
        widgets.ErrorSurface.applyPageErrors(pageErrors);
    };

    widgets.applyRealtimeUpdate = function(event) {
        if (!event || typeof event !== 'object') {
            return;
        }
        if (event.type === 'component') {
            var componentIndex = typeof event.component_index === 'number'
                ? event.component_index
                : parseInt(event.component_index, 10);
            if (!isNaN(componentIndex)) {
                widgets.updateComponent(componentIndex, event.payload || {}, event.meta || {});
            }
            return;
        }
        if (event.type === 'rollback') {
            var rollbackIndex = typeof event.component_index === 'number'
                ? event.component_index
                : parseInt(event.component_index, 10);
            if (!isNaN(rollbackIndex)) {
                widgets.rollbackComponent(rollbackIndex);
            }
            return;
        }
        if (event.type === 'snapshot' || event.type === 'hydration') {
            if (event.payload && typeof event.payload === 'object') {
                widgets.hydratePage(event.slug || widgets.__pageSlug || null, event.payload);
            }
            return;
        }
        if (event.payload && typeof event.payload === 'object') {
            var mergedState = crud && typeof crud.mergePartial === 'function'
                ? crud.mergePartial(global.N3_PAGE_STATE || {}, event.payload, { copy: true })
                : deepMerge(cloneObject(global.N3_PAGE_STATE || {}), event.payload);
            widgets.hydratePage(event.slug || widgets.__pageSlug || null, mergedState);
        }
    };

    widgets.bootstrap = function(definitions) {
        if (!Array.isArray(definitions)) {
            return;
        }
        widgets.__definitions__ = definitions.slice();
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
                widgets.renderChart(
                    def.id,
                    def.config || {},
                    def.layout || {},
                    def.insight || null,
                    def.errors || []
                );
                if (hasIndex) {
                    widgets.rememberSnapshot(def.componentIndex, def.config || {});
                }
                if (def.endpoint) {
                    widgets.fetchComponentData(def);
                }
            } else if (def.type === 'table') {
                if (def.insight && def.data && typeof def.data === 'object') {
                    def.data.insight = def.insight;
                }
                widgets.renderTable(
                    def.id,
                    def.data || {},
                    def.layout || {},
                    def.insight || null,
                    (def.data && def.data.errors) || def.errors || []
                );
                if (hasIndex) {
                    widgets.rememberSnapshot(def.componentIndex, def.data || {});
                }
                if (def.endpoint) {
                    widgets.fetchComponentData(def);
                }
            } else if (def.type === 'insight') {
                widgets.renderInsight(def.id, def);
            } else if (def.type === 'form') {
                widgets.registerForm(def);
            } else if (def.type === 'action') {
                widgets.registerAction(def);
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
})(window);