(function(global) {
    'use strict';

    var widgets = global.N3Widgets || (global.N3Widgets = {});

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
        var crud = global.N3Crud;
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

            var crud = global.N3Crud;
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

            var crud = global.N3Crud;
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
            var crud = global.N3Crud;
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

})(typeof window !== 'undefined' ? window : this);
