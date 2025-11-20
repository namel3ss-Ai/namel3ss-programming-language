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
            return String(value).replace(/"/g, '\\"');
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

})(typeof window !== 'undefined' ? window : this);
