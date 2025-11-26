"""
React component generators for widgets and UI elements.

This module contains functions that generate React/TypeScript component files
for the Vite frontend, including navigation, layout, toast notifications,
and various widget types (charts, tables, forms, text blocks).
"""

import json
import textwrap
from pathlib import Path
from typing import Dict, List

from .utils import write_file


def write_navigation(lib_dir: Path, nav_links: List[Dict[str, str]]) -> None:
    """Generate navigation.ts with NavLink interface and constants."""
    rendered = json.dumps(nav_links, indent=2)
    content = textwrap.dedent(
        f"""
        export interface NavLink {{
          label: string;
          path: string;
        }}

        export const NAV_LINKS: NavLink[] = {rendered} as const;
        """
    ).strip() + "\n"
    write_file(lib_dir / "navigation.ts", content)


def write_toast_component(components_dir: Path) -> None:
    """Generate Toast.tsx with ToastProvider and useToast hook."""
    content = textwrap.dedent(
        """
        import { createContext, PropsWithChildren, useCallback, useContext, useMemo, useState } from "react";

        interface ToastContextValue {
          show: (message: string, timeoutMs?: number) => void;
        }

        const ToastContext = createContext<ToastContextValue>({
          show: () => undefined,
        });

        export function ToastProvider({ children }: PropsWithChildren) {
          const [message, setMessage] = useState<string | null>(null);
          const [timer, setTimer] = useState<number | undefined>(undefined);

          const show = useCallback((nextMessage: string, timeoutMs = 2800) => {
            setMessage(nextMessage);
            if (timer) {
              window.clearTimeout(timer);
            }
            const id = window.setTimeout(() => setMessage(null), timeoutMs);
            setTimer(id);
          }, [timer]);

          const value = useMemo<ToastContextValue>(() => ({ show }), [show]);

          return (
            <ToastContext.Provider value={value}>
              {children}
              {message ? <div className="n3-toast" role="status">{message}</div> : null}
            </ToastContext.Provider>
          );
        }

        export function useToast() {
          return useContext(ToastContext);
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "Toast.tsx", content)


def write_layout_component(components_dir: Path) -> None:
    """Generate Layout.tsx with header, navigation, and main content area."""
    content = textwrap.dedent(
        """
        import { NavLink } from "../lib/navigation";
        import { Link, useLocation } from "react-router-dom";
        import type { PropsWithChildren } from "react";

        interface LayoutProps {
          title: string;
          description?: string | null;
          navLinks: readonly NavLink[];
        }

        export default function Layout({ title, description, navLinks, children }: PropsWithChildren<LayoutProps>) {
          const location = useLocation();

          return (
            <div className="n3-app">
              <header style={{ padding: "1.25rem clamp(1rem, 4vw, 4rem)" }}>
                <h1 style={{ marginBottom: "0.25rem" }}>{title}</h1>
                {description ? <p style={{ marginTop: 0, color: "var(--text-muted, #475569)" }}>{description}</p> : null}
                <nav className="n3-nav">
                  {navLinks.map((link) => (
                    <Link key={link.path} to={link.path} className={location.pathname === link.path ? "active" : undefined}>
                      {link.label}
                    </Link>
                  ))}
                </nav>
              </header>
              <main className="n3-main">{children}</main>
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "Layout.tsx", content)


def write_chart_widget(components_dir: Path) -> None:
    """Generate ChartWidget.tsx for displaying chart data."""
    content = textwrap.dedent(
        """
        import type { ChartWidgetConfig } from "../lib/n3Client";
        import { ensureArray } from "../lib/n3Client";

        interface ChartWidgetProps {
          widget: ChartWidgetConfig;
          data: unknown;
        }

        export default function ChartWidget({ widget, data }: ChartWidgetProps) {
          const labels = Array.isArray((data as any)?.labels) ? (data as any).labels as string[] : [];
          const datasets = ensureArray<{ label?: string; data?: unknown[] }>((data as any)?.datasets);

          return (
            <section className="n3-widget">
              <h3>{widget.title}</h3>
              {labels.length && datasets.length ? (
                <div>
                  {datasets.map((dataset, index) => (
                    <div key={dataset.label ?? index} style={{ marginBottom: "0.75rem" }}>
                      <strong>{dataset.label ?? `Series ${index + 1}`}</strong>
                      <ul style={{ listStyle: "none", paddingLeft: 0 }}>
                        {labels.map((label, idx) => (
                          <li key={label + idx}>
                            <span style={{ fontWeight: 500 }}>{label}:</span> {Array.isArray(dataset.data) ? dataset.data[idx] : "n/a"}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              ) : (
                <pre style={{ marginTop: "0.75rem" }}>{JSON.stringify(data ?? widget, null, 2)}</pre>
              )}
            </section>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "ChartWidget.tsx", content)


def write_table_widget(components_dir: Path) -> None:
    """Generate TableWidget.tsx for displaying tabular data."""
    content = textwrap.dedent(
        """
        import type { TableWidgetConfig } from "../lib/n3Client";
        import { mapTableClasses } from "../lib/designTokens";

        interface TableWidgetProps {
          widget: TableWidgetConfig;
          data: unknown;
        }

        export default function TableWidget({ widget, data }: TableWidgetProps) {
          const rows = Array.isArray((data as any)?.rows) ? (data as any).rows as Record<string, unknown>[] : [];
          const columns = widget.columns && widget.columns.length ? widget.columns : rows.length ? Object.keys(rows[0]) : [];

          // Apply design tokens to table
          const tableClass = mapTableClasses(
            widget.variant || 'elevated',
            widget.tone || 'neutral',
            widget.size || 'md',
            widget.density || 'comfortable'
          );

          return (
            <section className="n3-widget">
              <h3>{widget.title}</h3>
              {rows.length ? (
                <div style={{ overflowX: "auto" }}>
                  <table className={tableClass}>
                    <thead>
                      <tr>
                        {columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, idx) => (
                        <tr key={idx}>
                          {columns.map((column) => (
                            <td key={column}>{String((row as any)[column] ?? "")}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <pre>{JSON.stringify(data ?? widget, null, 2)}</pre>
              )}
            </section>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "TableWidget.tsx", content)


def write_form_widget(components_dir: Path) -> None:
    """Generate FormWidget.tsx for interactive form submission with comprehensive field types."""
    content = textwrap.dedent(
        """
        import { FormEvent, useState, ChangeEvent, useEffect } from "react";
        import type { FormWidgetConfig } from "../lib/n3Client";
        import { useToast } from "./Toast";
        import { mapFormClasses, mapButtonClasses, mapInputClasses } from "../lib/designTokens";

        interface FormWidgetProps {
          widget: FormWidgetConfig;
          pageSlug: string;
        }

        interface FormField {
          name: string;
          component: string;
          label?: string;
          placeholder?: string;
          help_text?: string;
          required?: boolean;
          default?: string;
          validation?: {
            min_length?: number;
            max_length?: number;
            pattern?: string;
            min_value?: number;
            max_value?: number;
            step?: number;
          };
          options_binding?: string;
          options?: Array<string | { value: string; label: string }>;
          disabled?: string | boolean;
          visible?: string | boolean;
          accept?: string;
          max_file_size?: number;
          upload_endpoint?: string;
          multiple?: boolean;
          // Design tokens
          variant?: string;
          tone?: string;
          size?: string;
          density?: string;
        }

        export default function FormWidget({ widget, pageSlug }: FormWidgetProps) {
          const toast = useToast();
          const [submitting, setSubmitting] = useState(false);
          const [formData, setFormData] = useState<Record<string, any>>({});
          const [errors, setErrors] = useState<Record<string, string>>({});
          const [touched, setTouched] = useState<Record<string, boolean>>({});

          // Initialize form data with defaults
          useEffect(() => {
            const initial: Record<string, any> = {};
            (widget.fields || []).forEach((field: any) => {
              if (field.default !== undefined) {
                initial[field.name] = field.default;
              } else if (field.component === 'checkbox' || field.component === 'switch') {
                initial[field.name] = false;
              } else if (field.component === 'multiselect') {
                initial[field.name] = [];
              } else {
                initial[field.name] = '';
              }
            });
            setFormData(initial);
          }, [widget.fields]);

          const validateField = (field: FormField, value: any): string | null => {
            if (field.required && (value === '' || value === null || value === undefined)) {
              return `${field.label || field.name} is required`;
            }

            if (!field.validation) return null;

            if (typeof value === 'string') {
              if (field.validation.min_length && value.length < field.validation.min_length) {
                return `Minimum length is ${field.validation.min_length}`;
              }
              if (field.validation.max_length && value.length > field.validation.max_length) {
                return `Maximum length is ${field.validation.max_length}`;
              }
              if (field.validation.pattern) {
                const regex = new RegExp(field.validation.pattern);
                if (!regex.test(value)) {
                  return 'Invalid format';
                }
              }
            }

            if (typeof value === 'number') {
              if (field.validation.min_value !== undefined && value < field.validation.min_value) {
                return `Minimum value is ${field.validation.min_value}`;
              }
              if (field.validation.max_value !== undefined && value > field.validation.max_value) {
                return `Maximum value is ${field.validation.max_value}`;
              }
            }

            return null;
          };

          const handleChange = (fieldName: string, value: any) => {
            setFormData(prev => ({ ...prev, [fieldName]: value }));

            // Validate on change if validation_mode is 'on_change'
            if (widget.validation_mode === 'on_change') {
              const field = (widget.fields || []).find((f: any) => f.name === fieldName);
              if (field) {
                const error = validateField(field, value);
                setErrors(prev => ({ ...prev, [fieldName]: error || '' }));
              }
            }
          };

          const handleBlur = (fieldName: string) => {
            setTouched(prev => ({ ...prev, [fieldName]: true }));

            // Validate on blur if validation_mode is 'on_blur' (default)
            const validationMode = widget.validation_mode || 'on_blur';
            if (validationMode === 'on_blur') {
              const field = (widget.fields || []).find((f: any) => f.name === fieldName);
              if (field) {
                const error = validateField(field, formData[fieldName]);
                setErrors(prev => ({ ...prev, [fieldName]: error || '' }));
              }
            }
          };

          async function handleSubmit(event: FormEvent<HTMLFormElement>) {
            event.preventDefault();

            // Validate all fields before submit
            const newErrors: Record<string, string> = {};
            let hasErrors = false;

            (widget.fields || []).forEach((field: any) => {
              const error = validateField(field, formData[field.name]);
              if (error) {
                newErrors[field.name] = error;
                hasErrors = true;
              }
            });

            setErrors(newErrors);

            if (hasErrors) {
              toast.show(widget.error_message || "Please fix the errors in the form");
              return;
            }

            try {
              setSubmitting(true);
              const response = await fetch(`/api/pages/${pageSlug}/forms/${widget.id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
              });
              if (!response.ok) {
                throw new Error(`Request failed: ${response.status}`);
              }
              toast.show(widget.success_message ?? "Form submitted");
              
              // Reset form if needed
              if (widget.reset_button) {
                const initial: Record<string, any> = {};
                (widget.fields || []).forEach((field: any) => {
                  if (field.default !== undefined) {
                    initial[field.name] = field.default;
                  } else if (field.component === 'checkbox' || field.component === 'switch') {
                    initial[field.name] = false;
                  } else if (field.component === 'multiselect') {
                    initial[field.name] = [];
                  } else {
                    initial[field.name] = '';
                  }
                });
                setFormData(initial);
                setErrors({});
                setTouched({});
              }
            } catch (error) {
              console.warn("Form submission failed", error);
              toast.show(widget.error_message || "Unable to submit form right now");
            } finally {
              setSubmitting(false);
            }
          }

          const renderField = (field: FormField) => {
            const value = formData[field.name];
            const error = touched[field.name] && errors[field.name];
            const disabled = field.disabled === true || field.disabled === 'true';
            const visible = field.visible === undefined || field.visible === true || field.visible === 'true';

            if (!visible) return null;

            const fieldStyle: React.CSSProperties = {
              display: 'grid',
              gap: '0.35rem',
              marginBottom: error ? '0.5rem' : undefined,
            };

            // Apply design tokens to input fields
            const inputClass = mapInputClasses(
              field.variant || widget.variant as any,
              field.tone || widget.tone as any,
              field.size || widget.size as any
            );

            const labelStyle: React.CSSProperties = {
              fontWeight: 600,
              display: 'flex',
              gap: '0.25rem',
              alignItems: 'center',
            };

            const label = (
              <span style={labelStyle}>
                {field.label || field.name}
                {field.required && <span style={{ color: '#ef4444' }}>*</span>}
              </span>
            );

            // Text Input
            if (field.component === 'text_input') {
              return (
                <label key={field.name} style={fieldStyle}>
                  {label}
                  <input
                    type="text"
                    value={value || ''}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    onBlur={() => handleBlur(field.name)}
                    placeholder={field.placeholder}
                    disabled={disabled}
                    className={inputClass}
                    style={{ borderColor: error ? '#ef4444' : undefined }}
                  />
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // Textarea
            if (field.component === 'textarea') {
              return (
                <label key={field.name} style={fieldStyle}>
                  {label}
                  <textarea
                    value={value || ''}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    onBlur={() => handleBlur(field.name)}
                    placeholder={field.placeholder}
                    disabled={disabled}
                    rows={4}
                    style={{ ...inputStyle, resize: 'vertical' }}
                  />
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // Select
            if (field.component === 'select') {
              const options = field.options || [];
              return (
                <label key={field.name} style={fieldStyle}>
                  {label}
                  <select
                    value={value || ''}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    onBlur={() => handleBlur(field.name)}
                    disabled={disabled}
                    style={inputStyle}
                  >
                    <option value="">-- Select --</option>
                    {options.map((opt, idx) => {
                      const optValue = typeof opt === 'string' ? opt : opt.value;
                      const optLabel = typeof opt === 'string' ? opt : opt.label;
                      return <option key={idx} value={optValue}>{optLabel}</option>;
                    })}
                  </select>
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // Multiselect
            if (field.component === 'multiselect') {
              const options = field.options || [];
              return (
                <label key={field.name} style={fieldStyle}>
                  {label}
                  <select
                    multiple
                    value={value || []}
                    onChange={(e) => {
                      const selected = Array.from(e.target.selectedOptions).map(opt => opt.value);
                      handleChange(field.name, selected);
                    }}
                    onBlur={() => handleBlur(field.name)}
                    disabled={disabled}
                    style={{ ...inputStyle, minHeight: '100px' }}
                  >
                    {options.map((opt, idx) => {
                      const optValue = typeof opt === 'string' ? opt : opt.value;
                      const optLabel = typeof opt === 'string' ? opt : opt.label;
                      return <option key={idx} value={optValue}>{optLabel}</option>;
                    })}
                  </select>
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // Checkbox
            if (field.component === 'checkbox') {
              return (
                <label key={field.name} style={{ ...fieldStyle, flexDirection: 'row', gap: '0.5rem', alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={value || false}
                    onChange={(e) => handleChange(field.name, e.target.checked)}
                    onBlur={() => handleBlur(field.name)}
                    disabled={disabled}
                  />
                  {label}
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // Switch (styled checkbox)
            if (field.component === 'switch') {
              return (
                <label key={field.name} style={{ ...fieldStyle, flexDirection: 'row', gap: '0.5rem', alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={value || false}
                    onChange={(e) => handleChange(field.name, e.target.checked)}
                    onBlur={() => handleBlur(field.name)}
                    disabled={disabled}
                    style={{ width: '40px', height: '20px' }}
                  />
                  {label}
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // Radio Group
            if (field.component === 'radio_group') {
              const options = field.options || [];
              return (
                <div key={field.name} style={fieldStyle}>
                  {label}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {options.map((opt, idx) => {
                      const optValue = typeof opt === 'string' ? opt : opt.value;
                      const optLabel = typeof opt === 'string' ? opt : opt.label;
                      return (
                        <label key={idx} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                          <input
                            type="radio"
                            name={field.name}
                            value={optValue}
                            checked={value === optValue}
                            onChange={(e) => handleChange(field.name, e.target.value)}
                            onBlur={() => handleBlur(field.name)}
                            disabled={disabled}
                          />
                          <span>{optLabel}</span>
                        </label>
                      );
                    })}
                  </div>
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </div>
              );
            }

            // Slider
            if (field.component === 'slider') {
              const min = field.validation?.min_value ?? 0;
              const max = field.validation?.max_value ?? 100;
              const step = field.validation?.step ?? 1;
              return (
                <label key={field.name} style={fieldStyle}>
                  {label}
                  <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                    <input
                      type="range"
                      min={min}
                      max={max}
                      step={step}
                      value={value || min}
                      onChange={(e) => handleChange(field.name, Number(e.target.value))}
                      onBlur={() => handleBlur(field.name)}
                      disabled={disabled}
                      style={{ flex: 1 }}
                    />
                    <span style={{ minWidth: '3rem', textAlign: 'right', fontWeight: 600 }}>{value || min}</span>
                  </div>
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // Date Picker
            if (field.component === 'date_picker') {
              return (
                <label key={field.name} style={fieldStyle}>
                  {label}
                  <input
                    type="date"
                    value={value || ''}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    onBlur={() => handleBlur(field.name)}
                    disabled={disabled}
                    style={inputStyle}
                  />
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // DateTime Picker
            if (field.component === 'datetime_picker') {
              return (
                <label key={field.name} style={fieldStyle}>
                  {label}
                  <input
                    type="datetime-local"
                    value={value || ''}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                    onBlur={() => handleBlur(field.name)}
                    disabled={disabled}
                    style={inputStyle}
                  />
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // File Upload
            if (field.component === 'file_upload') {
              return (
                <label key={field.name} style={fieldStyle}>
                  {label}
                  <input
                    type="file"
                    accept={field.accept}
                    multiple={field.multiple}
                    onChange={(e) => handleChange(field.name, e.target.files)}
                    onBlur={() => handleBlur(field.name)}
                    disabled={disabled}
                    style={inputStyle}
                  />
                  {field.help_text && <span style={{ fontSize: '0.875rem', color: '#64748b' }}>{field.help_text}</span>}
                  {field.max_file_size && (
                    <span style={{ fontSize: '0.875rem', color: '#64748b' }}>
                      Max size: {(field.max_file_size / 1024 / 1024).toFixed(1)}MB
                    </span>
                  )}
                  {error && <span style={{ fontSize: '0.875rem', color: '#ef4444' }}>{error}</span>}
                </label>
              );
            }

            // Fallback
            return null;
          };

          const layoutStyle: React.CSSProperties = 
            widget.layout_mode === 'horizontal' 
              ? { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }
              : { display: 'grid', gap: '0.75rem' };

          // Apply design tokens to form container
          const formContainerClass = mapFormClasses(
            widget.variant as any,
            widget.tone as any,
            widget.size as any
          );

          // Apply design tokens to submit button
          const submitButtonClass = mapButtonClasses(
            widget.variant as any || 'elevated',
            widget.tone as any || 'primary',
            widget.size as any
          );

          return (
            <section className="n3-widget">
              <h3>{widget.title}</h3>
              <div className={formContainerClass}>
                <form onSubmit={handleSubmit} style={{ maxWidth: widget.layout_mode === 'vertical' ? '420px' : undefined }}>
                  <div style={layoutStyle}>
                    {(widget.fields || []).map((field: any) => renderField(field))}
                  </div>
                  <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem' }}>
                    <button 
                      type="submit" 
                      disabled={submitting} 
                      className={submitButtonClass}
                      style={{ 
                        cursor: submitting ? 'not-allowed' : 'pointer',
                        opacity: submitting ? 0.6 : 1,
                      }}
                    >
                      {submitting ? (widget.loading_text || "Submitting...") : (widget.submit_button_text || "Submit")}
                    </button>
                    {widget.reset_button && (
                      <button 
                        type="button" 
                        onClick={() => {
                          const initial: Record<string, any> = {};
                          (widget.fields || []).forEach((field: any) => {
                            if (field.default !== undefined) {
                              initial[field.name] = field.default;
                            } else if (field.component === 'checkbox' || field.component === 'switch') {
                              initial[field.name] = false;
                            } else if (field.component === 'multiselect') {
                              initial[field.name] = [];
                            } else {
                              initial[field.name] = '';
                            }
                          });
                          setFormData(initial);
                          setErrors({});
                          setTouched({});
                        }}
                        className={mapButtonClasses('outlined', 'neutral', widget.size as any)}
                        style={{ cursor: 'pointer' }}
                      >
                        Reset
                      </button>
                    )}
                  </div>
                </form>
              </div>
            </section>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "FormWidget.tsx", content)


def write_text_widget(components_dir: Path) -> None:
    """Generate TextBlock.tsx for displaying styled text content."""
    content = textwrap.dedent(
        """
        import type { CSSProperties } from "react";
        import type { TextWidgetConfig } from "../lib/n3Client";

        interface TextBlockProps {
          widget: TextWidgetConfig;
        }

        function normaliseStyles(styles: Record<string, string> | undefined): Record<string, string> {
          const result: Record<string, string> = {};
          if (!styles) {
            return result;
          }

          const sizeScale: Record<string, string> = {
            small: "0.875rem",
            medium: "1rem",
            large: "1.35rem",
            "x-large": "1.75rem",
            "xx-large": "2rem",
          };

          Object.entries(styles).forEach(([rawKey, rawValue]) => {
            const key = rawKey.toLowerCase();
            const value = rawValue;
            if (key === "align") {
              result.textAlign = value;
              return;
            }
            if (key === "weight") {
              const weight = value.toLowerCase();
              if (weight === "bold") {
                result.fontWeight = "700";
              } else if (weight === "light") {
                result.fontWeight = "300";
              } else if (weight === "normal") {
                result.fontWeight = "400";
              } else {
                result.fontWeight = value;
              }
              return;
            }
            if (key === "size") {
              const size = sizeScale[value.toLowerCase()] ?? value;
              result.fontSize = size;
              return;
            }
            const parts = rawKey.split(/[-_\s]+/).filter(Boolean);
            if (!parts.length) {
              return;
            }
            const camel = parts[0] + parts.slice(1).map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1)).join("");
            result[camel] = value;
          });
          return result;
        }

        export default function TextBlock({ widget }: TextBlockProps) {
          return (
            <section className="n3-widget" style={normaliseStyles(widget.styles) as CSSProperties}>
              <p style={{ margin: 0 }}>{widget.text}</p>
            </section>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "TextBlock.tsx", content)
