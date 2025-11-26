import { FormEvent, useState, ChangeEvent, useEffect } from "react";
import type { FormWidgetConfig } from "../lib/n3Client";
import { useToast } from "./Toast";

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

    const inputStyle: React.CSSProperties = {
      padding: '0.55rem 0.75rem',
      borderRadius: '0.5rem',
      border: `1px solid ${error ? '#ef4444' : 'rgba(15,23,42,0.18)'}`,
      width: '100%',
    };

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
            style={inputStyle}
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

  return (
    <section className="n3-widget">
      <h3>{widget.title}</h3>
      <form onSubmit={handleSubmit} style={{ maxWidth: widget.layout_mode === 'vertical' ? '420px' : undefined }}>
        <div style={layoutStyle}>
          {(widget.fields || []).map((field: any) => renderField(field))}
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem' }}>
          <button 
            type="submit" 
            disabled={submitting} 
            style={{ 
              padding: '0.65rem 1.25rem', 
              borderRadius: '0.65rem', 
              border: 'none', 
              background: 'var(--primary, #2563eb)', 
              color: '#fff', 
              fontWeight: 600,
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
              style={{ 
                padding: '0.65rem 1.25rem', 
                borderRadius: '0.65rem', 
                border: '1px solid rgba(15,23,42,0.18)', 
                background: '#fff', 
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Reset
            </button>
          )}
        </div>
      </form>
    </section>
  );
}
