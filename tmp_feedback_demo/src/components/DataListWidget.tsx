import React from 'react';

interface ListItemConfig {
  avatar?: {
    field?: string;
    fallback?: string;
    size?: 'sm' | 'md' | 'lg';
  };
  title: {
    text?: string;
    field?: string;
  };
  subtitle?: {
    text?: string;
    field?: string;
  };
  metadata?: Array<{
    field?: string;
    text?: string;
    icon?: string;
    format?: string;
  }>;
  badge?: {
    field?: string;
    text?: string;
    style?: string;
    transform?: string | Record<string, any>;
  };
  actions?: Array<{
    label: string;
    action: string;
    icon?: string;
    condition?: string;
  }>;
}

interface EmptyStateConfig {
  icon?: string;
  title: string;
  message?: string;
  actionLabel?: string;
  actionLink?: string;
}

interface DataListWidgetConfig {
  id: string;
  type: 'data_list';
  title?: string;
  source: {
    kind: string;
    name: string;
  };
  itemConfig: ListItemConfig;
  emptyState?: EmptyStateConfig;
}

interface DataListWidgetProps {
  widget: DataListWidgetConfig;
  data: unknown;
}

function renderTemplate(template: string, item: any): string {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, key) => {
    return item[key] !== undefined ? String(item[key]) : '';
  });
}

function getFieldValue(config: { field?: string; text?: string }, item: any): string {
  if (config.text) {
    return renderTemplate(config.text, item);
  }
  if (config.field) {
    return String(item[config.field] || '');
  }
  return '';
}

function formatMetadata(value: any, format?: string): string {
  if (!value) return '';

  if (format === 'relative') {
    const date = new Date(value);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  }

  if (format === 'date') {
    return new Date(value).toLocaleDateString();
  }

  return String(value);
}

function EmptyState({ config }: { config: EmptyStateConfig }) {
  return (
    <div className="n3-empty-state" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
      {config.icon && (
        <div className="empty-state-icon" style={{ fontSize: '2rem', marginBottom: '1rem', opacity: 0.5 }}>
          📝
        </div>
      )}
      <h3 style={{ margin: '0 0 0.5rem', fontSize: '1.25rem', fontWeight: 600 }}>{config.title}</h3>
      {config.message && (
        <p style={{ margin: '0 0 1rem', color: '#666' }}>{config.message}</p>
      )}
      {config.actionLabel && config.actionLink && (
        <a href={config.actionLink} className="btn btn-primary">
          {config.actionLabel}
        </a>
      )}
    </div>
  );
}

export function DataListWidget({ widget, data }: DataListWidgetProps) {
  const items = Array.isArray(data) ? data : [];

  if (items.length === 0 && widget.emptyState) {
    return <EmptyState config={widget.emptyState} />;
  }

  const avatarSizeMap = { sm: '32px', md: '40px', lg: '48px' };

  return (
    <div className="n3-data-list" style={{ width: '100%' }}>
      {widget.title && (
        <h2 style={{ margin: '0 0 1rem', fontSize: '1.5rem', fontWeight: 600 }}>{widget.title}</h2>
      )}

      <div className="list-items" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {items.map((item, idx) => {
          const { itemConfig } = widget;
          const avatarSize = avatarSizeMap[itemConfig.avatar?.size || 'md'];

          return (
            <div
              key={idx}
              className="list-item"
              style={{
                display: 'flex',
                gap: '1rem',
                padding: '1rem',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                backgroundColor: '#fff',
                transition: 'box-shadow 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              {itemConfig.avatar && (
                <div
                  className="list-item-avatar"
                  style={{
                    width: avatarSize,
                    height: avatarSize,
                    borderRadius: '50%',
                    backgroundColor: '#e5e7eb',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    overflow: 'hidden',
                  }}
                >
                  {itemConfig.avatar.field && item[itemConfig.avatar.field] ? (
                    <img
                      src={item[itemConfig.avatar.field]}
                      alt=""
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  ) : (
                    <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#6b7280' }}>
                      {itemConfig.avatar.fallback || '?'}
                    </span>
                  )}
                </div>
              )}

              <div className="list-item-content" style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                  <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#111827' }}>
                    {getFieldValue(itemConfig.title, item)}
                  </h3>
                  {itemConfig.badge && (
                    <span
                      className={`badge ${itemConfig.badge.style || ''}`}
                      style={{
                        padding: '0.125rem 0.5rem',
                        fontSize: '0.75rem',
                        borderRadius: '9999px',
                        backgroundColor: '#e5e7eb',
                        color: '#374151',
                      }}
                    >
                      {getFieldValue({ field: itemConfig.badge.field, text: itemConfig.badge.text }, item)}
                    </span>
                  )}
                </div>

                {itemConfig.subtitle && (
                  <p style={{ margin: '0 0 0.5rem', fontSize: '0.875rem', color: '#6b7280' }}>
                    {getFieldValue(itemConfig.subtitle, item)}
                  </p>
                )}

                {itemConfig.metadata && itemConfig.metadata.length > 0 && (
                  <div className="list-item-metadata" style={{ display: 'flex', gap: '1rem', fontSize: '0.75rem', color: '#9ca3af' }}>
                    {itemConfig.metadata.map((meta, metaIdx) => (
                      <span key={metaIdx} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        {meta.icon && <span>{meta.icon}</span>}
                        {meta.field
                          ? formatMetadata(item[meta.field], meta.format)
                          : meta.text
                          ? renderTemplate(meta.text, item)
                          : ''}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {itemConfig.actions && itemConfig.actions.length > 0 && (
                <div className="list-item-actions" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  {itemConfig.actions.map((action, actionIdx) => (
                    <button
                      key={actionIdx}
                      onClick={() => console.log(action.action, item)}
                      className="btn btn-sm"
                      style={{
                        padding: '0.375rem 0.75rem',
                        fontSize: '0.875rem',
                        border: '1px solid #d1d5db',
                        borderRadius: '4px',
                        backgroundColor: '#fff',
                        cursor: 'pointer',
                      }}
                    >
                      {action.icon && <span style={{ marginRight: '0.25rem' }}>{action.icon}</span>}
                      {action.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
