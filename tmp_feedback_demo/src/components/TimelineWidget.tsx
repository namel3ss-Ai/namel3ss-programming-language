import React, { useMemo } from 'react';

interface TimelineItem {
  timestamp: string;
  icon?: string;
  status?: string;
  title: {
    text?: string;
    field?: string;
  };
  description?: {
    text?: string;
    field?: string;
  };
}

interface EmptyStateConfig {
  icon?: string;
  title: string;
  message?: string;
  actionLabel?: string;
  actionLink?: string;
}

interface TimelineWidgetConfig {
  id: string;
  type: 'timeline';
  title?: string;
  source: {
    kind: string;
    name: string;
  };
  items: TimelineItem[];
  groupByDate?: boolean;
  emptyState?: EmptyStateConfig;
}

interface TimelineWidgetProps {
  widget: TimelineWidgetConfig;
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

function formatDate(timestamp: string): string {
  return new Date(timestamp).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

function formatTime(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  });
}

function getStatusColor(status?: string): string {
  if (status === 'success') return '#10b981';
  if (status === 'error') return '#ef4444';
  if (status === 'warning') return '#f59e0b';
  if (status === 'info') return '#3b82f6';
  return '#6b7280';
}

function EmptyState({ config }: { config: EmptyStateConfig }) {
  return (
    <div className="n3-empty-state" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
      {config.icon && (
        <div className="empty-state-icon" style={{ fontSize: '2rem', marginBottom: '1rem', opacity: 0.5 }}>
          ⏱️
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

export function TimelineWidget({ widget, data }: TimelineWidgetProps) {
  const items = Array.isArray(data) ? data : [];

  const groupedItems = useMemo(() => {
    if (!widget.groupByDate) {
      return { ungrouped: items };
    }

    const groups: Record<string, any[]> = {};
    items.forEach((item, idx) => {
      const itemConfig = widget.items[idx % widget.items.length];
      const timestamp = item[itemConfig.timestamp];
      if (timestamp) {
        const date = formatDate(timestamp);
        if (!groups[date]) groups[date] = [];
        groups[date].push({ ...item, _config: itemConfig, _timestamp: timestamp });
      }
    });

    return groups;
  }, [items, widget.items, widget.groupByDate]);

  if (items.length === 0 && widget.emptyState) {
    return <EmptyState config={widget.emptyState} />;
  }

  const renderTimelineItem = (item: any, itemConfig: TimelineItem, idx: number) => {
    const timestamp = item._timestamp || item[itemConfig.timestamp];
    const statusColor = getStatusColor(itemConfig.status || item.status);

    return (
      <div key={idx} className="timeline-item" style={{ display: 'flex', gap: '1rem', position: 'relative' }}>
        <div className="timeline-marker" style={{ position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div
            className="timeline-icon"
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '50%',
              backgroundColor: statusColor,
              color: '#fff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '0.875rem',
              fontWeight: 600,
              zIndex: 1,
            }}
          >
            {itemConfig.icon || '•'}
          </div>
          <div
            className="timeline-line"
            style={{
              width: '2px',
              flex: 1,
              backgroundColor: '#e5e7eb',
              marginTop: '-4px',
              minHeight: '20px',
            }}
          />
        </div>

        <div className="timeline-content" style={{ flex: 1, paddingBottom: '1.5rem' }}>
          <div className="timeline-time" style={{ fontSize: '0.75rem', color: '#9ca3af', marginBottom: '0.25rem' }}>
            {formatTime(timestamp)}
          </div>
          <h4 style={{ margin: '0 0 0.25rem', fontSize: '1rem', fontWeight: 600, color: '#111827' }}>
            {getFieldValue(itemConfig.title, item)}
          </h4>
          {itemConfig.description && (
            <p style={{ margin: 0, fontSize: '0.875rem', color: '#6b7280' }}>
              {getFieldValue(itemConfig.description, item)}
            </p>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="n3-timeline" style={{ width: '100%' }}>
      {widget.title && (
        <h2 style={{ margin: '0 0 1.5rem', fontSize: '1.5rem', fontWeight: 600 }}>{widget.title}</h2>
      )}

      {widget.groupByDate ? (
        <div className="timeline-groups">
          {Object.entries(groupedItems).map(([date, groupItems]) => (
            <div key={date} className="timeline-group" style={{ marginBottom: '2rem' }}>
              <h3 style={{ margin: '0 0 1rem', fontSize: '1.125rem', fontWeight: 600, color: '#374151' }}>
                {date}
              </h3>
              <div className="timeline-items">
                {groupItems.map((item: any, idx: number) =>
                  renderTimelineItem(item, item._config, idx)
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="timeline-items">
          {items.map((item, idx) => {
            const itemConfig = widget.items[idx % widget.items.length];
            return renderTimelineItem(item, itemConfig, idx);
          })}
        </div>
      )}
    </div>
  );
}
