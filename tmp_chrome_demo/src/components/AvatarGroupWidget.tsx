import React, { useState } from 'react';

interface AvatarItem {
  imageField?: string;
  nameField: string;
  statusField?: string;
  tooltipTemplate?: string;
}

interface AvatarGroupWidgetConfig {
  id: string;
  type: 'avatar_group';
  title?: string;
  source: {
    kind: string;
    name: string;
  };
  items: AvatarItem[];
  maxVisible?: number;
  size?: 'sm' | 'md' | 'lg';
  showStatus?: boolean;
}

interface AvatarGroupWidgetProps {
  widget: AvatarGroupWidgetConfig;
  data: unknown;
}

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  return name.substring(0, 2).toUpperCase();
}

function getStatusColor(status?: string): string {
  if (status === 'online') return '#10b981';
  if (status === 'offline') return '#9ca3af';
  if (status === 'busy') return '#ef4444';
  if (status === 'away') return '#f59e0b';
  return '#9ca3af';
}

function renderTemplate(template: string, item: any): string {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, key) => {
    return item[key] !== undefined ? String(item[key]) : '';
  });
}

export function AvatarGroupWidget({ widget, data }: AvatarGroupWidgetProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const items = Array.isArray(data) ? data : [];
  const maxVisible = widget.maxVisible || 5;
  const visibleItems = items.slice(0, maxVisible);
  const overflowCount = Math.max(0, items.length - maxVisible);

  const sizeMap = { sm: '32px', md: '40px', lg: '48px' };
  const size = sizeMap[widget.size || 'md'];

  if (items.length === 0) {
    return (
      <div className="n3-avatar-group" style={{ display: 'flex', alignItems: 'center' }}>
        {widget.title && (
          <span style={{ marginRight: '0.75rem', fontSize: '0.875rem', color: '#6b7280' }}>
            {widget.title}
          </span>
        )}
        <span style={{ fontSize: '0.875rem', color: '#9ca3af' }}>No users</span>
      </div>
    );
  }

  return (
    <div className="n3-avatar-group" style={{ display: 'flex', alignItems: 'center' }}>
      {widget.title && (
        <span style={{ marginRight: '0.75rem', fontSize: '0.875rem', fontWeight: 600, color: '#374151' }}>
          {widget.title}
        </span>
      )}

      <div className="avatar-list" style={{ display: 'flex', marginLeft: '-8px' }}>
        {visibleItems.map((item, idx) => {
          const itemConfig = widget.items[idx % widget.items.length];
          const imageUrl = itemConfig.imageField ? item[itemConfig.imageField] : null;
          const name = item[itemConfig.nameField] || 'Unknown';
          const status = widget.showStatus && itemConfig.statusField ? item[itemConfig.statusField] : null;
          const tooltip = itemConfig.tooltipTemplate
            ? renderTemplate(itemConfig.tooltipTemplate, item)
            : name;

          return (
            <div
              key={idx}
              className="avatar-wrapper"
              style={{ position: 'relative', marginLeft: '8px' }}
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              <div
                className="avatar"
                style={{
                  width: size,
                  height: size,
                  borderRadius: '50%',
                  backgroundColor: '#e5e7eb',
                  border: '2px solid #fff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                  cursor: 'pointer',
                  transition: 'transform 0.2s',
                  transform: hoveredIdx === idx ? 'scale(1.1)' : 'scale(1)',
                }}
              >
                {imageUrl ? (
                  <img
                    src={imageUrl}
                    alt={name}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                ) : (
                  <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#6b7280' }}>
                    {getInitials(name)}
                  </span>
                )}
              </div>

              {status && (
                <div
                  className="avatar-status"
                  style={{
                    position: 'absolute',
                    bottom: '0',
                    right: '0',
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    backgroundColor: getStatusColor(status),
                    border: '2px solid #fff',
                  }}
                />
              )}

              {hoveredIdx === idx && (
                <div
                  className="avatar-tooltip"
                  style={{
                    position: 'absolute',
                    bottom: '100%',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    marginBottom: '8px',
                    padding: '0.5rem 0.75rem',
                    backgroundColor: '#1f2937',
                    color: '#fff',
                    fontSize: '0.75rem',
                    borderRadius: '4px',
                    whiteSpace: 'nowrap',
                    pointerEvents: 'none',
                    zIndex: 10,
                  }}
                >
                  {tooltip}
                  <div
                    style={{
                      position: 'absolute',
                      top: '100%',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      width: 0,
                      height: 0,
                      borderLeft: '4px solid transparent',
                      borderRight: '4px solid transparent',
                      borderTop: '4px solid #1f2937',
                    }}
                  />
                </div>
              )}
            </div>
          );
        })}

        {overflowCount > 0 && (
          <div
            className="avatar-overflow"
            style={{
              width: size,
              height: size,
              borderRadius: '50%',
              backgroundColor: '#f3f4f6',
              border: '2px solid #fff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginLeft: '8px',
              fontSize: '0.75rem',
              fontWeight: 600,
              color: '#6b7280',
              cursor: 'pointer',
            }}
            title={`${overflowCount} more`}
          >
            +{overflowCount}
          </div>
        )}
      </div>
    </div>
  );
}
