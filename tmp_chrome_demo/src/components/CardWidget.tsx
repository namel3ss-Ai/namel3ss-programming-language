import type { CSSProperties } from "react";

interface EmptyStateConfig {
  icon?: string;
  iconSize?: "small" | "medium" | "large";
  title: string;
  message?: string;
  actionLabel?: string;
  actionLink?: string;
}

interface BadgeConfig {
  field?: string;
  text?: string;
  style?: string;
  transform?: string | Record<string, any>;
  icon?: string;
  condition?: string;
}

interface FieldValueConfig {
  field?: string;
  text?: string;
  format?: string;
  style?: string;
  transform?: string | Record<string, any>;
}

interface InfoGridItem {
  icon?: string;
  label?: string;
  field?: string;
  values?: FieldValueConfig[];
}

interface CardSection {
  type: string;
  condition?: string;
  style?: string;
  title?: string;
  icon?: string;
  columns?: number;
  items?: InfoGridItem[];
  content?: Record<string, any>;
  listItems?: Record<string, any>;
}

interface ConditionalAction {
  label: string;
  icon?: string;
  style?: string;
  action?: string;
  link?: string;
  params?: string;
  condition?: string;
}

interface CardItemConfig {
  type: string;
  style?: string;
  stateClass?: Record<string, string>;
  roleClass?: string;
  header?: {
    title?: string;
    subtitle?: string;
    badges?: BadgeConfig[];
    avatar?: Record<string, any>;
  };
  sections?: CardSection[];
  actions?: ConditionalAction[];
  footer?: {
    condition?: string;
    text?: string;
    style?: string;
    left?: Record<string, any>;
    right?: Record<string, any>;
  };
  avatar?: Record<string, any>;
  content?: any;
  body?: Record<string, any>;
  attachments?: Record<string, any>;
  badge?: BadgeConfig | Record<string, any>;
}

interface CardWidgetConfig {
  id: string;
  type: "card";
  title: string;
  source: {
    kind: string;
    name: string;
  };
  emptyState?: EmptyStateConfig;
  itemConfig?: CardItemConfig;
  groupBy?: string;
}

interface CardWidgetProps {
  widget: CardWidgetConfig;
  data: unknown;
}

function evaluateCondition(condition: string | undefined, item: any): boolean {
  if (!condition) return true;

  try {
    // Simple expression evaluation (production should use a proper parser)
    const expr = condition
      .replace(/(\w+)/g, (match) => {
        if (match === 'null' || match === 'true' || match === 'false') return match;
        return `item.${match}`;
      });

    // eslint-disable-next-line no-new-func
    return new Function('item', `return ${expr}`)(item);
  } catch {
    return false;
  }
}

function applyTransform(value: any, transform: string | Record<string, any> | undefined): any {
  if (!transform) return value;

  if (typeof transform === 'string') {
    if (transform === 'humanize') {
      return String(value).replace(/_/g, ' ').replace(/\w/g, (l) => l.toUpperCase());
    }
    if (transform === 'relative') {
      // Simplified relative time (production would use date-fns or similar)
      return String(value);
    }
  } else if (typeof transform === 'object') {
    if (transform.format) {
      // Simplified date formatting (production would use date-fns)
      return String(value);
    }
    if (transform.truncate) {
      const str = String(value);
      return str.length > transform.truncate ? str.substring(0, transform.truncate) + '...' : str;
    }
  }

  return value;
}

function renderTemplate(template: string, item: any): string {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, key) => {
    return item[key] !== undefined ? String(item[key]) : '';
  });
}

function renderFieldValue(config: FieldValueConfig, item: any): string {
  if (config.text) {
    return renderTemplate(config.text, item);
  }
  if (config.field) {
    let value = item[config.field];
    value = applyTransform(value, config.transform);
    return String(value || '—');
  }
  return '—';
}

function EmptyState({ config }: { config: EmptyStateConfig }) {
  const iconSizeClass = config.iconSize === 'large' ? 'text-4xl' : config.iconSize === 'small' ? 'text-xl' : 'text-2xl';

  return (
    <div 
      className="n3-empty-state" 
      role="status"
      aria-label="No items to display"
      style={{ textAlign: 'center', padding: '3rem 1rem' }}
    >
      {config.icon && (
        <div 
          className={`empty-state-icon ${iconSizeClass}`} 
          aria-hidden="true"
          style={{ marginBottom: '1rem', opacity: 0.5 }}
        >
          📅
        </div>
      )}
      <h3 style={{ marginBottom: '0.5rem', fontWeight: 600 }}>{config.title}</h3>
      {config.message && <p style={{ color: 'var(--text-muted, #64748b)' }}>{config.message}</p>}
      {config.actionLabel && config.actionLink && (
        <a 
          href={config.actionLink} 
          className="n3-empty-state-action"
          style={{ marginTop: '1rem', display: 'inline-block' }}
        >
          {config.actionLabel}
        </a>
      )}
    </div>
  );
}

function InfoGrid({ section, item }: { section: CardSection; item: any }) {
  const minColumnWidth = section.columns === 1 ? '100%' : '200px';
  const maxColumns = section.columns || 2;

  const gridStyle: CSSProperties = {
    display: 'grid',
    gridTemplateColumns: `repeat(auto-fit, minmax(${minColumnWidth}, 1fr))`,
    maxWidth: maxColumns === 1 ? '100%' : undefined,
    gap: 'var(--spacing-md, 1rem)',
    marginBottom: 'var(--spacing-md, 1rem)',
  };

  const sectionId = `info-grid-${Math.random().toString(36).substr(2, 9)}`;
  const titleId = section.title ? `${sectionId}-title` : undefined;

  return (
    <section 
      className="info-grid" 
      aria-labelledby={titleId}
      style={gridStyle}
    >
      {section.title && (
        <h3 
          id={titleId}
          className="info-grid-title"
          style={{ 
            gridColumn: '1 / -1',
            fontSize: 'var(--font-size-md, 1rem)',
            fontWeight: 600,
            marginBottom: 'var(--spacing-sm, 0.5rem)'
          }}
        >
          {section.title}
        </h3>
      )}
      {section.items?.map((gridItem, idx) => (
        <div key={idx} className="info-grid-item">
          {gridItem.label && (
            <dt 
              className="info-grid-label"
              style={{ 
                fontWeight: 500, 
                marginBottom: 'var(--spacing-xs, 0.25rem)', 
                fontSize: 'var(--font-size-sm, 0.875rem)',
                color: 'var(--text-secondary, #64748b)'
              }}
            >
              {gridItem.icon && (
                <span 
                  className="info-grid-icon" 
                  aria-hidden="true"
                  style={{ marginRight: 'var(--spacing-xs, 0.5rem)' }}
                >
                  📍
                </span>
              )}
              {gridItem.label}
            </dt>
          )}
          {gridItem.values?.map((valueConfig, vidx) => (
            <dd 
              key={vidx} 
              className={`info-grid-value ${valueConfig.style || ''}`}
              style={{ 
                margin: 0,
                fontSize: 'var(--font-size-base, 1rem)',
                color: 'var(--text-primary, #0f172a)'
              }}
            >
              {renderFieldValue(valueConfig, item)}
            </dd>
          ))}
        </div>
      ))}
    </section>
  );
}

function TextSection({ section, item }: { section: CardSection; item: any }) {
  if (!evaluateCondition(section.condition, item)) return null;

  return (
    <section 
      className={`text-section ${section.style || ''}`} 
      style={{ marginBottom: 'var(--spacing-md, 1rem)' }}
    >
      {section.content?.label && (
        <strong 
          className="text-section-label"
          style={{ marginRight: 'var(--spacing-sm, 0.5rem)' }}
        >
          {section.content.label}
        </strong>
      )}
      {section.content?.text && (
        <span className="text-section-content">
          {renderTemplate(section.content.text, item)}
        </span>
      )}
    </section>
  );
}

function CardItem({ item, config }: { item: any; config: CardItemConfig }) {
  const stateClasses = Object.entries(config.stateClass || {})
    .filter(([_, condition]) => evaluateCondition(condition, item))
    .map(([className]) => className)
    .join(' ');

  // Determine semantic type for proper HTML element and ARIA role
  const cardType = config.type || 'card';
  const isMessageBubble = cardType === 'message_bubble';
  const isArticle = cardType === 'article_card';
  const ariaRole = isMessageBubble ? 'article' : undefined;

  const cardId = `card-${Math.random().toString(36).substr(2, 9)}`;
  const headerId = config.header?.title ? `${cardId}-header` : undefined;

  return (
    <article 
      className={`n3-card-item n3-card-${cardType} ${config.style || ''} ${config.roleClass || ''} ${stateClasses}`}
      role={ariaRole}
      aria-labelledby={headerId}
      style={{ 
        border: '1px solid var(--border-color, #e2e8f0)', 
        borderRadius: 'var(--radius-md, 0.5rem)', 
        padding: 'var(--spacing-lg, 1rem)', 
        marginBottom: 'var(--spacing-lg, 1rem)',
        backgroundColor: 'var(--surface, white)'
      }}
    >
      {/* Header */}
      {config.header && (
        <header 
          id={headerId}
          className="n3-card-header" 
          style={{ marginBottom: 'var(--spacing-md, 1rem)' }}
        >
          {config.header.avatar && (
            <div 
              className="n3-card-avatar" 
              aria-hidden="true"
              style={{ 
                width: '40px', 
                height: '40px', 
                borderRadius: '50%',
                marginBottom: 'var(--spacing-sm, 0.5rem)'
              }}
            >
              {typeof config.header.avatar === 'string' ? (
                <img 
                  src={config.header.avatar} 
                  alt=""
                  style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '50%' }}
                />
              ) : config.header.avatar.image_url ? (
                <img 
                  src={config.header.avatar.image_url} 
                  alt={config.header.avatar.alt_text || ''}
                  style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '50%' }}
                />
              ) : null}
            </div>
          )}

          {config.header.title && (
            <h3 
              className="n3-card-title"
              style={{ 
                margin: 0,
                fontSize: 'var(--font-size-lg, 1.125rem)',
                fontWeight: 600,
                color: 'var(--text-primary, #0f172a)'
              }}
            >
              {renderTemplate(config.header.title, item)}
            </h3>
          )}

          {config.header.subtitle && (
            <p 
              className="n3-card-subtitle"
              style={{ 
                margin: 'var(--spacing-xs, 0.25rem) 0 0',
                fontSize: 'var(--font-size-sm, 0.875rem)',
                color: 'var(--text-secondary, #64748b)'
              }}
            >
              {renderTemplate(config.header.subtitle, item)}
            </p>
          )}

          {config.header.badges && config.header.badges.length > 0 && (
            <div 
              className="n3-card-badges"
              role="list"
              aria-label="Status badges"
              style={{ 
                display: 'flex', 
                gap: 'var(--spacing-xs, 0.5rem)', 
                marginTop: 'var(--spacing-sm, 0.5rem)',
                flexWrap: 'wrap'
              }}
            >
              {config.header.badges
                .filter((badge) => evaluateCondition(badge.condition, item))
                .map((badge, idx) => {
                  const badgeText = badge.text || (badge.field ? applyTransform(item[badge.field], badge.transform) : '');
                  return (
                    <span 
                      key={idx} 
                      className={`n3-badge ${badge.style || ''}`}
                      role="listitem"
                      style={{ 
                        padding: '0.25rem 0.75rem', 
                        borderRadius: 'var(--radius-full, 1rem)', 
                        fontSize: 'var(--font-size-xs, 0.75rem)',
                        fontWeight: 500,
                        backgroundColor: 'var(--surface-secondary, #f1f5f9)',
                        border: '1px solid var(--border-color, #e2e8f0)',
                        color: 'var(--text-primary, #0f172a)'
                      }}
                    >
                      {badge.icon && (
                        <span aria-hidden="true" style={{ marginRight: '0.25rem' }}>
                          {badge.icon}
                        </span>
                      )}
                      {badgeText}
                    </span>
                  );
                })}
            </div>
          )}
        </header>
      )}

      {/* Sections */}
      {config.sections?.map((section, idx) => {
        if (!evaluateCondition(section.condition, item)) return null;

        if (section.type === 'info_grid') {
          return <InfoGrid key={idx} section={section} item={item} />;
        }
        if (section.type === 'text_section') {
          return <TextSection key={idx} section={section} item={item} />;
        }
        return null;
      })}

      {/* Actions */}
      {config.actions && config.actions.length > 0 && (
        <nav 
          className="n3-card-actions"
          aria-label="Card actions"
          style={{ 
            display: 'flex', 
            gap: 'var(--spacing-sm, 0.5rem)', 
            marginTop: 'var(--spacing-md, 1rem)',
            flexWrap: 'wrap'
          }}
        >
          {config.actions
            .filter((action) => evaluateCondition(action.condition, item))
            .map((action, idx) => (
              <button 
                key={idx} 
                className={`n3-btn n3-btn-${action.style || 'primary'}`}
                aria-label={action.label}
                style={{ 
                  padding: 'var(--spacing-sm, 0.5rem) var(--spacing-md, 1rem)', 
                  borderRadius: 'var(--radius-sm, 0.25rem)', 
                  border: '1px solid var(--border-color, #e2e8f0)',
                  background: action.style === 'secondary' ? 'var(--surface, white)' : 'var(--primary, #3b82f6)',
                  color: action.style === 'secondary' ? 'var(--text-primary, #0f172a)' : 'white',
                  cursor: 'pointer',
                  fontSize: 'var(--font-size-sm, 0.875rem)',
                  fontWeight: 500,
                  transition: 'all 0.2s ease'
                }}
                onClick={() => {
                  if (action.link) {
                    window.location.href = renderTemplate(action.link, item);
                  } else if (action.action) {
                    console.log('Action:', action.action, action.params ? renderTemplate(action.params, item) : '');
                  }
                }}
              >
                {action.icon && (
                  <span aria-hidden="true" style={{ marginRight: 'var(--spacing-xs, 0.5rem)' }}>
                    ⚡
                  </span>
                )}
                {action.label}
              </button>
            ))}
        </nav>
      )}

      {/* Footer */}
      {config.footer && evaluateCondition(config.footer.condition, item) && (
        <footer 
          className={`n3-card-footer ${config.footer.style || ''}`} 
          style={{ 
            marginTop: 'var(--spacing-md, 1rem)', 
            paddingTop: 'var(--spacing-md, 1rem)', 
            borderTop: '1px solid var(--border-color, #e2e8f0)', 
            fontSize: 'var(--font-size-sm, 0.875rem)',
            color: 'var(--text-secondary, #64748b)'
          }}
        >
          {config.footer.text && renderTemplate(config.footer.text, item)}
        </footer>
      )}
    </article>
  );
}

export default function CardWidget({ widget, data }: CardWidgetProps) {
  const items = Array.isArray(data) ? data : [];
  const widgetId = `widget-${widget.id}`;
  const titleId = `${widgetId}-title`;

  return (
    <section 
      className="n3-widget n3-card-widget" 
      aria-labelledby={titleId}
    >
      <h2 
        id={titleId}
        className="n3-widget-title"
        style={{ 
          marginBottom: 'var(--spacing-md, 1rem)',
          fontSize: 'var(--font-size-xl, 1.5rem)',
          fontWeight: 700,
          color: 'var(--text-primary, #0f172a)'
        }}
      >
        {widget.title}
      </h2>
      {items.length === 0 && widget.emptyState ? (
        <EmptyState config={widget.emptyState} />
      ) : (
        <div className="n3-card-list" role="list">
          {items.map((item, idx) => (
            widget.itemConfig ? (
              <CardItem key={idx} item={item} config={widget.itemConfig} />
            ) : (
              <article 
                key={idx} 
                className="n3-card-item n3-card-fallback"
                style={{ 
                  padding: 'var(--spacing-md, 1rem)', 
                  border: '1px solid var(--border-color, #e2e8f0)', 
                  marginBottom: 'var(--spacing-sm, 0.5rem)',
                  borderRadius: 'var(--radius-sm, 0.25rem)',
                  backgroundColor: 'var(--surface, white)'
                }}
              >
                <pre style={{ margin: 0, fontSize: 'var(--font-size-sm, 0.875rem)' }}>
                  {JSON.stringify(item, null, 2)}
                </pre>
              </article>
            )
          ))}
        </div>
      )}
    </section>
  );
}
