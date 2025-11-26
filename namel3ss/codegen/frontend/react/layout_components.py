"""
Production-ready React layout primitive components.

Implements StackLayout, GridLayout, SplitLayout, TabsLayout, and AccordionLayout
with full accessibility, state management, and responsive behavior.
"""

LAYOUT_COMPONENTS_TSX = """
import React, { useState, useEffect, useRef, CSSProperties } from 'react';

// =============================================================================
// Type Definitions
// =============================================================================

interface StackLayoutProps {
  direction?: 'vertical' | 'horizontal';
  gap?: string | number;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'space_between' | 'space_around' | 'space_evenly';
  wrap?: boolean;
  children: React.ReactNode[];
  style?: Record<string, any>;
  className?: string;
}

interface GridLayoutProps {
  columns?: number | 'auto';
  minColumnWidth?: string;
  gap?: string | number;
  responsive?: boolean;
  children: React.ReactNode[];
  style?: Record<string, any>;
  className?: string;
}

interface SplitLayoutProps {
  left: React.ReactNode[];
  right: React.ReactNode[];
  ratio?: number;
  resizable?: boolean;
  orientation?: 'horizontal' | 'vertical';
  style?: Record<string, any>;
  className?: string;
}

interface TabItem {
  id: string;
  label: string;
  icon?: string;
  badge?: string | number;
  content: React.ReactNode[];
}

interface TabsLayoutProps {
  tabs: TabItem[];
  defaultTab?: string;
  persistState?: boolean;
  style?: Record<string, any>;
  className?: string;
}

interface AccordionItem {
  id: string;
  title: string;
  description?: string;
  icon?: string;
  defaultOpen?: boolean;
  content: React.ReactNode[];
}

interface AccordionLayoutProps {
  items: AccordionItem[];
  multiple?: boolean;
  style?: Record<string, any>;
  className?: string;
}

// =============================================================================
// Utility Functions
// =============================================================================

function normalizeGap(gap: string | number): string {
  if (typeof gap === 'number') {
    return `${gap}px`;
  }
  const gapMap: Record<string, string> = {
    small: '0.5rem',
    medium: '1rem',
    large: '1.5rem',
  };
  return gapMap[gap] || gap;
}

function normalizeJustify(justify: string): string {
  const justifyMap: Record<string, string> = {
    start: 'flex-start',
    center: 'center',
    end: 'flex-end',
    space_between: 'space-between',
    space_around: 'space-around',
    space_evenly: 'space-evenly',
  };
  return justifyMap[justify] || justify;
}

function normalizeAlign(align: string): string {
  const alignMap: Record<string, string> = {
    start: 'flex-start',
    center: 'center',
    end: 'flex-end',
    stretch: 'stretch',
  };
  return alignMap[align] || align;
}

// =============================================================================
// Stack Layout Component
// =============================================================================

export function StackLayout({
  direction = 'vertical',
  gap = 'medium',
  align = 'stretch',
  justify = 'start',
  wrap = false,
  children,
  style = {},
  className = '',
}: StackLayoutProps) {
  const stackStyle: CSSProperties = {
    display: 'flex',
    flexDirection: direction === 'vertical' ? 'column' : 'row',
    gap: normalizeGap(gap),
    alignItems: normalizeAlign(align),
    justifyContent: normalizeJustify(justify),
    flexWrap: wrap ? 'wrap' : 'nowrap',
    ...style,
  };

  return (
    <div className={`n3-stack-layout ${className}`} style={stackStyle}>
      {children.map((child, index) => (
        <React.Fragment key={index}>{child}</React.Fragment>
      ))}
    </div>
  );
}

// =============================================================================
// Grid Layout Component
// =============================================================================

export function GridLayout({
  columns = 'auto',
  minColumnWidth,
  gap = 'medium',
  responsive = true,
  children,
  style = {},
  className = '',
}: GridLayoutProps) {
  const gridStyle: CSSProperties = {
    display: 'grid',
    gap: normalizeGap(gap),
    ...style,
  };

  if (columns === 'auto' && minColumnWidth && responsive) {
    // Auto-fit responsive grid
    gridStyle.gridTemplateColumns = `repeat(auto-fit, minmax(${minColumnWidth}, 1fr))`;
  } else if (typeof columns === 'number') {
    // Fixed column count
    gridStyle.gridTemplateColumns = `repeat(${columns}, 1fr)`;
  } else {
    // Fallback
    gridStyle.gridTemplateColumns = `repeat(auto-fit, minmax(250px, 1fr))`;
  }

  return (
    <div className={`n3-grid-layout ${className}`} style={gridStyle}>
      {children.map((child, index) => (
        <React.Fragment key={index}>{child}</React.Fragment>
      ))}
    </div>
  );
}

// =============================================================================
// Split Layout Component
// =============================================================================

export function SplitLayout({
  left,
  right,
  ratio = 0.5,
  resizable = false,
  orientation = 'horizontal',
  style = {},
  className = '',
}: SplitLayoutProps) {
  const [splitRatio, setSplitRatio] = useState(ratio);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!resizable) return;
    e.preventDefault();
    setIsDragging(true);
  };

  useEffect(() => {
    if (!resizable || !isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      let newRatio: number;

      if (orientation === 'horizontal') {
        const x = e.clientX - rect.left;
        newRatio = Math.max(0.1, Math.min(0.9, x / rect.width));
      } else {
        const y = e.clientY - rect.top;
        newRatio = Math.max(0.1, Math.min(0.9, y / rect.height));
      }

      setSplitRatio(newRatio);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, orientation, resizable]);

  const containerStyle: CSSProperties = {
    display: 'flex',
    flexDirection: orientation === 'horizontal' ? 'row' : 'column',
    height: '100%',
    position: 'relative',
    ...style,
  };

  const leftStyle: CSSProperties = {
    flex: `0 0 ${splitRatio * 100}%`,
    overflow: 'auto',
  };

  const rightStyle: CSSProperties = {
    flex: `1 1 ${(1 - splitRatio) * 100}%`,
    overflow: 'auto',
  };

  const handleStyle: CSSProperties = {
    flex: '0 0 4px',
    cursor: orientation === 'horizontal' ? 'col-resize' : 'row-resize',
    backgroundColor: isDragging ? 'var(--color-primary, #3b82f6)' : 'var(--color-border, #e5e7eb)',
    transition: isDragging ? 'none' : 'background-color 0.2s',
    userSelect: 'none',
  };

  return (
    <div
      ref={containerRef}
      className={`n3-split-layout ${className}`}
      style={containerStyle}
    >
      <div className="n3-split-left" style={leftStyle}>
        {left.map((child, index) => (
          <React.Fragment key={index}>{child}</React.Fragment>
        ))}
      </div>

      {resizable && (
        <div
          className="n3-split-handle"
          style={handleStyle}
          onMouseDown={handleMouseDown}
          role="separator"
          aria-label="Resize panels"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
              e.preventDefault();
              setSplitRatio(Math.max(0.1, splitRatio - 0.05));
            } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
              e.preventDefault();
              setSplitRatio(Math.min(0.9, splitRatio + 0.05));
            }
          }}
        />
      )}

      <div className="n3-split-right" style={rightStyle}>
        {right.map((child, index) => (
          <React.Fragment key={index}>{child}</React.Fragment>
        ))}
      </div>
    </div>
  );
}

// =============================================================================
// Tabs Layout Component
// =============================================================================

export function TabsLayout({
  tabs,
  defaultTab,
  persistState = true,
  style = {},
  className = '',
}: TabsLayoutProps) {
  const [activeTab, setActiveTab] = useState(() => {
    if (persistState && typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const tabFromUrl = urlParams.get('tab');
      if (tabFromUrl && tabs.some((t) => t.id === tabFromUrl)) {
        return tabFromUrl;
      }
    }
    return defaultTab || (tabs.length > 0 ? tabs[0].id : '');
  });

  useEffect(() => {
    if (persistState && typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      urlParams.set('tab', activeTab);
      const newUrl = `${window.location.pathname}?${urlParams.toString()}`;
      window.history.replaceState({}, '', newUrl);
    }
  }, [activeTab, persistState]);

  const handleTabClick = (tabId: string) => {
    setActiveTab(tabId);
  };

  const handleKeyDown = (e: React.KeyboardEvent, tabId: string, index: number) => {
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      const nextIndex = (index + 1) % tabs.length;
      setActiveTab(tabs[nextIndex].id);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      const prevIndex = (index - 1 + tabs.length) % tabs.length;
      setActiveTab(tabs[prevIndex].id);
    } else if (e.key === 'Home') {
      e.preventDefault();
      setActiveTab(tabs[0].id);
    } else if (e.key === 'End') {
      e.preventDefault();
      setActiveTab(tabs[tabs.length - 1].id);
    }
  };

  const containerStyle: CSSProperties = {
    ...style,
  };

  const tabListStyle: CSSProperties = {
    display: 'flex',
    borderBottom: '2px solid var(--color-border, #e5e7eb)',
    gap: '0.5rem',
    marginBottom: '1rem',
    overflowX: 'auto',
  };

  const tabButtonStyle = (isActive: boolean): CSSProperties => ({
    padding: '0.75rem 1rem',
    border: 'none',
    background: 'none',
    cursor: 'pointer',
    fontWeight: isActive ? 600 : 400,
    color: isActive ? 'var(--color-primary, #3b82f6)' : 'var(--color-text, #374151)',
    borderBottom: isActive ? '2px solid var(--color-primary, #3b82f6)' : '2px solid transparent',
    marginBottom: '-2px',
    transition: 'all 0.2s',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    whiteSpace: 'nowrap',
  });

  const activeTabContent = tabs.find((tab) => tab.id === activeTab);

  return (
    <div className={`n3-tabs-layout ${className}`} style={containerStyle}>
      <div
        role="tablist"
        aria-label="Content tabs"
        style={tabListStyle}
      >
        {tabs.map((tab, index) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`tabpanel-${tab.id}`}
            id={`tab-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            style={tabButtonStyle(activeTab === tab.id)}
            onClick={() => handleTabClick(tab.id)}
            onKeyDown={(e) => handleKeyDown(e, tab.id, index)}
          >
            {tab.icon && <span className="tab-icon">{tab.icon}</span>}
            <span>{tab.label}</span>
            {tab.badge && (
              <span
                className="tab-badge"
                style={{
                  backgroundColor: 'var(--color-primary, #3b82f6)',
                  color: 'white',
                  borderRadius: '9999px',
                  padding: '0.125rem 0.5rem',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                }}
              >
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {activeTabContent && (
        <div
          role="tabpanel"
          id={`tabpanel-${activeTabContent.id}`}
          aria-labelledby={`tab-${activeTabContent.id}`}
          tabIndex={0}
        >
          {activeTabContent.content.map((child, index) => (
            <React.Fragment key={index}>{child}</React.Fragment>
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Accordion Layout Component
// =============================================================================

export function AccordionLayout({
  items,
  multiple = false,
  style = {},
  className = '',
}: AccordionLayoutProps) {
  const [openItems, setOpenItems] = useState<Set<string>>(() => {
    const defaultOpen = new Set<string>();
    items.forEach((item) => {
      if (item.defaultOpen) {
        defaultOpen.add(item.id);
      }
    });
    return defaultOpen;
  });

  const toggleItem = (itemId: string) => {
    setOpenItems((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        if (!multiple) {
          newSet.clear();
        }
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent, itemId: string) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggleItem(itemId);
    }
  };

  const containerStyle: CSSProperties = {
    ...style,
  };

  const itemStyle: CSSProperties = {
    borderBottom: '1px solid var(--color-border, #e5e7eb)',
  };

  const headerStyle = (isOpen: boolean): CSSProperties => ({
    padding: '1rem',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: isOpen ? 'var(--color-surface, #f9fafb)' : 'transparent',
    transition: 'background-color 0.2s',
    userSelect: 'none',
  });

  const contentStyle = (isOpen: boolean): CSSProperties => ({
    maxHeight: isOpen ? '10000px' : '0',
    overflow: 'hidden',
    transition: 'max-height 0.3s ease-in-out',
    padding: isOpen ? '1rem' : '0 1rem',
  });

  return (
    <div className={`n3-accordion-layout ${className}`} style={containerStyle}>
      {items.map((item) => {
        const isOpen = openItems.has(item.id);
        return (
          <div key={item.id} className="n3-accordion-item" style={itemStyle}>
            <div
              role="button"
              aria-expanded={isOpen}
              aria-controls={`accordion-content-${item.id}`}
              id={`accordion-header-${item.id}`}
              tabIndex={0}
              style={headerStyle(isOpen)}
              onClick={() => toggleItem(item.id)}
              onKeyDown={(e) => handleKeyDown(e, item.id)}
            >
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {item.icon && <span className="accordion-icon">{item.icon}</span>}
                  <span style={{ fontWeight: 600, fontSize: '1rem' }}>{item.title}</span>
                </div>
                {item.description && (
                  <div style={{ fontSize: '0.875rem', color: 'var(--color-text-muted, #6b7280)', marginTop: '0.25rem' }}>
                    {item.description}
                  </div>
                )}
              </div>
              <svg
                style={{
                  width: '1.25rem',
                  height: '1.25rem',
                  transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>

            <div
              id={`accordion-content-${item.id}`}
              role="region"
              aria-labelledby={`accordion-header-${item.id}`}
              style={contentStyle(isOpen)}
            >
              {item.content.map((child, index) => (
                <React.Fragment key={index}>{child}</React.Fragment>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
"""


def generate_layout_components() -> str:
    """Generate the layout components TypeScript/React code."""
    return LAYOUT_COMPONENTS_TSX
