"""
React component generators for declarative UI (Card, List, InfoGrid, etc.).

This module generates TypeScript React components for the declarative UI syntax
introduced in Namel3ss 0.6.0. These components support semantic layout primitives
like cards, info grids, badges, conditional actions, and empty states.
"""

import textwrap
from pathlib import Path
from typing import Any, Dict

from .utils import write_file


def write_card_widget(components_dir: Path) -> None:
    """
    Generate CardWidget.tsx for rendering card-based lists.
    
    Supports:
    - Empty states with icons
    - Card sections (info_grid, text_section, etc.)
    - Header badges
    - Conditional actions
    - Footer content
    """
    content = textwrap.dedent(
        """
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
              return String(value).replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
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
          return template.replace(/\\{\\{\\s*(\\w+)\\s*\\}\\}/g, (_, key) => {
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
            <div className="n3-empty-state" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
              {config.icon && (
                <div className={`empty-state-icon ${iconSizeClass}`} style={{ marginBottom: '1rem', opacity: 0.5 }}>
                  📅
                </div>
              )}
              <h3 style={{ marginBottom: '0.5rem', fontWeight: 600 }}>{config.title}</h3>
              {config.message && <p style={{ color: 'var(--text-muted, #64748b)' }}>{config.message}</p>}
              {config.actionLabel && config.actionLink && (
                <a href={config.actionLink} style={{ marginTop: '1rem', display: 'inline-block' }}>
                  {config.actionLabel}
                </a>
              )}
            </div>
          );
        }

        function InfoGrid({ section, item }: { section: CardSection; item: any }) {
          const gridStyle: CSSProperties = {
            display: 'grid',
            gridTemplateColumns: `repeat(${section.columns || 2}, 1fr)`,
            gap: '1rem',
            marginBottom: '1rem',
          };

          return (
            <div className="info-grid" style={gridStyle}>
              {section.items?.map((gridItem, idx) => (
                <div key={idx} className="info-grid-item">
                  {gridItem.label && (
                    <div style={{ fontWeight: 500, marginBottom: '0.25rem', fontSize: '0.875rem' }}>
                      {gridItem.icon && <span style={{ marginRight: '0.5rem' }}>📍</span>}
                      {gridItem.label}
                    </div>
                  )}
                  {gridItem.values?.map((valueConfig, vidx) => (
                    <div key={vidx} className={valueConfig.style || ''}>
                      {renderFieldValue(valueConfig, item)}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          );
        }

        function TextSection({ section, item }: { section: CardSection; item: any }) {
          if (!evaluateCondition(section.condition, item)) return null;
          
          return (
            <div className={`text-section ${section.style || ''}`} style={{ marginBottom: '1rem' }}>
              {section.content?.label && (
                <strong style={{ marginRight: '0.5rem' }}>{section.content.label}</strong>
              )}
              {section.content?.text && <span>{renderTemplate(section.content.text, item)}</span>}
            </div>
          );
        }

        function CardItem({ item, config }: { item: any; config: CardItemConfig }) {
          const stateClasses = Object.entries(config.stateClass || {})
            .filter(([_, condition]) => evaluateCondition(condition, item))
            .map(([className]) => className)
            .join(' ');

          return (
            <div className={`card-item ${config.type} ${config.style || ''} ${stateClasses}`} 
                 style={{ 
                   border: '1px solid #e2e8f0', 
                   borderRadius: '0.5rem', 
                   padding: '1rem', 
                   marginBottom: '1rem',
                   backgroundColor: 'white'
                 }}>
              {/* Header */}
              {config.header && (
                <div className="card-header" style={{ marginBottom: '1rem' }}>
                  {config.header.title && <h4 style={{ margin: 0 }}>{renderTemplate(config.header.title, item)}</h4>}
                  {config.header.badges && config.header.badges.length > 0 && (
                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                      {config.header.badges
                        .filter((badge) => evaluateCondition(badge.condition, item))
                        .map((badge, idx) => {
                          const badgeText = badge.text || (badge.field ? applyTransform(item[badge.field], badge.transform) : '');
                          return (
                            <span key={idx} className={`badge ${badge.style || ''}`} 
                                  style={{ 
                                    padding: '0.25rem 0.75rem', 
                                    borderRadius: '1rem', 
                                    fontSize: '0.75rem',
                                    backgroundColor: '#f1f5f9',
                                    border: '1px solid #e2e8f0'
                                  }}>
                              {badgeText}
                            </span>
                          );
                        })}
                    </div>
                  )}
                </div>
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
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                  {config.actions
                    .filter((action) => evaluateCondition(action.condition, item))
                    .map((action, idx) => (
                      <button key={idx} className={`btn-${action.style || 'primary'}`}
                              style={{ 
                                padding: '0.5rem 1rem', 
                                borderRadius: '0.25rem', 
                                border: '1px solid #e2e8f0',
                                background: action.style === 'secondary' ? 'white' : '#3b82f6',
                                color: action.style === 'secondary' ? 'inherit' : 'white',
                                cursor: 'pointer'
                              }}
                              onClick={() => {
                                if (action.link) {
                                  window.location.href = renderTemplate(action.link, item);
                                } else if (action.action) {
                                  console.log('Action:', action.action, action.params ? renderTemplate(action.params, item) : '');
                                }
                              }}>
                        {action.icon && <span style={{ marginRight: '0.5rem' }}>⚡</span>}
                        {action.label}
                      </button>
                    ))}
                </div>
              )}

              {/* Footer */}
              {config.footer && evaluateCondition(config.footer.condition, item) && (
                <div className={`card-footer ${config.footer.style || ''}`} 
                     style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #e2e8f0', fontSize: '0.875rem' }}>
                  {config.footer.text && renderTemplate(config.footer.text, item)}
                </div>
              )}
            </div>
          );
        }

        export default function CardWidget({ widget, data }: CardWidgetProps) {
          const items = Array.isArray(data) ? data : [];

          return (
            <section className="n3-widget n3-card-widget">
              <h3 style={{ marginBottom: '1rem' }}>{widget.title}</h3>
              {items.length === 0 && widget.emptyState ? (
                <EmptyState config={widget.emptyState} />
              ) : (
                <div className="card-list">
                  {items.map((item, idx) => (
                    widget.itemConfig ? (
                      <CardItem key={idx} item={item} config={widget.itemConfig} />
                    ) : (
                      <div key={idx} style={{ padding: '1rem', border: '1px solid #e2e8f0', marginBottom: '0.5rem' }}>
                        {JSON.stringify(item)}
                      </div>
                    )
                  ))}
                </div>
              )}
            </section>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "CardWidget.tsx", content)


def write_list_widget(components_dir: Path) -> None:
    """Generate ListWidget.tsx for generic list displays with semantic types."""
    content = textwrap.dedent(
        """
        import CardWidget from "./CardWidget";

        interface ListWidgetConfig {
          id: string;
          type: "list";
          title: string;
          source: {
            kind: string;
            name: string;
          };
          listType?: string;
          emptyState?: any;
          itemConfig?: any;
          enableSearch?: boolean;
          searchPlaceholder?: string;
          filters?: any[];
          pageSize?: number;
          columns?: number;
        }

        interface ListWidgetProps {
          widget: ListWidgetConfig;
          data: unknown;
        }

        export default function ListWidget({ widget, data }: ListWidgetProps) {
          // For now, delegate to CardWidget as they share similar structure
          // In production, you'd differentiate based on listType
          const cardWidget = {
            ...widget,
            type: "card" as const,
          };

          return <CardWidget widget={cardWidget} data={data} />;
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "ListWidget.tsx", content)


def write_data_table_widget(components_dir: Path) -> None:
    """
    Generate DataTableWidget.tsx for professional data tables.
    
    Supports:
    - Column configuration with sorting
    - Toolbar with search, filters, bulk actions
    - Row actions (view, edit, delete)
    - Pagination
    - Empty states
    """
    content = textwrap.dedent(
        """
        import React, { useState, useMemo } from 'react';
        import {
          useReactTable,
          getCoreRowModel,
          getSortedRowModel,
          getFilteredRowModel,
          getPaginationRowModel,
          flexRender,
          createColumnHelper,
          type ColumnDef,
          type SortingState,
        } from '@tanstack/react-table';

        interface ColumnConfig {
          field: string;
          header: string;
          width?: string;
          sortable?: boolean;
          format?: string | Record<string, any>;
          align?: 'left' | 'center' | 'right';
        }

        interface ToolbarConfig {
          searchable?: boolean;
          searchFields?: string[];
          filters?: Array<{
            field: string;
            label: string;
            type: 'select' | 'multiselect' | 'date' | 'range';
            options?: Array<{ value: string; label: string }>;
          }>;
          bulkActions?: Array<{
            label: string;
            action: string;
            icon?: string;
            requiresSelection?: boolean;
          }>;
        }

        interface RowAction {
          label: string;
          action: string;
          icon?: string;
          condition?: string;
        }

        interface EmptyStateConfig {
          icon?: string;
          title: string;
          message?: string;
          actionLabel?: string;
          actionLink?: string;
        }

        interface DataTableWidgetConfig {
          id: string;
          type: 'data_table';
          title?: string;
          source: {
            kind: string;
            name: string;
          };
          columns: ColumnConfig[];
          toolbar?: ToolbarConfig;
          rowActions?: RowAction[];
          rowsPerPage?: number;
          emptyState?: EmptyStateConfig;
        }

        interface DataTableWidgetProps {
          widget: DataTableWidgetConfig;
          data: unknown;
        }

        function formatValue(value: any, format?: string | Record<string, any>): string {
          if (value === null || value === undefined) return '—';
          
          if (typeof format === 'string') {
            if (format === 'currency') {
              return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(Number(value));
            }
            if (format === 'percentage') {
              return `${Number(value).toFixed(1)}%`;
            }
            if (format === 'date') {
              return new Date(value).toLocaleDateString();
            }
            if (format === 'datetime') {
              return new Date(value).toLocaleString();
            }
          } else if (typeof format === 'object') {
            if (format.type === 'number' && format.decimals !== undefined) {
              return Number(value).toFixed(format.decimals);
            }
          }
          
          return String(value);
        }

        function EmptyState({ config }: { config: EmptyStateConfig }) {
          return (
            <div className="n3-empty-state" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
              {config.icon && (
                <div className="empty-state-icon" style={{ fontSize: '2rem', marginBottom: '1rem', opacity: 0.5 }}>
                  📊
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

        export function DataTableWidget({ widget, data }: DataTableWidgetProps) {
          const [sorting, setSorting] = useState<SortingState>([]);
          const [globalFilter, setGlobalFilter] = useState('');
          const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());

          const tableData = useMemo(() => {
            if (!data || !Array.isArray(data)) return [];
            return data;
          }, [data]);

          const columns = useMemo<ColumnDef<any>[]>(() => {
            const cols: ColumnDef<any>[] = widget.columns.map((col) => ({
              accessorKey: col.field,
              header: col.header,
              cell: (info) => formatValue(info.getValue(), col.format),
              enableSorting: col.sortable !== false,
              size: col.width ? parseInt(col.width) : undefined,
            }));

            if (widget.rowActions && widget.rowActions.length > 0) {
              cols.push({
                id: 'actions',
                header: 'Actions',
                cell: ({ row }) => (
                  <div className="row-actions" style={{ display: 'flex', gap: '0.5rem' }}>
                    {widget.rowActions?.map((action, idx) => (
                      <button
                        key={idx}
                        onClick={() => console.log(action.action, row.original)}
                        className="btn btn-sm"
                        title={action.label}
                      >
                        {action.icon || action.label}
                      </button>
                    ))}
                  </div>
                ),
              });
            }

            return cols;
          }, [widget.columns, widget.rowActions]);

          const table = useReactTable({
            data: tableData,
            columns,
            state: {
              sorting,
              globalFilter,
            },
            onSortingChange: setSorting,
            onGlobalFilterChange: setGlobalFilter,
            getCoreRowModel: getCoreRowModel(),
            getSortedRowModel: getSortedRowModel(),
            getFilteredRowModel: getFilteredRowModel(),
            getPaginationRowModel: getPaginationRowModel(),
            initialState: {
              pagination: {
                pageSize: widget.rowsPerPage || 10,
              },
            },
          });

          if (tableData.length === 0 && widget.emptyState) {
            return <EmptyState config={widget.emptyState} />;
          }

          return (
            <div className="n3-data-table" style={{ width: '100%' }}>
              {widget.title && (
                <h2 style={{ margin: '0 0 1rem', fontSize: '1.5rem', fontWeight: 600 }}>{widget.title}</h2>
              )}
              
              {widget.toolbar?.searchable && (
                <div className="table-toolbar" style={{ marginBottom: '1rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <input
                    type="text"
                    placeholder="Search..."
                    value={globalFilter ?? ''}
                    onChange={(e) => setGlobalFilter(e.target.value)}
                    style={{ padding: '0.5rem', border: '1px solid #ddd', borderRadius: '4px', flex: 1 }}
                  />
                  {widget.toolbar.bulkActions && selectedRows.size > 0 && (
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      {widget.toolbar.bulkActions.map((action, idx) => (
                        <button key={idx} className="btn btn-sm">
                          {action.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div style={{ overflowX: 'auto', border: '1px solid #ddd', borderRadius: '4px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    {table.getHeaderGroups().map((headerGroup) => (
                      <tr key={headerGroup.id} style={{ backgroundColor: '#f5f5f5' }}>
                        {headerGroup.headers.map((header) => (
                          <th
                            key={header.id}
                            style={{
                              padding: '0.75rem',
                              textAlign: 'left',
                              fontWeight: 600,
                              borderBottom: '2px solid #ddd',
                              cursor: header.column.getCanSort() ? 'pointer' : 'default',
                            }}
                            onClick={header.column.getToggleSortingHandler()}
                          >
                            {flexRender(header.column.columnDef.header, header.getContext())}
                            {header.column.getIsSorted() && (
                              <span style={{ marginLeft: '0.25rem' }}>
                                {header.column.getIsSorted() === 'asc' ? '↑' : '↓'}
                              </span>
                            )}
                          </th>
                        ))}
                      </tr>
                    ))}
                  </thead>
                  <tbody>
                    {table.getRowModel().rows.map((row) => (
                      <tr
                        key={row.id}
                        style={{
                          borderBottom: '1px solid #eee',
                          transition: 'background-color 0.2s',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#f9f9f9';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                        }}
                      >
                        {row.getVisibleCells().map((cell) => (
                          <td key={cell.id} style={{ padding: '0.75rem' }}>
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="table-pagination" style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  Showing {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1} to{' '}
                  {Math.min((table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize, tableData.length)} of{' '}
                  {tableData.length} results
                </div>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    onClick={() => table.previousPage()}
                    disabled={!table.getCanPreviousPage()}
                    className="btn btn-sm"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => table.nextPage()}
                    disabled={!table.getCanNextPage()}
                    className="btn btn-sm"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "DataTableWidget.tsx", content)


def write_data_list_widget(components_dir: Path) -> None:
    """
    Generate DataListWidget.tsx for activity feeds and item lists.
    
    Supports:
    - Avatar display
    - Title and subtitle templates
    - Metadata (timestamps, tags)
    - Status badges
    - Action buttons
    - Empty states
    """
    content = textwrap.dedent(
        """
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
          return template.replace(/\\{\\{\\s*(\\w+)\\s*\\}\\}/g, (_, key) => {
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
        """
    ).strip() + "\n"
    write_file(components_dir / "DataListWidget.tsx", content)


def write_stat_summary_widget(components_dir: Path) -> None:
    """
    Generate StatSummaryWidget.tsx for KPI cards.
    
    Supports:
    - Primary value display
    - Delta/change indicators
    - Trend direction (up/down/neutral)
    - Mini sparkline charts
    - Comparison periods
    """
    content = textwrap.dedent(
        """
        import React from 'react';
        import { LineChart, Line, ResponsiveContainer } from 'recharts';

        interface SparklineConfig {
          data: string;
          color?: string;
          height?: string;
        }

        interface StatSummaryWidgetConfig {
          id: string;
          type: 'stat_summary';
          title?: string;
          label: string;
          value: {
            field?: string;
            text?: string;
            format?: string;
          };
          delta?: {
            field: string;
            format?: string;
            showSign?: boolean;
          };
          trend?: {
            field: string;
            upIsGood?: boolean;
          };
          sparkline?: SparklineConfig;
          comparison?: string;
        }

        interface StatSummaryWidgetProps {
          widget: StatSummaryWidgetConfig;
          data: unknown;
        }

        function formatValue(value: any, format?: string): string {
          if (value === null || value === undefined) return '—';
          
          if (format === 'currency') {
            return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(Number(value));
          }
          if (format === 'percentage') {
            return `${Number(value).toFixed(1)}%`;
          }
          if (format === 'number') {
            return new Intl.NumberFormat('en-US').format(Number(value));
          }
          if (format === 'compact') {
            const num = Number(value);
            if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
            if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
            return String(num);
          }
          
          return String(value);
        }

        function getTrendIcon(trend: 'up' | 'down' | 'neutral'): string {
          if (trend === 'up') return '↑';
          if (trend === 'down') return '↓';
          return '→';
        }

        function getTrendColor(trend: 'up' | 'down' | 'neutral', upIsGood: boolean): string {
          if (trend === 'neutral') return '#6b7280';
          if (trend === 'up') return upIsGood ? '#10b981' : '#ef4444';
          return upIsGood ? '#ef4444' : '#10b981';
        }

        export function StatSummaryWidget({ widget, data }: StatSummaryWidgetProps) {
          const item = data && typeof data === 'object' ? data : {};

          const value = widget.value.field
            ? (item as any)[widget.value.field]
            : widget.value.text || '—';

          const delta = widget.delta
            ? (item as any)[widget.delta.field]
            : null;

          const trendValue = widget.trend
            ? (item as any)[widget.trend.field]
            : null;

          let trend: 'up' | 'down' | 'neutral' = 'neutral';
          if (trendValue !== null && trendValue !== undefined) {
            if (Number(trendValue) > 0) trend = 'up';
            else if (Number(trendValue) < 0) trend = 'down';
          }

          const sparklineData = widget.sparkline
            ? ((item as any)[widget.sparkline.data] as any[] | undefined)
            : undefined;

          return (
            <div
              className="n3-stat-summary"
              style={{
                padding: '1.5rem',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                backgroundColor: '#fff',
                minWidth: '200px',
              }}
            >
              {widget.title && (
                <h3 style={{ margin: '0 0 1rem', fontSize: '1.25rem', fontWeight: 600, color: '#111827' }}>
                  {widget.title}
                </h3>
              )}
              
              <div className="stat-label" style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '0.5rem' }}>
                {widget.label}
              </div>
              
              <div className="stat-value" style={{ fontSize: '2rem', fontWeight: 700, color: '#111827', marginBottom: '0.5rem' }}>
                {formatValue(value, widget.value.format)}
              </div>
              
              <div className="stat-details" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                {widget.delta && delta !== null && delta !== undefined && (
                  <div
                    className="stat-delta"
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.25rem',
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      color: getTrendColor(trend, widget.trend?.upIsGood !== false),
                    }}
                  >
                    {getTrendIcon(trend)}
                    {widget.delta.showSign !== false && delta > 0 ? '+' : ''}
                    {formatValue(delta, widget.delta.format)}
                  </div>
                )}
                
                {widget.comparison && (
                  <div className="stat-comparison" style={{ fontSize: '0.75rem', color: '#9ca3af' }}>
                    {widget.comparison}
                  </div>
                )}
              </div>
              
              {sparklineData && Array.isArray(sparklineData) && sparklineData.length > 0 && (
                <div className="stat-sparkline" style={{ marginTop: '1rem', height: widget.sparkline?.height || '40px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={sparklineData}>
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke={widget.sparkline?.color || '#3b82f6'}
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "StatSummaryWidget.tsx", content)


def write_timeline_widget(components_dir: Path) -> None:
    """
    Generate TimelineWidget.tsx for chronological event displays.
    
    Supports:
    - Timeline items with timestamps
    - Event icons and status
    - Title and description
    - Date grouping
    - Empty states
    """
    content = textwrap.dedent(
        """
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
          return template.replace(/\\{\\{\\s*(\\w+)\\s*\\}\\}/g, (_, key) => {
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
        """
    ).strip() + "\n"
    write_file(components_dir / "TimelineWidget.tsx", content)


def write_avatar_group_widget(components_dir: Path) -> None:
    """
    Generate AvatarGroupWidget.tsx for user/entity displays.
    
    Supports:
    - Avatar images or initials
    - Status indicators (online/offline/busy)
    - Tooltips with names
    - Overflow (+N more)
    - Size variants
    """
    content = textwrap.dedent(
        """
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
          const parts = name.trim().split(/\\s+/);
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
          return template.replace(/\\{\\{\\s*(\\w+)\\s*\\}\\}/g, (_, key) => {
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
        """
    ).strip() + "\n"
    write_file(components_dir / "AvatarGroupWidget.tsx", content)


def write_data_chart_widget(components_dir: Path) -> None:
    """
    Generate DataChartWidget.tsx for data visualizations.
    
    Supports:
    - Multiple chart types (line, bar, area, pie, scatter)
    - Multi-series data
    - Legend and tooltips
    - Axes configuration
    - Responsive sizing
    """
    content = textwrap.dedent(
        """
        import React from 'react';
        import {
          ResponsiveContainer,
          LineChart,
          BarChart,
          AreaChart,
          PieChart,
          ScatterChart,
          Line,
          Bar,
          Area,
          Pie,
          Scatter,
          XAxis,
          YAxis,
          CartesianGrid,
          Tooltip,
          Legend,
          Cell,
        } from 'recharts';

        interface ChartSeries {
          dataKey: string;
          label: string;
          color?: string;
          type?: 'line' | 'bar' | 'area';
        }

        interface ChartConfig {
          type: 'line' | 'bar' | 'pie' | 'area' | 'scatter';
          xAxis?: string;
          yAxis?: string;
          series: ChartSeries[];
          legend?: boolean;
          grid?: boolean;
          height?: string;
        }

        interface EmptyStateConfig {
          icon?: string;
          title: string;
          message?: string;
          actionLabel?: string;
          actionLink?: string;
        }

        interface DataChartWidgetConfig {
          id: string;
          type: 'data_chart';
          title?: string;
          source: {
            kind: string;
            name: string;
          };
          chartConfig: ChartConfig;
          emptyState?: EmptyStateConfig;
        }

        interface DataChartWidgetProps {
          widget: DataChartWidgetConfig;
          data: unknown;
        }

        const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

        function EmptyState({ config }: { config: EmptyStateConfig }) {
          return (
            <div className="n3-empty-state" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
              {config.icon && (
                <div className="empty-state-icon" style={{ fontSize: '2rem', marginBottom: '1rem', opacity: 0.5 }}>
                  📈
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

        export function DataChartWidget({ widget, data }: DataChartWidgetProps) {
          const chartData = Array.isArray(data) ? data : [];
          const { chartConfig } = widget;
          const height = parseInt(chartConfig.height || '300');

          if (chartData.length === 0 && widget.emptyState) {
            return <EmptyState config={widget.emptyState} />;
          }

          const renderChart = () => {
            const commonProps = {
              data: chartData,
              margin: { top: 5, right: 30, left: 20, bottom: 5 },
            };

            switch (chartConfig.type) {
              case 'line':
                return (
                  <LineChart {...commonProps}>
                    {chartConfig.grid && <CartesianGrid strokeDasharray="3 3" />}
                    {chartConfig.xAxis && <XAxis dataKey={chartConfig.xAxis} />}
                    {chartConfig.yAxis && <YAxis />}
                    <Tooltip />
                    {chartConfig.legend !== false && <Legend />}
                    {chartConfig.series.map((series, idx) => (
                      <Line
                        key={series.dataKey}
                        type="monotone"
                        dataKey={series.dataKey}
                        name={series.label}
                        stroke={series.color || COLORS[idx % COLORS.length]}
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        activeDot={{ r: 5 }}
                      />
                    ))}
                  </LineChart>
                );

              case 'bar':
                return (
                  <BarChart {...commonProps}>
                    {chartConfig.grid && <CartesianGrid strokeDasharray="3 3" />}
                    {chartConfig.xAxis && <XAxis dataKey={chartConfig.xAxis} />}
                    {chartConfig.yAxis && <YAxis />}
                    <Tooltip />
                    {chartConfig.legend !== false && <Legend />}
                    {chartConfig.series.map((series, idx) => (
                      <Bar
                        key={series.dataKey}
                        dataKey={series.dataKey}
                        name={series.label}
                        fill={series.color || COLORS[idx % COLORS.length]}
                      />
                    ))}
                  </BarChart>
                );

              case 'area':
                return (
                  <AreaChart {...commonProps}>
                    {chartConfig.grid && <CartesianGrid strokeDasharray="3 3" />}
                    {chartConfig.xAxis && <XAxis dataKey={chartConfig.xAxis} />}
                    {chartConfig.yAxis && <YAxis />}
                    <Tooltip />
                    {chartConfig.legend !== false && <Legend />}
                    {chartConfig.series.map((series, idx) => (
                      <Area
                        key={series.dataKey}
                        type="monotone"
                        dataKey={series.dataKey}
                        name={series.label}
                        stroke={series.color || COLORS[idx % COLORS.length]}
                        fill={series.color || COLORS[idx % COLORS.length]}
                        fillOpacity={0.6}
                      />
                    ))}
                  </AreaChart>
                );

              case 'pie':
                return (
                  <PieChart>
                    <Pie
                      data={chartData}
                      dataKey={chartConfig.series[0]?.dataKey || 'value'}
                      nameKey={chartConfig.xAxis || 'name'}
                      cx="50%"
                      cy="50%"
                      outerRadius={Math.min(height * 0.4, 120)}
                      label
                    >
                      {chartData.map((_, idx) => (
                        <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    {chartConfig.legend !== false && <Legend />}
                  </PieChart>
                );

              case 'scatter':
                return (
                  <ScatterChart {...commonProps}>
                    {chartConfig.grid && <CartesianGrid strokeDasharray="3 3" />}
                    {chartConfig.xAxis && <XAxis dataKey={chartConfig.xAxis} name={chartConfig.xAxis} />}
                    {chartConfig.yAxis && <YAxis dataKey={chartConfig.series[0]?.dataKey} name={chartConfig.series[0]?.label} />}
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    {chartConfig.legend !== false && <Legend />}
                    {chartConfig.series.map((series, idx) => (
                      <Scatter
                        key={series.dataKey}
                        name={series.label}
                        dataKey={series.dataKey}
                        fill={series.color || COLORS[idx % COLORS.length]}
                      />
                    ))}
                  </ScatterChart>
                );

              default:
                return <div>Unsupported chart type: {chartConfig.type}</div>;
            }
          };

          return (
            <div className="n3-data-chart" style={{ width: '100%' }}>
              {widget.title && (
                <h2 style={{ margin: '0 0 1rem', fontSize: '1.5rem', fontWeight: 600 }}>{widget.title}</h2>
              )}
              
              <ResponsiveContainer width="100%" height={height}>
                {renderChart()}
              </ResponsiveContainer>
            </div>
          );
        }
        """
    ).strip() + "\n"
    write_file(components_dir / "DataChartWidget.tsx", content)


def write_all_declarative_components(components_dir: Path) -> None:
    """Generate all declarative UI React components."""
    write_card_widget(components_dir)
    write_list_widget(components_dir)
    write_data_table_widget(components_dir)
    write_data_list_widget(components_dir)
    write_stat_summary_widget(components_dir)
    write_timeline_widget(components_dir)
    write_avatar_group_widget(components_dir)
    write_data_chart_widget(components_dir)
