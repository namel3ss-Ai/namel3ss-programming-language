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
