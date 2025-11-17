import type { TableWidgetConfig } from "../lib/n3Client";

interface TableWidgetProps {
  widget: TableWidgetConfig;
  data: unknown;
}

export default function TableWidget({ widget, data }: TableWidgetProps) {
  const rows = Array.isArray((data as any)?.rows) ? (data as any).rows as Record<string, unknown>[] : [];
  const columns = widget.columns && widget.columns.length ? widget.columns : rows.length ? Object.keys(rows[0]) : [];
  const pending = Boolean((data as any)?.pending);

  return (
    <section className="n3-widget">
      <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <h3 style={{ margin: 0 }}>{widget.title}</h3>
        {pending ? (
          <span
            data-n3-pending="true"
            style={{ fontSize: "0.85rem", color: "var(--text-muted, #4b5563)" }}
            aria-live="polite"
          >
            Updating...
          </span>
        ) : null}
      </div>
      {rows.length ? (
        <div style={{ overflowX: "auto" }}>
          <table className="n3-table">
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
