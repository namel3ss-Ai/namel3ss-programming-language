import type { TableWidgetConfig } from "../lib/n3Client";

interface TableWidgetProps {
  widget: TableWidgetConfig;
  data: unknown;
}

export default function TableWidget({ widget, data }: TableWidgetProps) {
  const rows = Array.isArray((data as any)?.rows) ? (data as any).rows as Record<string, unknown>[] : [];
  const columns = widget.columns && widget.columns.length ? widget.columns : rows.length ? Object.keys(rows[0]) : [];

  return (
    <section className="n3-widget">
      <h3>{widget.title}</h3>
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
