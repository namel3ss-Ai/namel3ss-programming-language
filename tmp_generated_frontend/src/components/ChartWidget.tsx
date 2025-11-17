import type { ChartWidgetConfig } from "../lib/n3Client";
import { ensureArray } from "../lib/n3Client";

interface ChartWidgetProps {
  widget: ChartWidgetConfig;
  data: unknown;
}

export default function ChartWidget({ widget, data }: ChartWidgetProps) {
  const labels = Array.isArray((data as any)?.labels) ? (data as any).labels as string[] : [];
  const datasets = ensureArray<{ label?: string; data?: unknown[] }>((data as any)?.datasets);
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
      {labels.length && datasets.length ? (
        <div>
          {datasets.map((dataset, index) => (
            <div key={dataset.label ?? index} style={{ marginBottom: "0.75rem" }}>
              <strong>{dataset.label ?? `Series ${index + 1}`}</strong>
              <ul style={{ listStyle: "none", paddingLeft: 0 }}>
                {labels.map((label, idx) => (
                  <li key={label + idx}>
                    <span style={{ fontWeight: 500 }}>{label}:</span> {Array.isArray(dataset.data) ? dataset.data[idx] : "n/a"}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      ) : (
        <pre style={{ marginTop: "0.75rem" }}>{JSON.stringify(data ?? widget, null, 2)}</pre>
      )}
    </section>
  );
}
