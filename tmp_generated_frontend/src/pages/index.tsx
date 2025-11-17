import Layout from "../components/Layout";
import ChartWidget from "../components/ChartWidget";
import TableWidget from "../components/TableWidget";
import FormWidget from "../components/FormWidget";
import TextBlock from "../components/TextBlock";
import ErrorBanner from "../components/ErrorBanner";
import { PageLoader } from "../components/LoadingState";
import { NAV_LINKS } from "../lib/navigation";
import { PageDefinition, resolveWidgetData, usePageData } from "../lib/n3Client";
import { useRealtimePage } from "../lib/realtime";

const PAGE_DEFINITION: PageDefinition = {
  "slug": "home",
  "route": "/",
  "title": "Home",
  "description": null,
  "reactive": false,
  "realtime": false,
  "widgets": [
    {
      "id": "text_1",
      "type": "text",
      "text": "Welcome to CoffeeHub!",
      "styles": {
        "color": "var(--primary)",
        "size": "large",
        "align": "center",
        "weight": "bold"
      }
    },
    {
      "id": "chart_1",
      "type": "chart",
      "title": "Revenue Growth",
      "chartType": "line",
      "source": {
        "kind": "dataset",
        "name": "monthly_sales"
      },
      "x": "month",
      "y": "total_revenue"
    },
    {
      "id": "table_1",
      "type": "table",
      "title": "Recent Orders",
      "source": {
        "kind": "table",
        "name": "orders"
      },
      "columns": [
        "id",
        "customer_name",
        "total",
        "status"
      ]
    }
  ],
  "preview": {
    "chart_1": {
      "labels": [
        "Apr",
        "Apr",
        "Feb",
        "Apr",
        "Apr",
        "Apr"
      ],
      "datasets": [
        {
          "label": "Total Revenue",
          "data": [
            235.31,
            8294.01,
            1585.65,
            5004.04,
            2433.79,
            1236.86
          ]
        }
      ]
    },
    "table_1": {
      "columns": [
        "id",
        "customer_name",
        "total",
        "status"
      ],
      "rows": [
        {
          "id": 7178,
          "customer_name": "Customer Name 1",
          "total": 1911.01,
          "status": "2024-01-01T20:36:00"
        },
        {
          "id": 7590,
          "customer_name": "Customer Name 2",
          "total": 4554.17,
          "status": "2024-01-01T13:56:00"
        },
        {
          "id": 738,
          "customer_name": "Customer Name 3",
          "total": 7719.28,
          "status": "2024-01-01T16:27:00"
        },
        {
          "id": 8676,
          "customer_name": "Customer Name 4",
          "total": 7305.86,
          "status": "2024-01-01T10:54:00"
        },
        {
          "id": 5038,
          "customer_name": "Customer Name 5",
          "total": 4453.6,
          "status": "2024-01-01T20:53:00"
        },
        {
          "id": 8472,
          "customer_name": "Customer Name 6",
          "total": 8408.03,
          "status": "2024-01-01T16:42:00"
        }
      ]
    }
  },
  "requiresAuth": false,
  "allowedRoles": [],
  "public": false,
  "redirectTo": null,
  "showInNav": true,
  "nav": {}
} as const;
const OPTIMISTIC_WIDGET_IDS = PAGE_DEFINITION.widgets
  .filter((widget) => widget.type === "table" || widget.type === "chart")
  .map((widget) => widget.id);

export default function IndexPage() {
  const {
    data,
    loading,
    error,
    pageErrors,
    fieldErrors,
    reload,
    applyRealtime,
    applyOptimistic,
    rollbackOptimistic,
    clearOptimistic,
  } = usePageData(PAGE_DEFINITION);
  const { connected: realtimeConnected, lastError: realtimeError } = useRealtimePage(PAGE_DEFINITION, {
    onEvent: (event) => {
      if (!event) {
        return;
      }
      if (event.payload && typeof event.payload === "object") {
        const replace = event.type === "snapshot" || event.type === "hydration";
        applyRealtime(event.payload, { replace });
        return;
      }
      if (event.type === "snapshot" || event.type === "hydration") {
        reload({ silent: true });
      }
      if (event.meta && typeof event.meta === "object") {
        const refresh = (event.meta as Record<string, unknown>).refresh;
        if (refresh === true || refresh === "page") {
          reload({ silent: true });
        }
      }
    },
    fallbackIntervalSeconds: 15,
  });

  return (
    <Layout title={PAGE_DEFINITION.title} description={PAGE_DEFINITION.description} navLinks={NAV_LINKS}>
      {PAGE_DEFINITION.realtime ? (
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem", marginBottom: "0.75rem", color: "var(--text-muted, #475569)" }}>
          <span
            aria-hidden="true"
            style={{
              width: "0.55rem",
              height: "0.55rem",
              borderRadius: "9999px",
              backgroundColor: realtimeConnected ? "var(--success, #16a34a)" : "var(--warning, #dc2626)",
              boxShadow: realtimeConnected ? "0 0 0 2px rgba(22, 163, 74, 0.25)" : "0 0 0 2px rgba(220, 38, 38, 0.25)",
            }}
          />
          <span>{realtimeConnected ? "Live updates active" : "Waiting for live updates"}</span>
          {realtimeError ? <span style={{ color: "var(--warning, #dc2626)" }}>• {realtimeError}</span> : null}
        </div>
      ) : null}
      {pageErrors.length ? <ErrorBanner errors={pageErrors} tone="warning" /> : null}
      {error ? <ErrorBanner errors={[error]} tone="error" /> : null}
      {loading ? (
        <PageLoader />
      ) : !error ? (
        <div style={{ display: "grid", gap: "1.25rem" }}>
          {PAGE_DEFINITION.widgets.map((widget) => {
            const widgetData = resolveWidgetData(widget.id, data) ?? PAGE_DEFINITION.preview[widget.id];
            if (widget.type === "text") {
              return <TextBlock key={widget.id} widget={widget} />;
            }
            if (widget.type === "chart") {
              return <ChartWidget key={widget.id} widget={widget} data={widgetData} />;
            }
            if (widget.type === "table") {
              return <TableWidget key={widget.id} widget={widget} data={widgetData} />;
            }
            if (widget.type === "form") {
              return (
                <FormWidget
                  key={widget.id}
                  widget={widget}
                  pageSlug={PAGE_DEFINITION.slug}
                  fieldErrors={fieldErrors}
                  optimisticTargets={OPTIMISTIC_WIDGET_IDS}
                  onReload={reload}
                  applyOptimistic={applyOptimistic}
                  rollbackOptimistic={rollbackOptimistic}
                  clearOptimistic={clearOptimistic}
                />
              );
            }
            return null;
          })}
        </div>
      ) : null}
    </Layout>
  );
}
