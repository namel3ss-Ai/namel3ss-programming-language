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
  "slug": "feedback",
  "route": "/feedback",
  "title": "Feedback",
  "description": null,
  "reactive": false,
  "realtime": false,
  "widgets": [
    {
      "id": "form_1",
      "type": "form",
      "title": "Submit Feedback",
      "fields": [
        {
          "name": "name",
          "type": "text"
        },
        {
          "name": "email",
          "type": "email"
        },
        {
          "name": "message",
          "type": "text"
        }
      ],
      "successMessage": "Thank you for your feedback!"
    }
  ],
  "preview": {
    "form_1": {
      "fields": [
        {
          "name": "name",
          "type": "text"
        },
        {
          "name": "email",
          "type": "email"
        },
        {
          "name": "message",
          "type": "text"
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

export default function FeedbackPage() {
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
