import Layout from "../components/Layout";
import ChartWidget from "../components/ChartWidget";
import TableWidget from "../components/TableWidget";
import FormWidget from "../components/FormWidget";
import TextBlock from "../components/TextBlock";
import CardWidget from "../components/CardWidget";
import ListWidget from "../components/ListWidget";
import { StackLayout, GridLayout, SplitLayout, TabsLayout, AccordionLayout } from "../components/LayoutComponents";
import { DataTableWidget } from "../components/DataTableWidget";
import { DataListWidget } from "../components/DataListWidget";
import { StatSummaryWidget } from "../components/StatSummaryWidget";
import { TimelineWidget } from "../components/TimelineWidget";
import { AvatarGroupWidget } from "../components/AvatarGroupWidget";
import { DataChartWidget } from "../components/DataChartWidget";
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";
import Breadcrumbs from "../components/Breadcrumbs";
import CommandPalette from "../components/CommandPalette";
import Modal from "../components/Modal";
import Toast from "../components/Toast";
import { NAV_LINKS } from "../lib/navigation";
import { PageDefinition, resolveWidgetData, usePageData } from "../lib/n3Client";
import { useRealtimePage } from "../lib/realtime";

const PAGE_DEFINITION: PageDefinition = {
  "slug": "user_settings",
  "route": "/settings",
  "title": "User Settings",
  "description": null,
  "reactive": false,
  "realtime": false,
  "widgets": [
    {
      "id": "navbar_1",
      "type": "navbar",
      "logo": null,
      "title": "User Settings",
      "actions": [],
      "position": "top",
      "sticky": true
    },
    {
      "id": "modal_1",
      "type": "modal",
      "modal_id": "confirm_reset",
      "title": "Reset Settings",
      "description": "This will restore all settings to their default values",
      "size": "md",
      "dismissible": true,
      "trigger": "show_confirm_reset",
      "actions": [
        {
          "label": "Cancel",
          "variant": "ghost",
          "close": true
        },
        {
          "label": "Reset All",
          "action": "reset_settings",
          "variant": "destructive",
          "close": true
        }
      ],
      "content": [
        {
          "type": "text",
          "text": "\u26a0\ufe0f Warning: This action will reset:",
          "styles": {}
        },
        {
          "type": "text",
          "text": "\u2022 Theme preferences",
          "styles": {}
        },
        {
          "type": "text",
          "text": "\u2022 Notification settings",
          "styles": {}
        },
        {
          "type": "text",
          "text": "\u2022 Privacy options",
          "styles": {}
        },
        {
          "type": "text",
          "text": "\u2022 Display preferences",
          "styles": {}
        }
      ]
    },
    {
      "id": "toast_1",
      "type": "toast",
      "toast_id": "reset_complete",
      "title": "Settings Reset",
      "description": "All settings have been restored to defaults",
      "variant": "success",
      "duration": 3000,
      "action_label": null,
      "action": null,
      "position": "top-right",
      "trigger": "show_reset_complete"
    },
    {
      "id": "text_1",
      "type": "text",
      "text": "User Settings Page",
      "styles": {}
    },
    {
      "id": "text_2",
      "type": "text",
      "text": "Configure your account preferences and privacy settings.",
      "styles": {}
    }
  ],
  "preview": {}
} as const;

function renderWidget(widget: any, data: any): React.ReactNode {
  const widgetData = resolveWidgetData(widget.id, data) ?? PAGE_DEFINITION.preview[widget.id];

  switch (widget.type) {
    case "text":
      return <TextBlock key={widget.id} widget={widget} />;
    case "chart":
      return <ChartWidget key={widget.id} widget={widget} data={widgetData} />;
    case "table":
      return <TableWidget key={widget.id} widget={widget} data={widgetData} />;
    case "form":
      return <FormWidget key={widget.id} widget={widget} pageSlug={PAGE_DEFINITION.slug} />;
    case "card":
      return <CardWidget key={widget.id} widget={widget} data={widgetData} />;
    case "list":
      return <ListWidget key={widget.id} widget={widget} data={widgetData} />;

    // Data display components
    case "data_table":
      return <DataTableWidget key={widget.id} widget={widget} data={widgetData} />;
    case "data_list":
      return <DataListWidget key={widget.id} widget={widget} data={widgetData} />;
    case "stat_summary":
      return <StatSummaryWidget key={widget.id} widget={widget} data={widgetData} />;
    case "timeline":
      return <TimelineWidget key={widget.id} widget={widget} data={widgetData} />;
    case "avatar_group":
      return <AvatarGroupWidget key={widget.id} widget={widget} data={widgetData} />;
    case "data_chart":
      return <DataChartWidget key={widget.id} widget={widget} data={widgetData} />;

    // Navigation & Chrome components
    case "sidebar":
      return <Sidebar key={widget.id} {...widget} />;
    case "navbar":
      return <Navbar key={widget.id} {...widget} />;
    case "breadcrumbs":
      return <Breadcrumbs key={widget.id} {...widget} />;
    case "command_palette":
      return <CommandPalette key={widget.id} {...widget} />;

    // Feedback components
    case "modal":
      return (
        <Modal
          key={widget.id}
          id={widget.modal_id}
          title={widget.title}
          description={widget.description}
          content={widget.content?.map((child: any) => renderWidget(child, data)) || []}
          actions={widget.actions}
          size={widget.size}
          dismissible={widget.dismissible}
          trigger={widget.trigger}
        />
      );
    case "toast":
      return (
        <Toast
          key={widget.id}
          id={widget.toast_id}
          title={widget.title}
          description={widget.description}
          variant={widget.variant}
          duration={widget.duration}
          action_label={widget.action_label}
          action={widget.action}
          position={widget.position}
          trigger={widget.trigger}
        />
      );

    case "stack":
      return (
        <StackLayout
          key={widget.id}
          direction={widget.direction}
          gap={widget.gap}
          align={widget.align}
          justify={widget.justify}
          wrap={widget.wrap}
          style={widget.style}
        >
          {widget.children?.map((child: any) => renderWidget(child, data)) || []}
        </StackLayout>
      );

    case "grid":
      return (
        <GridLayout
          key={widget.id}
          columns={widget.columns}
          minColumnWidth={widget.minColumnWidth}
          gap={widget.gap}
          responsive={widget.responsive}
          style={widget.style}
        >
          {widget.children?.map((child: any) => renderWidget(child, data)) || []}
        </GridLayout>
      );

    case "split":
      return (
        <SplitLayout
          key={widget.id}
          left={widget.left?.map((child: any) => renderWidget(child, data)) || []}
          right={widget.right?.map((child: any) => renderWidget(child, data)) || []}
          ratio={widget.ratio}
          resizable={widget.resizable}
          orientation={widget.orientation}
          style={widget.style}
        />
      );

    case "tabs":
      return (
        <TabsLayout
          key={widget.id}
          tabs={widget.tabs?.map((tab: any) => ({
            id: tab.id,
            label: tab.label,
            icon: tab.icon,
            badge: tab.badge,
            content: tab.content?.map((child: any) => renderWidget(child, data)) || [],
          })) || []}
          defaultTab={widget.defaultTab}
          persistState={widget.persistState}
          style={widget.style}
        />
      );

    case "accordion":
      return (
        <AccordionLayout
          key={widget.id}
          items={widget.items?.map((item: any) => ({
            id: item.id,
            title: item.title,
            description: item.description,
            icon: item.icon,
            defaultOpen: item.defaultOpen,
            content: item.content?.map((child: any) => renderWidget(child, data)) || [],
          })) || []}
          multiple={widget.multiple}
          style={widget.style}
        />
      );

    default:
      return null;
  }
}

export default function SettingsPage() {
  const { data, loading, error } = usePageData(PAGE_DEFINITION);
  useRealtimePage(PAGE_DEFINITION);

  return (
    <Layout title={PAGE_DEFINITION.title} description={PAGE_DEFINITION.description} navLinks={NAV_LINKS}>
      {loading ? (
        <p>Loading page data...</p>
      ) : error ? (
        <p role="alert">Failed to load page: {error}</p>
      ) : (
        <div style={{ display: "grid", gap: "1.25rem" }}>
          {PAGE_DEFINITION.widgets.map((widget) => renderWidget(widget, data))}
        </div>
      )}
    </Layout>
  );
}
