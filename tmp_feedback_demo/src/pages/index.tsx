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
  "slug": "feedback_demo",
  "route": "/",
  "title": "Feedback Demo",
  "description": null,
  "reactive": false,
  "realtime": false,
  "widgets": [
    {
      "id": "navbar_1",
      "type": "navbar",
      "logo": null,
      "title": "Feedback Components Demo",
      "actions": [
        {
          "id": "theme",
          "type": "toggle",
          "label": "Theme",
          "icon": "\ud83c\udfa8"
        }
      ],
      "position": "top",
      "sticky": true
    },
    {
      "id": "text_1",
      "type": "text",
      "text": "Feedback Components Demonstration",
      "styles": {}
    },
    {
      "id": "text_2",
      "type": "text",
      "text": "This demo showcases Modal dialogs and Toast notifications with various configurations.",
      "styles": {}
    },
    {
      "id": "modal_1",
      "type": "modal",
      "modal_id": "confirm_delete",
      "title": "Confirm Delete",
      "description": "Are you sure you want to delete this item?",
      "size": "md",
      "dismissible": true,
      "trigger": "show_confirm_delete",
      "actions": [
        {
          "label": "Cancel",
          "variant": "ghost",
          "close": true
        },
        {
          "label": "Delete",
          "action": "do_delete",
          "variant": "destructive",
          "close": true
        }
      ],
      "content": [
        {
          "type": "text",
          "text": "This action cannot be undone. The item will be permanently removed from the database.",
          "styles": {}
        }
      ]
    },
    {
      "id": "modal_2",
      "type": "modal",
      "modal_id": "edit_profile",
      "title": "Edit Profile",
      "description": "Update your profile information",
      "size": "lg",
      "dismissible": true,
      "trigger": "show_edit_profile",
      "actions": [
        {
          "label": "Cancel",
          "variant": "ghost",
          "close": true
        },
        {
          "label": "Save Draft",
          "action": "save_draft",
          "variant": "default",
          "close": false
        },
        {
          "label": "Save & Close",
          "action": "save_profile",
          "variant": "primary",
          "close": true
        }
      ],
      "content": [
        {
          "type": "text",
          "text": "Name: John Doe",
          "styles": {}
        },
        {
          "type": "text",
          "text": "Email: john@example.com",
          "styles": {}
        },
        {
          "type": "text",
          "text": "Role: Administrator",
          "styles": {}
        }
      ]
    },
    {
      "id": "modal_3",
      "type": "modal",
      "modal_id": "quick_tip",
      "title": "Quick Tip",
      "description": null,
      "size": "sm",
      "dismissible": true,
      "trigger": "show_quick_tip",
      "actions": [
        {
          "label": "Got it",
          "variant": "primary",
          "close": true
        }
      ],
      "content": [
        {
          "type": "text",
          "text": "\ud83d\udca1 Press Ctrl+K to open the command palette from anywhere!",
          "styles": {}
        }
      ]
    },
    {
      "id": "modal_4",
      "type": "modal",
      "modal_id": "terms_conditions",
      "title": "Terms and Conditions",
      "description": "Please read our terms carefully",
      "size": "xl",
      "dismissible": false,
      "trigger": "show_terms",
      "actions": [
        {
          "label": "Decline",
          "action": "decline_terms",
          "variant": "ghost",
          "close": true
        },
        {
          "label": "Accept",
          "action": "accept_terms",
          "variant": "primary",
          "close": true
        }
      ],
      "content": [
        {
          "type": "text",
          "text": "1. Acceptance of Terms",
          "styles": {}
        },
        {
          "type": "text",
          "text": "By using this service, you agree to be bound by these terms and conditions.",
          "styles": {}
        },
        {
          "type": "text",
          "text": "2. User Responsibilities",
          "styles": {}
        },
        {
          "type": "text",
          "text": "You are responsible for maintaining the confidentiality of your account.",
          "styles": {}
        },
        {
          "type": "text",
          "text": "3. Privacy Policy",
          "styles": {}
        },
        {
          "type": "text",
          "text": "We respect your privacy and protect your personal information.",
          "styles": {}
        }
      ]
    },
    {
      "id": "toast_1",
      "type": "toast",
      "toast_id": "save_success",
      "title": "Changes Saved",
      "description": "Your profile has been updated successfully",
      "variant": "success",
      "duration": 3000,
      "action_label": "View",
      "action": "view_profile",
      "position": "top-right",
      "trigger": "show_save_success"
    },
    {
      "id": "toast_2",
      "type": "toast",
      "toast_id": "connection_error",
      "title": "Connection Failed",
      "description": "Unable to connect to the server. Please check your internet connection.",
      "variant": "error",
      "duration": 0,
      "action_label": "Retry",
      "action": "retry_connection",
      "position": "top-right",
      "trigger": "show_connection_error"
    },
    {
      "id": "toast_3",
      "type": "toast",
      "toast_id": "storage_warning",
      "title": "Storage Almost Full",
      "description": "You have used 90% of your storage quota",
      "variant": "warning",
      "duration": 5000,
      "action_label": "Manage Storage",
      "action": "manage_storage",
      "position": "bottom-right",
      "trigger": "show_storage_warning"
    },
    {
      "id": "toast_4",
      "type": "toast",
      "toast_id": "update_available",
      "title": "Update Available",
      "description": "A new version of the app is available",
      "variant": "info",
      "duration": 4000,
      "action_label": "Update Now",
      "action": "start_update",
      "position": "top",
      "trigger": "show_update_available"
    },
    {
      "id": "toast_5",
      "type": "toast",
      "toast_id": "clipboard_copy",
      "title": "Copied to Clipboard",
      "description": null,
      "variant": "default",
      "duration": 2000,
      "action_label": null,
      "action": null,
      "position": "bottom-left",
      "trigger": "show_clipboard_copy"
    },
    {
      "id": "toast_6",
      "type": "toast",
      "toast_id": "item_deleted",
      "title": "Item Deleted",
      "description": "The item has been removed",
      "variant": "success",
      "duration": 3000,
      "action_label": null,
      "action": null,
      "position": "bottom",
      "trigger": "show_item_deleted"
    },
    {
      "id": "text_3",
      "type": "text",
      "text": "Feedback components provide essential user interaction patterns for modern web applications.",
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

export default function IndexPage() {
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
