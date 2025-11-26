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
