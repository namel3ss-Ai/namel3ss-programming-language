"""Page and UI statement AST nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union, TYPE_CHECKING, Set

from .base import Expression, LogStatement
from .design_tokens import VariantType, ToneType, DensityType, SizeType

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .models import InferenceTarget


@dataclass
class Statement:
    """Base class for page statements."""

    pass


@dataclass
class ShowText(Statement):
    text: str
    styles: Dict[str, str] = field(default_factory=dict)


@dataclass
class LayoutSpec:
    width: Optional[int] = None
    height: Optional[int] = None
    variant: Optional[str] = None
    order: Optional[int] = None
    area: Optional[str] = None
    breakpoint: Optional[str] = None
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutMeta:
    """Layout metadata for pages and components."""
    # New fields matching backend encoder expectations
    direction: Optional[str] = None  # "row" | "column"
    spacing: Optional[str] = None  # "small" | "medium" | "large"
    # Legacy fields (kept for backward compatibility)
    width: Optional[int] = None
    height: Optional[int] = None
    variant: Optional[str] = None
    align: Optional[str] = None
    emphasis: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBindingConfig:
    """
    Configuration for dynamic data binding on UI components.
    
    Controls how components fetch, display, and update data from datasets.
    """
    # Read behavior
    auto_refresh: bool = False  # Auto-refresh when dataset changes
    refresh_interval: Optional[int] = None  # Polling interval in seconds (if not using realtime)
    page_size: int = 50  # Default items per page
    enable_sorting: bool = True  # Allow user to sort columns
    enable_filtering: bool = True  # Allow user to filter rows
    enable_search: bool = False  # Show search box
    cache_ttl: Optional[int] = None  # Client-side cache TTL in seconds
    
    # Write behavior
    editable: bool = False  # Allow inline editing
    enable_create: bool = False  # Show "Add" button
    enable_update: bool = False  # Allow row updates
    enable_delete: bool = False  # Show delete actions
    
    # Realtime behavior
    subscribe_to_changes: bool = False  # Subscribe to WebSocket updates (requires realtime extra)
    
    # Advanced
    field_mapping: Dict[str, str] = field(default_factory=dict)  # Map component fields to dataset columns
    write_endpoint: Optional[str] = None  # Custom write endpoint
    optimistic_updates: bool = True  # Update UI immediately before server confirms


@dataclass
class ShowTable(Statement):
    title: str
    source_type: str
    source: str
    columns: Optional[List[str]] = None
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    layout: Optional[LayoutMeta] = None
    insight: Optional[str] = None
    dynamic_columns: Optional[Dict[str, Any]] = None
    
    # Data binding configuration
    binding: Optional[DataBindingConfig] = None
    
    # Design tokens
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    density: Optional[DensityType] = None
    size: Optional[SizeType] = None
    theme: Optional[ThemeType] = None
    color_scheme: Optional[ColorSchemeType] = None


@dataclass
class ShowChart(Statement):
    heading: str
    source_type: str
    source: str
    chart_type: str = "bar"
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    layout: Optional[LayoutMeta] = None
    insight: Optional[str] = None
    encodings: Dict[str, Any] = field(default_factory=dict)
    style: Optional[Dict[str, Any]] = field(default_factory=dict)
    title: Optional[str] = None
    legend: Optional[Dict[str, Any]] = None
    
    # Data binding configuration
    binding: Optional[DataBindingConfig] = None


@dataclass
class FormField:
    """Production-grade form field definition.
    
    Supports semantic field components (text_input, select, etc.) with
    validation, bindings, and conditional rendering.
    """
    name: str
    component: str = "text_input"  # Field component type (text_input, select, textarea, etc.)
    label: Optional[str] = None
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    required: bool = False
    default: Optional[Expression] = None
    initial_value: Optional[Expression] = None
    
    # Validation rules
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex pattern for text validation
    min_value: Optional[float] = None  # For numeric/slider
    max_value: Optional[float] = None
    step: Optional[float] = None  # For numeric/slider/datetime
    
    # Options for select/multiselect/radio_group
    options_binding: Optional[str] = None  # Bind to dataset or options provider
    options: Optional[List[Dict[str, Any]]] = field(default_factory=list)  # Static options fallback
    
    # Conditional rendering
    disabled: Optional[Expression] = None
    visible: Optional[Expression] = None
    
    # Component-specific config
    multiple: bool = False  # For select/file_upload
    accept: Optional[str] = None  # MIME types for file_upload
    max_file_size: Optional[int] = None  # Bytes for file_upload
    upload_endpoint: Optional[str] = None  # Upload target for file_upload
    
    # Backward compatibility
    field_type: Optional[str] = None  # Deprecated: use 'component' instead
    
    # Design tokens
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    size: Optional[SizeType] = None


@dataclass
class ActionOperation:
    """Base class for action operations."""

    pass


@dataclass
class UpdateOperation(ActionOperation):
    table: str
    set_expression: str
    where_expression: Optional[str] = None


@dataclass
class ToastOperation(ActionOperation):
    message: str


@dataclass
class GoToPageOperation(ActionOperation):
    page_name: str


@dataclass
class CallPythonOperation(ActionOperation):
    module: str
    method: str
    arguments: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class AskConnectorOperation(ActionOperation):
    connector_name: str
    arguments: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class RunChainOperation(ActionOperation):
    chain_name: str
    inputs: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class RunPromptOperation(ActionOperation):
    prompt_name: str
    arguments: Dict[str, Expression] = field(default_factory=dict)


@dataclass
class ShowForm(Statement):
    """Production-grade form component.
    
    Supports declarative form definitions with data bindings, validation,
    and integration with the action system.
    """
    title: str
    fields: List[FormField] = field(default_factory=list)
    
    # Layout configuration
    layout_mode: str = "vertical"  # "vertical" | "horizontal" | "inline"
    
    # Action integration
    submit_action: Optional[str] = None  # Reference to action name or expression
    on_submit_ops: List['ActionOperationType'] = field(default_factory=list)  # Legacy operations
    
    # Data bindings
    initial_values_binding: Optional[str] = None  # Bind to dataset/query for initial values
    bound_dataset: Optional[str] = None  # Deprecated: use initial_values_binding
    bound_record_id: Optional[Expression] = None  # Specific record ID to load
    
    # Form-level configuration
    validation_mode: str = "on_blur"  # "on_blur" | "on_change" | "on_submit"
    submit_button_text: Optional[str] = None
    reset_button: bool = False
    loading_text: Optional[str] = None
    success_message: Optional[str] = None
    error_message: Optional[str] = None
    
    # Legacy/styling
    styles: Dict[str, str] = field(default_factory=dict)
    layout: Optional[LayoutSpec] = None  # Deprecated: use layout_mode
    effects: Set[str] = field(default_factory=set)
    binding: Optional[DataBindingConfig] = None
    
    # Design tokens
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    density: Optional[DensityType] = None
    size: Optional[SizeType] = None
    theme: Optional[ThemeType] = None
    color_scheme: Optional[ColorSchemeType] = None


@dataclass
class Action(Statement):
    name: str
    trigger: str
    operations: List['ActionOperationType'] = field(default_factory=list)
    declared_effect: Optional[str] = None
    effects: Set[str] = field(default_factory=set)


@dataclass
class VariableAssignment(Statement):
    name: str
    value: Expression


@dataclass
class IfBlock(Statement):
    condition: Expression
    body: List['PageStatement'] = field(default_factory=list)
    elifs: List['ElifBlock'] = field(default_factory=list)
    else_body: Optional[List['PageStatement']] = None


@dataclass
class ForLoop(Statement):
    loop_var: str
    source_kind: Literal["dataset", "table", "frame"]
    source_name: str
    body: List['PageStatement'] = field(default_factory=list)


@dataclass
class ElifBlock:
    condition: Expression
    body: List['PageStatement'] = field(default_factory=list)


@dataclass
class WhileLoop(Statement):
    condition: Expression
    body: List['PageStatement'] = field(default_factory=list)


@dataclass
class BreakStatement(Statement):
    pass


@dataclass
class ContinueStatement(Statement):
    pass


def _default_inference_target() -> 'InferenceTarget':
    from .models import InferenceTarget

    return InferenceTarget()


@dataclass
class PredictStatement(Statement):
    model_name: str
    input_kind: Literal["dataset", "table", "variables", "payload"] = "dataset"
    input_ref: Optional[str] = None
    assign: 'InferenceTarget' = field(default_factory=_default_inference_target)
    parameters: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Declarative UI Components (Card, List, Sections)
# =============================================================================

@dataclass
class EmptyStateConfig:
    """Configuration for empty state display when no data is available."""
    icon: Optional[str] = None
    icon_size: Optional[str] = None  # "small" | "medium" | "large"
    title: str = "No items"
    message: Optional[str] = None
    action_label: Optional[str] = None
    action_link: Optional[str] = None


@dataclass
class BadgeConfig:
    """Badge configuration for displaying metadata on cards."""
    field: Optional[str] = None  # Field name to display
    text: Optional[str] = None  # Static text (overrides field)
    style: Optional[str] = None  # CSS class name - DEPRECATED, use design tokens
    transform: Optional[Union[str, Dict[str, Any]]] = None  # Transform to apply (e.g., "humanize", {"format": "..."})
    icon: Optional[str] = None
    condition: Optional[str] = None  # Expression to evaluate for conditional display
    
    # Design tokens
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    size: Optional[SizeType] = None


@dataclass
class FieldValueConfig:
    """Configuration for a single field value display."""
    field: Optional[str] = None  # Field name from data
    text: Optional[str] = None  # Static text (supports templates like "Dr. {{ provider }}")
    format: Optional[str] = None  # Format string (e.g., "MMMM DD, YYYY")
    style: Optional[str] = None  # CSS class name
    transform: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class InfoGridItem:
    """Single item in an info grid section."""
    icon: Optional[str] = None
    label: Optional[str] = None
    field_name: Optional[str] = None  # Single field (renamed to avoid shadowing dataclasses.field)
    values: List[FieldValueConfig] = field(default_factory=list)  # Multiple values


@dataclass
class CardSection:
    """A section within a card (e.g., info_grid, text_section, key_points)."""
    type: str  # "info_grid" | "text_section" | "key_points" | "questions" | etc.
    condition: Optional[str] = None  # Expression for conditional rendering
    style: Optional[str] = None
    title: Optional[str] = None
    icon: Optional[str] = None
    
    # For info_grid sections
    columns: Optional[int] = None
    items: List[InfoGridItem] = field(default_factory=list)
    
    # For text_section
    content: Optional[Dict[str, Any]] = None  # {label, text, ...}
    
    # For list sections (key_points, questions)
    list_items: Optional[Dict[str, Any]] = None  # {field, style, ...}


@dataclass
class ConditionalAction:
    """Action button with optional conditional display."""
    label: str
    icon: Optional[str] = None
    style: Optional[str] = None  # "primary" | "secondary" | "danger" - DEPRECATED, use tone
    action: Optional[str] = None  # Action name to invoke
    link: Optional[str] = None  # Navigation link
    params: Optional[str] = None  # Parameters to pass (can include templates)
    condition: Optional[str] = None  # Expression for conditional display
    
    # Design tokens (for button styling)
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    size: Optional[SizeType] = None


@dataclass
class CardHeader:
    """Card header configuration with title and badges."""
    title: Optional[str] = None  # Field name or template
    subtitle: Optional[str] = None
    badges: List[BadgeConfig] = field(default_factory=list)
    avatar: Optional[Dict[str, Any]] = None  # Avatar configuration


@dataclass
class CardFooter:
    """Card footer configuration."""
    condition: Optional[str] = None
    text: Optional[str] = None  # Template string
    style: Optional[str] = None
    left: Optional[Dict[str, Any]] = None  # Left-aligned content
    right: Optional[Dict[str, Any]] = None  # Right-aligned content


@dataclass
class CardItemConfig:
    """Configuration for how individual items are rendered in a card list."""
    type: str  # "card" | "message_bubble" | "article_card" | etc.
    style: Optional[str] = None
    state_class: Optional[Dict[str, str]] = None  # Dynamic CSS classes based on state
    role_class: Optional[str] = None  # Template for role-based class
    
    header: Optional[Union[CardHeader, Dict[str, Any]]] = None
    sections: List[CardSection] = field(default_factory=list)
    actions: List[ConditionalAction] = field(default_factory=list)
    footer: Optional[Union[CardFooter, Dict[str, Any]]] = None
    
    # For message bubbles and semantic components
    avatar: Optional[Dict[str, Any]] = None
    content: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None  # Content structure
    body: Optional[Dict[str, Any]] = None
    attachments: Optional[Dict[str, Any]] = None
    badge: Optional[Union[BadgeConfig, Dict[str, Any]]] = None


@dataclass
class ShowCard(Statement):
    """
    Declarative card-based list component.
    
    Displays data from a data source as styled cards with sections,
    badges, conditional actions, and empty states.
    
    Example:
        show card "Appointments" from dataset appointments:
            variant: elevated
            tone: primary
            size: md
            empty_state:
                icon: calendar
                title: "No appointments"
            item:
                type: card
                sections:
                    - type: info_grid
                      columns: 2
                      items: [...]
                actions:
                    - label: "Edit"
                      condition: "status == 'pending'"
    """
    title: str
    source_type: str  # "dataset" | "table" | "frame" | "endpoint"
    source: str
    
    # Display configuration
    empty_state: Optional[EmptyStateConfig] = None
    item_config: Optional[CardItemConfig] = None
    
    # Grouping and filtering
    group_by: Optional[str] = None
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    
    # Layout and styling
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingConfig] = None
    
    # Design tokens
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    density: Optional[DensityType] = None
    size: Optional[SizeType] = None


@dataclass
class ShowList(Statement):
    """
    Generic list component for displaying collections.
    
    Similar to ShowCard but with more flexible rendering options.
    Supports custom item templates and semantic component types.
    """
    title: str
    source_type: str
    source: str
    
    list_type: str = "default"  # "default" | "conversation" | "message" | "article"
    empty_state: Optional[EmptyStateConfig] = None
    item_config: Optional[CardItemConfig] = None
    
    # Search and filtering
    enable_search: bool = False
    search_placeholder: Optional[str] = None
    filters: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pagination
    page_size: int = 50
    
    # Grouping and sorting
    group_by: Optional[str] = None
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    
    # Layout and styling
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None
    columns: Optional[int] = None
    
    # Data binding
    binding: Optional[DataBindingConfig] = None
    
    # Design tokens
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    density: Optional[DensityType] = None
    size: Optional[SizeType] = None
    page_size: int = 50
    enable_pagination: bool = True
    
    # Layout
    columns: Optional[int] = None  # For grid layout
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingConfig] = None


# =============================================================================
# Data Display Components (Professional Data Visualization)
# =============================================================================

@dataclass
class ColumnConfig:
    """Configuration for a table column."""
    id: str
    label: str
    field: Optional[str] = None  # Data field name (defaults to id if not specified)
    width: Optional[Union[str, int]] = None  # CSS width value or pixel number
    align: str = "left"  # "left" | "center" | "right"
    sortable: bool = True
    format: Optional[str] = None  # Date/number/currency format string
    transform: Optional[Union[str, Dict[str, Any]]] = None  # Value transformation
    render_template: Optional[str] = None  # Custom template string (e.g., "Dr. {{ name }}")


@dataclass
class ToolbarConfig:
    """Toolbar configuration for data tables and lists."""
    search: Optional[Dict[str, Any]] = None  # {field: str, placeholder?: str}
    filters: List[Dict[str, Any]] = field(default_factory=list)
    bulk_actions: List[ConditionalAction] = field(default_factory=list)
    actions: List[ConditionalAction] = field(default_factory=list)  # Top-level toolbar actions


@dataclass
class ShowDataTable(Statement):
    """
    Professional data table component with sorting, filtering, and actions.
    
    Example:
        show data_table "Orders" from dataset orders:
            columns:
                - id: order_id
                  label: "Order #"
                  width: 120
                  sortable: true
                - id: customer
                  label: "Customer"
                  sortable: true
                - id: total
                  label: "Total"
                  align: right
                  format: currency
                  sortable: true
                - id: status
                  label: "Status"
                  sortable: true
            row_actions:
                - label: "View Details"
                  icon: eye
                  action: view_order
                  params: "{{ id }}"
                - label: "Cancel"
                  icon: x
                  style: danger
                  action: cancel_order
                  params: "{{ id }}"
                  condition: "status == 'pending'"
            toolbar:
                search:
                    field: customer
                    placeholder: "Search customers..."
                filters:
                    - field: status
                      label: "Status"
                      options: ["pending", "completed", "cancelled"]
                bulk_actions:
                    - label: "Export Selected"
                      icon: download
                      action: export_orders
    """
    title: str
    source_type: str  # "dataset" | "table" | "frame" | "endpoint"
    source: str
    
    # Column configuration
    columns: List[ColumnConfig] = field(default_factory=list)
    
    # Row-level actions
    row_actions: List[ConditionalAction] = field(default_factory=list)
    
    # Toolbar configuration
    toolbar: Optional[ToolbarConfig] = None
    
    # Filtering and sorting
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
    default_sort: Optional[Dict[str, str]] = None  # {column: str, direction: "asc"|"desc"}
    
    # Pagination
    page_size: int = 50
    enable_pagination: bool = True
    
    # Empty state
    empty_state: Optional[EmptyStateConfig] = None
    
    # Layout and styling
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingConfig] = None


@dataclass
class ListItemConfig:
    """Configuration for individual list items."""
    title: Union[str, Dict[str, Any]]  # Field name or template (required, must come first)
    avatar: Optional[Dict[str, Any]] = None  # Avatar binding or configuration
    subtitle: Optional[Union[str, Dict[str, Any]]] = None
    metadata: Dict[str, Union[str, Dict[str, Any]]] = field(default_factory=dict)  # Key-value metadata
    actions: List[ConditionalAction] = field(default_factory=list)
    badge: Optional[Union[BadgeConfig, Dict[str, Any]]] = None
    icon: Optional[str] = None
    state_class: Optional[Dict[str, str]] = None  # Dynamic CSS classes


@dataclass
class ShowDataList(Statement):
    """
    Vertical list component for activity feeds, summaries, and compact displays.
    
    Example:
        show data_list "Recent Activity" from dataset activity_log:
            item:
                avatar:
                    type: icon
                    icon: "{{ event_icon }}"
                    color: "{{ event_color }}"
                title: "{{ event_title }}"
                subtitle: "{{ event_description }}"
                metadata:
                    timestamp: "{{ created_at | format:'relative' }}"
                    user: "{{ user_name }}"
                actions:
                    - label: "View Details"
                      icon: arrow-right
                      action: view_activity
                      params: "{{ id }}"
    """
    title: str
    source_type: str
    source: str
    
    # Item configuration
    item: Optional[ListItemConfig] = None
    
    # List styling
    variant: str = "default"  # "default" | "compact" | "detailed"
    dividers: bool = True
    
    # Filtering and search
    filter_by: Optional[str] = None
    enable_search: bool = False
    search_placeholder: Optional[str] = None
    
    # Pagination
    page_size: int = 50
    enable_pagination: bool = True
    
    # Empty state
    empty_state: Optional[EmptyStateConfig] = None
    
    # Layout and styling
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingConfig] = None


@dataclass
class SparklineConfig:
    """Configuration for sparkline mini-charts."""
    data_source: str  # Binding to time series data
    x_field: str
    y_field: str
    color: Optional[str] = None
    variant: str = "line"  # "line" | "bar" | "area"


@dataclass
class ShowStatSummary(Statement):
    """
    KPI/metric card component for displaying statistics.
    
    Example:
        show stat_summary "Revenue" from dataset metrics:
            value: "{{ total_revenue }}"
            label: "Total Revenue"
            format: currency
            delta:
                value: "{{ revenue_change }}"
                format: percentage
            trend: "{{ revenue_trend }}"
            sparkline:
                data_source: revenue_history
                x_field: date
                y_field: amount
    """
    label: str
    source_type: str
    source: str
    
    # Value configuration
    value: str  # Expression or field binding
    format: Optional[str] = None  # "currency" | "number" | "percentage"
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    
    # Comparison metrics
    delta: Optional[Dict[str, Any]] = None  # {value: expr, format: str, label?: str}
    trend: Optional[Union[str, Dict[str, Any]]] = None  # "up" | "down" | "neutral" | expr
    comparison_period: Optional[str] = None  # Description like "vs last month"
    
    # Optional sparkline
    sparkline: Optional[SparklineConfig] = None
    
    # Styling
    color: Optional[str] = None
    icon: Optional[str] = None
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None

    # Extended configuration
    stats: Optional[List[Dict[str, Any]]] = None

    @property
    def title(self) -> str:
        return self.label

    @title.setter
    def title(self, value: str) -> None:
        self.label = value
    
    # Data binding
    binding: Optional[DataBindingConfig] = None


@dataclass
class TimelineItem:
    """Single timeline event configuration."""
    timestamp: Union[str, Dict[str, Any]]  # Field or expression
    title: Union[str, Dict[str, Any]]
    description: Optional[Union[str, Dict[str, Any]]] = None
    icon: Optional[Union[str, Dict[str, Any]]] = None
    user: Optional[Union[str, Dict[str, Any]]] = None
    status: Optional[Union[str, Dict[str, Any]]] = None  # "success" | "warning" | "error" | "info" | expr
    color: Optional[str] = None
    actions: List[ConditionalAction] = field(default_factory=list)


@dataclass
class ShowTimeline(Statement):
    """
    Timeline component for events, logs, and activity history.
    
    Example:
        show timeline "Order History" from dataset order_events:
            item:
                timestamp: "{{ event_time }}"
                title: "{{ event_title }}"
                description: "{{ event_description }}"
                icon: "{{ event_icon }}"
                status: "{{ event_status }}"
                actions:
                    - label: "View Details"
                      icon: eye
                      action: view_event
                      params: "{{ id }}"
    """
    title: str
    source_type: str
    source: str
    
    # Item configuration
    item: Optional[TimelineItem] = None
    
    # Display options
    variant: str = "default"  # "default" | "compact" | "detailed"
    show_timestamps: bool = True
    group_by_date: bool = False
    
    # Filtering
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
    
    # Pagination
    page_size: int = 50
    enable_pagination: bool = True
    
    # Empty state
    empty_state: Optional[EmptyStateConfig] = None
    
    # Layout and styling
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingConfig] = None


@dataclass
class AvatarItem:
    """Single avatar configuration for avatar groups."""
    name: Optional[Union[str, Dict[str, Any]]] = None  # Field or expression
    image_url: Optional[Union[str, Dict[str, Any]]] = None
    initials: Optional[Union[str, Dict[str, Any]]] = None
    color: Optional[Union[str, Dict[str, Any]]] = None
    status: Optional[Union[str, Dict[str, Any]]] = None  # "online" | "offline" | "busy" | "away"
    tooltip: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class ShowAvatarGroup(Statement):
    """
    Avatar group component for displaying multiple users/agents compactly.
    
    Example:
        show avatar_group "Team Members" from dataset team_members:
            item:
                name: "{{ name }}"
                image_url: "{{ avatar_url }}"
                initials: "{{ initials }}"
                status: "{{ online_status }}"
                tooltip: "{{ name }} - {{ role }}"
            max_visible: 5
            size: md
    """
    source_type: str  # Required fields first
    source: str
    title: Optional[str] = None
    
    # Item configuration
    item: Optional[AvatarItem] = None
    
    # Display options
    max_visible: int = 5  # Show "+N more" for additional avatars
    size: str = "md"  # "xs" | "sm" | "md" | "lg" | "xl"
    variant: str = "stacked"  # "stacked" | "grid"
    
    # Filtering
    filter_by: Optional[str] = None
    
    # Layout and styling
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None
    
    # Data binding
    binding: Optional[DataBindingConfig] = None


@dataclass
class ChartConfig:
    """Enhanced chart configuration with multi-series support."""
    # Data mapping (required fields first)
    x_field: str
    variant: str = "line"  # "line" | "bar" | "pie" | "area" | "scatter" | "combo"
    y_fields: List[str] = field(default_factory=list)  # Multiple metrics for multi-series
    group_by: Optional[str] = None  # For series grouping
    
    # Chart styling
    stacked: bool = False  # For bar/area charts
    smooth: bool = True  # For line/area charts
    fill: bool = True  # For area charts
    
    # Legend configuration
    legend: Optional[Dict[str, Any]] = None  # {position: "top"|"bottom"|"left"|"right", show: bool}
    
    # Tooltip configuration
    tooltip: Optional[Dict[str, Any]] = None  # {show: bool, format: str}
    
    # Axes configuration
    x_axis: Optional[Dict[str, Any]] = None  # {label: str, format: str, rotate: number}
    y_axis: Optional[Dict[str, Any]] = None  # {label: str, format: str}
    
    # Color scheme
    colors: List[str] = field(default_factory=list)
    color_scheme: Optional[str] = None  # Predefined color scheme name


@dataclass
class ShowDataChart(Statement):
    """
    Advanced chart component with multi-series support and variants.
    
    Example:
        show data_chart "Sales Trends" from dataset sales_data:
            config:
                variant: line
                x_field: date
                y_fields: [revenue, costs, profit]
                group_by: region
                smooth: true
                legend:
                    position: bottom
                    show: true
                tooltip:
                    show: true
                    format: currency
                x_axis:
                    label: "Date"
                    format: "MMM YYYY"
                y_axis:
                    label: "Amount"
                    format: currency
    """
    title: str
    source_type: str
    source: str
    
    # Chart configuration
    config: Optional[ChartConfig] = None
    
    # Filtering
    filter_by: Optional[str] = None
    sort_by: Optional[str] = None
    
    # Empty state
    empty_state: Optional[EmptyStateConfig] = None
    
    # Layout and styling
    layout: Optional[LayoutMeta] = None
    style: Optional[Dict[str, Any]] = None
    height: Optional[Union[str, int]] = None
    
    # Data binding
    binding: Optional[DataBindingConfig] = None


# =============================================================================
# Layout Primitives (Stack, Grid, Split, Tabs, Accordion)
# =============================================================================

@dataclass
class StackLayout(Statement):
    """
    Flexbox-like stack layout for arranging children linearly.
    
    Example:
        layout stack:
            direction: vertical
            gap: medium
            align: center
            children:
                - show card "Data" from dataset items
                - show chart "Trends" from dataset metrics
    """
    direction: str = "vertical"  # "vertical" | "horizontal"
    gap: Union[str, int] = "medium"  # "small" | "medium" | "large" | numeric px value
    align: str = "stretch"  # "start" | "center" | "end" | "stretch"
    justify: str = "start"  # "start" | "center" | "end" | "space_between" | "space_around" | "space_evenly"
    wrap: bool = False
    
    # Children can be any page statement (cards, charts, text, nested layouts)
    children: List['PageStatement'] = field(default_factory=list)
    
    # Styling
    style: Optional[Dict[str, Any]] = None
    layout: Optional[LayoutMeta] = None


@dataclass
class GridLayout(Statement):
    """
    General-purpose grid layout for arranging children in a grid.
    
    Example:
        layout grid:
            columns: 3
            gap: large
            responsive: true
            min_column_width: 300px
            children:
                - show card "Sales" from dataset sales
                - show card "Leads" from dataset leads
                - show chart "Growth" from dataset metrics
    """
    columns: Union[int, str] = "auto"  # Number of columns or "auto"
    min_column_width: Optional[str] = None  # "200px" | "12rem" | token name
    gap: Union[str, int] = "medium"  # "small" | "medium" | "large" | numeric px value
    responsive: bool = True  # Adapt to viewport using min_column_width
    
    # Children
    children: List['PageStatement'] = field(default_factory=list)
    
    # Styling
    style: Optional[Dict[str, Any]] = None
    layout: Optional[LayoutMeta] = None


@dataclass
class SplitLayout(Statement):
    """
    Split layout with two resizable panes (left/right or top/bottom).
    
    Example:
        layout split:
            ratio: 0.3
            resizable: true
            left:
                - show table "Orders" from dataset orders
            right:
                - show card "Details" from dataset order_details
    """
    left: List['PageStatement'] = field(default_factory=list)
    right: List['PageStatement'] = field(default_factory=list)
    ratio: float = 0.5  # 0.0 to 1.0, proportion allocated to left pane
    resizable: bool = False  # Allow user to drag-resize the split
    orientation: str = "horizontal"  # "horizontal" (left/right) | "vertical" (top/bottom)
    
    # Styling
    style: Optional[Dict[str, Any]] = None
    layout: Optional[LayoutMeta] = None


@dataclass
class TabItem:
    """Single tab configuration for TabsLayout."""
    id: str
    label: str
    icon: Optional[str] = None
    badge: Optional[Union[str, BadgeConfig]] = None
    content: List['PageStatement'] = field(default_factory=list)


@dataclass
class TabsLayout(Statement):
    """
    Tabbed interface for switching between multiple content sections.
    
    Example:
        layout tabs:
            default_tab: overview
            tabs:
                - id: overview
                  label: "Overview"
                  icon: home
                  content:
                      - show card "Summary" from dataset summary
                      
                - id: details
                  label: "Details"
                  icon: list
                  content:
                      - show table "Data" from dataset items
    """
    tabs: List[TabItem] = field(default_factory=list)
    default_tab: Optional[str] = None  # Must match one of tabs[].id
    persist_state: bool = True  # Persist active tab in URL or local state
    
    # Styling
    style: Optional[Dict[str, Any]] = None
    layout: Optional[LayoutMeta] = None


@dataclass
class AccordionItem:
    """Single accordion item configuration."""
    id: str
    title: str
    description: Optional[str] = None
    icon: Optional[str] = None
    content: List['PageStatement'] = field(default_factory=list)
    default_open: bool = False


@dataclass
class AccordionLayout(Statement):
    """
    Collapsible accordion layout for structured content sections.
    
    Example:
        layout accordion:
            multiple: true
            items:
                - id: section1
                  title: "Personal Information"
                  icon: user
                  default_open: true
                  content:
                      - show form "Profile" with fields name, email
                      
                - id: section2
                  title: "Settings"
                  icon: settings
                  content:
                      - show form "Preferences" with fields theme, language
    """
    items: List[AccordionItem] = field(default_factory=list)
    multiple: bool = False  # Allow multiple items to be expanded at once
    
    # Styling
    style: Optional[Dict[str, Any]] = None
    layout: Optional[LayoutMeta] = None

    @property
    def allow_multiple(self) -> bool:
        return self.multiple

    @allow_multiple.setter
    def allow_multiple(self, value: bool) -> None:
        self.multiple = value

    @property
    def sections(self) -> List[AccordionItem]:
        return self.items

    @sections.setter
    def sections(self, value: List[AccordionItem]) -> None:
        self.items = value


# =============================================================================
# NAVIGATION & APP CHROME COMPONENTS
# =============================================================================


@dataclass
class NavItem:
    """Navigation item for sidebar or navbar."""
    id: str
    label: str
    route: Optional[str] = None
    icon: Optional[str] = None
    badge: Optional[BadgeConfig] = None
    action: Optional[str] = None  # Action ID if this triggers an action instead of navigation
    condition: Optional[str] = None  # Conditional visibility
    children: List['NavItem'] = field(default_factory=list)  # For nested navigation


@dataclass
class NavSection:
    """Section grouping in sidebar navigation."""
    id: str
    label: str
    items: List[str] = field(default_factory=list)  # IDs of nav items
    collapsible: bool = False
    collapsed_by_default: bool = False


@dataclass
class NavbarAction:
    """Action button in navbar (menu, toggle, button)."""
    id: str
    label: Optional[str] = None
    icon: Optional[str] = None
    type: Literal["button", "menu", "toggle"] = "button"
    action: Optional[str] = None  # Action ID to trigger
    menu_items: List[NavItem] = field(default_factory=list)  # For type="menu"
    condition: Optional[str] = None  # Conditional visibility


@dataclass
class Sidebar(Statement):
    """
    App-level sidebar navigation.
    
    Provides primary navigation through app routes with support for:
    - Hierarchical navigation items
    - Section grouping
    - Collapsible sidebar
    - Icons and badges
    - Conditional visibility
    """
    items: List[NavItem] = field(default_factory=list)
    sections: List[NavSection] = field(default_factory=list)
    collapsible: bool = False
    collapsed_by_default: bool = False
    width: Optional[str] = None  # "narrow" | "normal" | "wide" | specific px/rem
    position: Literal["left", "right"] = "left"


@dataclass
class Navbar(Statement):
    """
    Top-level application navigation bar.
    
    Provides app branding and global actions like:
    - User menu
    - Theme toggle
    - Notifications
    - Search trigger
    """
    logo: Optional[str] = None  # Asset reference or text
    title: Optional[str] = None  # App title or expression
    actions: List[NavbarAction] = field(default_factory=list)
    position: Literal["top", "bottom"] = "top"
    sticky: bool = True


@dataclass
class BreadcrumbItem:
    """Single breadcrumb item."""
    label: str  # Can be expression like "{{page_title}}"
    route: Optional[str] = None  # If None, renders as text (current page)


@dataclass
class Breadcrumbs(Statement):
    """
    Breadcrumb navigation showing page hierarchy.
    
    Supports both explicit items and auto-derivation from routing.
    """
    items: List[BreadcrumbItem] = field(default_factory=list)
    auto_derive: bool = False  # Auto-generate from route hierarchy
    separator: str = "/"  # Separator character between breadcrumbs


@dataclass
class CommandSource:
    """Source configuration for command palette."""
    type: Literal["routes", "actions", "custom", "api"] = "routes"
    filter: Optional[str] = None  # Filter expression
    custom_items: List[Dict[str, Any]] = field(default_factory=list)  # For type="custom"
    # API-backed source fields
    id: Optional[str] = None  # Unique identifier for the source
    endpoint: Optional[str] = None  # API endpoint URL
    label: Optional[str] = None  # Display label for the source


@dataclass
class CommandPalette(Statement):
    """
    Power-user command interface (Ctrl+K / Cmd+K).
    
    Provides quick access to:
    - Navigation to any route
    - Execution of registered actions
    - Custom commands
    """
    shortcut: str = "ctrl+k"  # Keyboard shortcut to trigger
    sources: List[CommandSource] = field(default_factory=list)
    placeholder: str = "Search commands..."
    max_results: int = 10


# ============================================================
# FEEDBACK COMPONENTS (Modal, Toast)
# ============================================================


@dataclass
class ModalAction:
    """Action button in a modal dialog."""
    label: str
    action: Optional[str] = None  # Action name to trigger
    variant: Optional[Literal["default", "primary", "destructive", "ghost", "link"]] = "default"  # DEPRECATED, use tone
    close: bool = True  # Whether clicking closes modal
    
    # Design tokens (for button styling)
    button_variant: Optional[VariantType] = None
    button_tone: Optional[ToneType] = None
    button_size: Optional[SizeType] = None


@dataclass
class Modal(Statement):
    """
    Modal dialog overlay for focused interactions.
    
    Use for:
    - Confirmations (delete, submit)
    - Forms requiring focus
    - Complex multi-step workflows
    - Critical information display
    """
    id: str  # Unique identifier for opening/closing
    title: str
    description: Optional[str] = None
    content: List[Statement] = field(default_factory=list)  # Content inside modal
    actions: List[ModalAction] = field(default_factory=list)  # Footer buttons
    size: Literal["sm", "md", "lg", "xl", "full"] = "md"  # DEPRECATED, use design token size
    dismissible: bool = True  # Can close with ESC or backdrop click
    trigger: Optional[str] = None  # Action name that opens modal
    
    # Design tokens
    variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None
    modal_size: Optional[SizeType] = None  # XS/SM/MD/LG/XL


@dataclass
class Toast(Statement):
    """
    Temporary notification message.
    
    Use for:
    - Success confirmations ("Saved successfully")
    - Error messages ("Failed to save")
    - Info messages ("Processing...")
    - Warning messages ("Unsaved changes")
    """
    id: str  # Unique identifier
    title: str
    description: Optional[str] = None
    variant: Literal["default", "success", "error", "warning", "info"] = "default"  # DEPRECATED, use tone
    duration: int = 3000  # Auto-dismiss after ms (0 = manual dismiss only)
    action_label: Optional[str] = None  # Optional action button
    action: Optional[str] = None  # Action to trigger on button click
    position: Literal["top", "top-right", "top-left", "bottom", "bottom-right", "bottom-left"] = "top-right"
    trigger: Optional[str] = None  # Action name that shows toast
    
    # Design tokens
    toast_variant: Optional[VariantType] = None
    tone: Optional[ToneType] = None  # Replaces variant (success/error/warning/info → success/danger/warning/primary)
    size: Optional[SizeType] = None


# =============================================================================
# AI Semantic Components
# =============================================================================

@dataclass
class ChatThread(Statement):
    """
    Multi-message AI conversation display.
    
    Displays a conversation between user, assistant, system, and agent roles.
    Supports streaming, auto-scroll, and message grouping.
    
    Binds to real conversation data from agents/runtime, never demo data.
    """
    id: str  # Unique identifier
    messages_binding: str  # Binding to conversation/messages list (e.g., "conversation.messages", "agent.chat_history")
    group_by: Literal["role", "speaker", "timestamp", "none"] = "role"
    show_timestamps: bool = True
    show_avatar: bool = True
    reverse_order: bool = False  # True for newest-first
    auto_scroll: bool = True  # Auto-scroll to latest message
    max_height: Optional[str] = None  # CSS height value (e.g., "600px", "80vh")
    # Streaming configuration
    streaming_enabled: bool = False
    streaming_source: Optional[str] = None  # Binding to streaming token source
    # Display options
    show_role_labels: bool = True
    show_token_count: bool = False
    enable_copy: bool = True  # Copy message content to clipboard
    enable_regenerate: bool = False  # Regenerate response button
    # Styling
    variant: Literal["default", "compact", "detailed"] = "default"


@dataclass
class AgentPanel(Statement):
    """
    Display agent-level state and metrics.
    
    Shows current agent info, status, tokens, cost, latency, and environment.
    Binds to real agent runtime state and metrics instrumentation.
    """
    id: str  # Unique identifier
    agent_binding: str  # Binding to agent or graph node (e.g., "current_agent", "agent.researcher")
    metrics_binding: Optional[str] = None  # Binding to metrics (e.g., "run.metrics", "agent.stats")
    # Display configuration
    show_status: bool = True
    show_metrics: bool = True
    show_profile: bool = False  # Show environment (dev/staging/prod)
    show_limits: bool = False  # Show rate limits, quotas
    show_last_error: bool = False
    show_tools: bool = False  # Show available tools
    # Metrics to display
    show_tokens: bool = True  # prompt/completion/total tokens
    show_cost: bool = True  # estimated cost
    show_latency: bool = True  # response time
    show_model: bool = True  # model name
    # Layout
    variant: Literal["card", "inline", "sidebar"] = "card"
    compact: bool = False


@dataclass
class ToolCallView(Statement):
    """
    Display tool invocations and their details.
    
    Shows tool calls with inputs, outputs, timing, and status.
    Binds to real tool invocation logs from runtime/logging system.
    """
    id: str  # Unique identifier
    calls_binding: str  # Binding to tool invocation logs (e.g., "run.tool_calls", "agent.tools_used")
    # Display configuration
    show_inputs: bool = True
    show_outputs: bool = True
    show_timing: bool = True
    show_status: bool = True
    show_raw_payload: bool = False  # Show raw JSON
    # Filtering
    filter_tool_name: Optional[List[str]] = None  # Filter by tool names
    filter_status: Optional[List[str]] = None  # Filter by status (success/failed/cancelled)
    # Layout
    variant: Literal["list", "table", "timeline"] = "list"
    expandable: bool = True  # Collapsible tool call details
    max_height: Optional[str] = None
    # Interaction
    enable_retry: bool = False  # Retry failed calls
    enable_copy: bool = True  # Copy call details


@dataclass
class LogView(Statement):
    """
    Tail/inspect logs and traces for a run.
    
    Displays logs with levels, timestamps, messages, and metadata.
    Binds to real logging data from runtime, supports large volumes.
    """
    id: str  # Unique identifier
    logs_binding: str  # Binding to log entries (e.g., "run.logs", "agent.traces", "app.logs")
    # Filtering
    level_filter: Optional[List[str]] = None  # Filter by level (info/warn/error/debug)
    search_enabled: bool = True  # Enable search box
    search_placeholder: str = "Search logs..."
    # Display configuration
    show_timestamp: bool = True
    show_level: bool = True
    show_metadata: bool = False  # Show structured metadata (JSON)
    show_source: bool = False  # Show log source/module
    # Behavior
    auto_scroll: bool = True  # Tail mode
    auto_refresh: bool = False
    refresh_interval: int = 5000  # ms
    max_entries: int = 1000  # Limit displayed entries
    # Layout
    variant: Literal["default", "compact", "detailed"] = "default"
    max_height: Optional[str] = None
    virtualized: bool = True  # Use virtualization for large logs
    # Interaction
    enable_copy: bool = True
    enable_download: bool = False  # Download logs as file


@dataclass
class EvaluationResult(Statement):
    """
    Display evaluation run metrics, histograms, and error analysis.
    
    Shows aggregate metrics, distributions, and error slices from eval runs.
    Binds to real evaluation data from evaluation infrastructure.
    """
    id: str  # Unique identifier
    eval_run_binding: str  # Binding to evaluation run (e.g., "eval.run_123", "latest_eval")
    # Display configuration
    show_summary: bool = True  # Aggregate metrics summary
    show_histograms: bool = True  # Score distributions
    show_error_table: bool = True  # Per-example errors
    show_metadata: bool = False  # Run metadata (timestamp, config, etc.)
    # Metrics configuration
    metrics_to_show: Optional[List[str]] = None  # Specific metrics to display
    primary_metric: Optional[str] = None  # Highlighted metric
    # Filtering
    filter_metric: Optional[str] = None  # Filter examples by metric
    filter_min_score: Optional[float] = None
    filter_max_score: Optional[float] = None
    filter_status: Optional[List[str]] = None  # Filter by pass/fail
    # Error analysis
    show_error_distribution: bool = True
    show_error_examples: bool = True
    max_error_examples: int = 10
    # Layout
    variant: Literal["dashboard", "detailed", "compact"] = "dashboard"
    # Comparison
    comparison_run_binding: Optional[str] = None  # Compare with another run


@dataclass
class DiffView(Statement):
    """
    Compare model outputs, prompts, or documents side-by-side or inline.
    
    Renders diffs for text or code with syntax highlighting.
    Uses library-backed diff algorithm, not naive string comparison.
    """
    id: str  # Unique identifier
    left_binding: str  # Binding to "before" text/code (e.g., "version.v1.output", "prompt.original")
    right_binding: str  # Binding to "after" text/code (e.g., "version.v2.output", "prompt.modified")
    # Display configuration
    mode: Literal["unified", "split"] = "split"
    content_type: Literal["text", "code", "markdown"] = "text"
    language: Optional[str] = None  # Programming language for syntax highlighting (e.g., "python", "javascript")
    # Diff options
    ignore_whitespace: bool = False
    ignore_case: bool = False
    context_lines: int = 3  # Lines of context around changes
    # Display options
    show_line_numbers: bool = True
    highlight_inline_changes: bool = True  # Highlight word-level changes
    show_legend: bool = True  # Show addition/deletion legend
    # Layout
    max_height: Optional[str] = None
    # Interaction
    enable_copy: bool = True
    enable_download: bool = False  # Download diff as file


ActionOperationType = Union[
    UpdateOperation,
    ToastOperation,
    GoToPageOperation,
    CallPythonOperation,
    AskConnectorOperation,
    RunChainOperation,
    RunPromptOperation,
]
PageStatement = Union[
    ShowText,
    ShowTable,
    ShowChart,
    ShowForm,
    ShowCard,
    ShowList,
    ShowDataTable,
    ShowDataList,
    ShowStatSummary,
    ShowTimeline,
    ShowAvatarGroup,
    ShowDataChart,
    StackLayout,
    GridLayout,
    SplitLayout,
    TabsLayout,
    AccordionLayout,
    # Navigation & Chrome
    Sidebar,
    Navbar,
    Breadcrumbs,
    CommandPalette,
    # Feedback Components
    Modal,
    Toast,
    # AI Semantic Components
    ChatThread,
    AgentPanel,
    ToolCallView,
    LogView,
    EvaluationResult,
    DiffView,
    # Control flow & Actions
    Action,
    IfBlock,
    ForLoop,
    WhileLoop,
    VariableAssignment,
    PredictStatement,
    BreakStatement,
    ContinueStatement,
    LogStatement,
]


__all__ = [
    "Statement",
    "ShowText",
    "LayoutSpec",
    "LayoutMeta",
    "DataBindingConfig",
    "ShowTable",
    "ShowChart",
    "FormField",
    "ShowForm",
    "ShowCard",
    "ShowList",
    "EmptyStateConfig",
    "BadgeConfig",
    "FieldValueConfig",
    "InfoGridItem",
    "CardSection",
    "ConditionalAction",
    "CardHeader",
    "CardFooter",
    "CardItemConfig",
    # Data display components
    "ColumnConfig",
    "ToolbarConfig",
    "ShowDataTable",
    "ListItemConfig",
    "ShowDataList",
    "SparklineConfig",
    "ShowStatSummary",
    "TimelineItem",
    "ShowTimeline",
    "AvatarItem",
    "ShowAvatarGroup",
    "ChartConfig",
    "ShowDataChart",
    # Layout primitives
    "StackLayout",
    "GridLayout",
    "SplitLayout",
    "TabsLayout",
    "AccordionLayout",
    "TabItem",
    "AccordionItem",
    # Navigation & Chrome
    "NavItem",
    "NavSection",
    "NavbarAction",
    "Sidebar",
    "Navbar",
    "BreadcrumbItem",
    "Breadcrumbs",
    "CommandSource",
    "CommandPalette",
    # Feedback Components
    "ModalAction",
    "Modal",
    "Toast",
    # AI Semantic Components
    "ChatThread",
    "AgentPanel",
    "ToolCallView",
    "LogView",
    "EvaluationResult",
    "DiffView",
    # Control flow and actions
    "Action",
    "VariableAssignment",
    "IfBlock",
    "ForLoop",
    "WhileLoop",
    "PredictStatement",
    "ElifBlock",
    "BreakStatement",
    "ContinueStatement",
    "UpdateOperation",
    "ToastOperation",
    "GoToPageOperation",
    "CallPythonOperation",
    "AskConnectorOperation",
    "RunChainOperation",
    "RunPromptOperation",
    "ActionOperation",
    "ActionOperationType",
    "PageStatement",
    "LogStatement",
]
