# Layout Primitives - Complete Reference

## Overview

Namel3ss provides five production-ready layout primitives that enable you to build sophisticated, responsive dashboards and applications without writing HTML or CSS. These primitives are first-class language constructs that work seamlessly with all other UI components (cards, charts, forms, etc.).

## Layout Primitives

### 1. Stack Layout

**Purpose**: Arrange children in a linear (vertical or horizontal) flow with flexbox-like control.

**Syntax**:
```yaml
layout stack:
  direction: vertical | horizontal
  gap: small | medium | large | <px value>
  align: start | center | end | stretch
  justify: start | center | end | space_between | space_around | space_evenly
  wrap: true | false
  
  children:
    - <any page statement>
    - <any page statement>
```

**Properties**:
- `direction` (default: `"vertical"`): Main axis direction
  - `"vertical"`: Stack items top-to-bottom
  - `"horizontal"`: Stack items left-to-right
  
- `gap` (default: `"medium"`): Space between children
  - Named tokens: `"small"`, `"medium"`, `"large"`
  - Numeric: Any positive integer (pixels)
  
- `align` (default: `"stretch"`): Cross-axis alignment
  - `"start"`: Align to start of cross axis
  - `"center"`: Center on cross axis
  - `"end"`: Align to end of cross axis
  - `"stretch"`: Stretch to fill cross axis
  
- `justify` (default: `"start"`): Main-axis distribution
  - `"start"`: Pack items at start
  - `"center"`: Center items
  - `"end"`: Pack items at end
  - `"space_between"`: Equal space between items
  - `"space_around"`: Equal space around items
  - `"space_evenly"`: Equal space between and around
  
- `wrap` (default: `false`): Allow wrapping to next line

**Example**:
```yaml
page dashboard:
  path: "/dashboard"
  
  layout stack:
    direction: vertical
    gap: large
    align: stretch
    
    children:
      # Header metrics in horizontal stack
      - layout stack:
          direction: horizontal
          gap: medium
          justify: space_between
          
          children:
            - show card "Users" from dataset user_metrics
            - show card "Revenue" from dataset revenue_metrics
            - show card "Conversions" from dataset conversion_metrics
      
      # Main content area
      - show chart "Trends" from dataset daily_trends
```

---

### 2. Grid Layout

**Purpose**: Arrange children in a responsive grid with automatic or fixed columns.

**Syntax**:
```yaml
layout grid:
  columns: <integer> | auto
  min_column_width: <css length>
  gap: small | medium | large | <px value>
  responsive: true | false
  
  children:
    - <any page statement>
    - <any page statement>
```

**Properties**:
- `columns` (default: `"auto"`): Number of columns
  - `"auto"`: Automatic based on `min_column_width`
  - Integer: Fixed number of columns
  
- `min_column_width` (optional): Minimum width per column
  - Examples: `"200px"`, `"12rem"`, `"15em"`
  - Only used when `columns: auto`
  
- `gap` (default: `"medium"`): Space between grid items
  
- `responsive` (default: `true`): Adapt to viewport
  - When `true`, grid adjusts columns based on available space
  - When `false`, maintains fixed layout

**Example**:
```yaml
page products:
  path: "/products"
  
  layout grid:
    columns: auto
    min_column_width: 300px
    gap: large
    responsive: true
    
    children:
      - show card "Electronics" from dataset products:
          filter_by: "category == 'electronics'"
      
      - show card "Clothing" from dataset products:
          filter_by: "category == 'clothing'"
      
      - show card "Home & Garden" from dataset products:
          filter_by: "category == 'home'"
      
      - show card "Sports" from dataset products:
          filter_by: "category == 'sports'"
```

---

### 3. Split Layout

**Purpose**: Create a two-pane layout (sidebar/content, editor/preview, etc.) with optional resizing.

**Syntax**:
```yaml
layout split:
  ratio: <float 0.0-1.0>
  resizable: true | false
  orientation: horizontal | vertical
  
  left:  # or "top" for vertical orientation
    - <any page statement>
  
  right:  # or "bottom" for vertical orientation
    - <any page statement>
```

**Properties**:
- `ratio` (default: `0.5`): Proportion allocated to left/top pane
  - Range: `0.0` to `1.0`
  - Example: `0.3` means 30% left, 70% right
  
- `resizable` (default: `false`): Enable drag-to-resize
  - When `true`, user can adjust split ratio
  - Persists in component state
  
- `orientation` (default: `"horizontal"`): Split direction
  - `"horizontal"`: Left and right panes
  - `"vertical"`: Top and bottom panes

**Example**:
```yaml
page orders:
  path: "/orders"
  
  layout split:
    ratio: 0.4
    resizable: true
    orientation: horizontal
    
    left:
      - show card "Order List" from dataset orders:
          item:
            type: card
            actions:
              - label: "View Details"
                action: select_order
                params: "{{ id }}"
    
    right:
      - show card "Order Details" from dataset order_details:
          filter_by: "order_id == selected_order_id"
          
          empty_state:
            icon: inbox
            title: "Select an order"
            message: "Click an order to view details"
```

---

### 4. Tabs Layout

**Purpose**: Create tabbed navigation for switching between multiple content sections.

**Syntax**:
```yaml
layout tabs:
  default_tab: <tab_id>
  persist_state: true | false
  
  tabs:
    - id: <unique_id>
      label: "<display label>"
      icon: <icon_name>
      badge: "<badge text or expression>"
      
      content:
        - <any page statement>
        - <any page statement>
```

**Properties**:
- `default_tab` (optional): ID of initially active tab
  - Must match one of `tabs[].id`
  - If omitted, first tab is active
  
- `persist_state` (default: `true`): Save active tab
  - When `true`, persists tab state in URL or local storage
  - Enables deep linking to specific tabs

**Tab Item Properties**:
- `id` (required): Unique identifier
- `label` (required): Display text
- `icon` (optional): Icon name (uses project icon system)
- `badge` (optional): Badge text or template expression
- `content` (required): List of page statements

**Example**:
```yaml
page analytics:
  path: "/analytics"
  
  layout tabs:
    default_tab: overview
    persist_state: true
    
    tabs:
      - id: overview
        label: "Overview"
        icon: home
        
        content:
          - layout grid:
              columns: 2
              children:
                - show chart "Revenue" from dataset revenue
                - show chart "Users" from dataset users
      
      - id: sales
        label: "Sales"
        icon: shopping-cart
        badge: "{{ new_orders_count }}"
        
        content:
          - show card "Recent Sales" from dataset sales
          - show chart "Sales Trends" from dataset sales_trends
      
      - id: customers
        label: "Customers"
        icon: users
        
        content:
          - show card "Customer List" from dataset customers
          - show chart "Segments" from dataset customer_segments
```

---

### 5. Accordion Layout

**Purpose**: Create collapsible sections for organizing structured content.

**Syntax**:
```yaml
layout accordion:
  multiple: true | false
  
  items:
    - id: <unique_id>
      title: "<section title>"
      description: "<optional description>"
      icon: <icon_name>
      default_open: true | false
      
      content:
        - <any page statement>
        - <any page statement>
```

**Properties**:
- `multiple` (default: `false`): Allow multiple sections expanded
  - When `false`, expanding one section collapses others
  - When `true`, any number of sections can be open

**Accordion Item Properties**:
- `id` (required): Unique identifier
- `title` (required): Section header text
- `description` (optional): Subtitle/help text
- `icon` (optional): Icon name
- `default_open` (default: `false`): Initially expanded
- `content` (required): List of page statements

**Example**:
```yaml
page settings:
  path: "/settings"
  
  layout accordion:
    multiple: false
    
    items:
      - id: profile
        title: "Profile Settings"
        description: "Manage your personal information"
        icon: user
        default_open: true
        
        content:
          - show form "Update Profile":
              fields:
                - name: full_name
                  type: text
                  label: "Full Name"
                
                - name: email
                  type: email
                  label: "Email Address"
      
      - id: security
        title: "Security"
        description: "Password and authentication settings"
        icon: lock
        default_open: false
        
        content:
          - show form "Change Password":
              fields:
                - name: current_password
                  type: password
                  label: "Current Password"
                
                - name: new_password
                  type: password
                  label: "New Password"
      
      - id: notifications
        title: "Notifications"
        description: "Configure notification preferences"
        icon: bell
        default_open: false
        
        content:
          - show form "Notification Preferences":
              fields:
                - name: email_notifications
                  type: checkbox
                  label: "Email Notifications"
                
                - name: push_notifications
                  type: checkbox
                  label: "Push Notifications"
```

---

## Nesting and Composition

All layout primitives support full nesting and composition. You can nest any layout inside any other layout to create sophisticated interfaces.

**Example**: Complex nested layouts
```yaml
page complex_dashboard:
  path: "/dashboard"
  
  # Top-level vertical stack
  layout stack:
    direction: vertical
    gap: large
    
    children:
      # Header grid
      - layout grid:
          columns: 4
          gap: medium
          children:
            - show card "Metric 1" from dataset metrics
            - show card "Metric 2" from dataset metrics
            - show card "Metric 3" from dataset metrics
            - show card "Metric 4" from dataset metrics
      
      # Main content with tabs
      - layout tabs:
          tabs:
            - id: workspace
              label: "Workspace"
              
              content:
                # Split layout inside tab
                - layout split:
                    ratio: 0.3
                    resizable: true
                    
                    left:
                      # Accordion in left pane
                      - layout accordion:
                          items:
                            - id: section1
                              title: "Section 1"
                              content:
                                - show card "Data" from dataset data1
                    
                    right:
                      # Grid in right pane
                      - layout grid:
                          columns: 2
                          children:
                            - show chart "Chart 1" from dataset chart_data
                            - show chart "Chart 2" from dataset chart_data
```

---

## Data Binding

Layout primitives work seamlessly with data binding. All child components (cards, charts, forms) maintain their data binding capabilities when placed inside layouts.

**Example**: Data-bound layouts
```yaml
layout split:
  ratio: 0.4
  resizable: true
  
  left:
    # List with real-time updates
    - show card "Live Orders" from dataset orders:
        binding:
          subscribe_to_changes: true
          auto_refresh: true
  
  right:
    # Details with conditional display
    - show card "Order Items" from dataset order_items:
        filter_by: "order_id == selected_order_id"
        
        binding:
          editable: true
          enable_update: true
```

---

## Responsive Behavior

Layouts automatically adapt to different screen sizes:

1. **Stack**: Respects `wrap` property for overflow
2. **Grid**: Uses `responsive` and `min_column_width` for automatic adjustment
3. **Split**: Provides mobile-friendly stacking on small screens
4. **Tabs**: Renders tab bar as dropdown on mobile
5. **Accordion**: Maintains collapsible behavior across all sizes

---

## Styling and Theming

All layout primitives accept optional `style` and `layout` properties for customization:

```yaml
layout stack:
  direction: vertical
  gap: medium
  
  style:
    background: "var(--color-surface)"
    border_radius: "8px"
    padding: "16px"
  
  layout:
    width: 1200
    variant: "contained"
  
  children:
    - show card "Data" from dataset data
```

---

## Accessibility

All layout primitives follow WCAG 2.1 AA standards:

- **Stack/Grid**: Proper semantic structure and focus order
- **Split**: Keyboard-accessible resize handle
- **Tabs**: ARIA tab roles, keyboard navigation (Arrow keys, Home, End)
- **Accordion**: ARIA accordion roles, keyboard navigation (Enter, Space)

---

## Performance Considerations

1. **Lazy Loading**: Child components in tabs/accordion lazy-load on first view
2. **Virtualization**: Large lists in grids automatically virtualize
3. **Memoization**: Layout calculations are cached and only recompute when props change
4. **Code Splitting**: Each layout type is independently loaded

---

## Examples

See comprehensive examples in:
- `examples/layout-primitives-demo.ai` - Full-featured dashboard
- `examples/hospital-ai/ui_clinician.ai` - Healthcare application
- `tests/parser/test_layout_primitives.py` - Test cases with valid syntax

---

## Migration Guide

### From HTML/CSS Layouts

**Before** (inline HTML):
```yaml
page old_dashboard:
  path: "/dashboard"
  
  ui:
    content:
      item_render: |
        <div class="dashboard-grid">
          <div class="grid-item">...</div>
          <div class="grid-item">...</div>
        </div>
```

**After** (layout primitives):
```yaml
page new_dashboard:
  path: "/dashboard"
  
  layout grid:
    columns: 2
    gap: medium
    
    children:
      - show card "Card 1" from dataset data1
      - show card "Card 2" from dataset data2
```

### Benefits of Migration

1. **Type Safety**: Layout props are validated at parse time
2. **Responsiveness**: Automatic responsive behavior without media queries
3. **Maintainability**: Declarative syntax is easier to read and modify
4. **Consistency**: Layouts use design system tokens automatically
5. **Accessibility**: Built-in ARIA roles and keyboard support

---

## Best Practices

1. **Use Stack for Simple Lists**: When arranging items in a single direction
2. **Use Grid for Equal-Width Items**: Cards, metrics, thumbnails
3. **Use Split for Master-Detail**: List/detail, sidebar/content patterns
4. **Use Tabs for Category Switching**: Different views of same data
5. **Use Accordion for Progressive Disclosure**: Long forms, settings panels

6. **Nest Thoughtfully**: Avoid excessive nesting (>3-4 levels)
7. **Name Tab/Accordion IDs Descriptively**: Use meaningful identifiers
8. **Set Reasonable Gap Values**: Consistent spacing improves readability
9. **Test Responsive Behavior**: Preview layouts at different viewport sizes
10. **Leverage Default Values**: Don't override defaults unnecessarily

---

## Troubleshooting

### Layout Not Rendering

**Issue**: Layout appears empty
**Solution**: Ensure `children` or `left/right` contain valid page statements

### Tabs Validation Error

**Issue**: "default_tab does not match any tab id"
**Solution**: Verify `default_tab` matches exactly one `tabs[].id`

### Split Ratio Error

**Issue**: "Ratio must be between 0.0 and 1.0"
**Solution**: Use decimal value (e.g., `0.3` not `30`)

### Nested Layouts Not Parsing

**Issue**: Child layout not recognized
**Solution**: Check indentation - children must be indented relative to parent

---

## API Reference

All layout primitives are defined in:
- **AST**: `namel3ss/ast/pages.py`
- **Parser**: `namel3ss/parser/components.py`
- **IR**: `namel3ss/ir/spec.py`
- **Codegen**: `namel3ss/codegen/frontend/react/`

For implementation details, see source code and inline documentation.
