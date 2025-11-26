# Data Display Components

**Status**: Production Ready  
**Version**: Added in Namel3ss 0.7.0  
**Last Updated**: November 26, 2025

This document describes the six new first-class data display components for building professional, data-rich dashboards and applications in Namel3ss.

---

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
   - [ShowDataTable](#showdatatable)
   - [ShowDataList](#showdatalist)
   - [ShowStatSummary](#showstatsummary)
   - [ShowTimeline](#showtimeline)
   - [ShowAvatarGroup](#showavatargroup)
   - [ShowDataChart](#showdatachart)
3. [Common Features](#common-features)
4. [Integration Patterns](#integration-patterns)
5. [Examples](#examples)

---

## Overview

The data display components provide production-ready UI elements for visualizing and interacting with data. Each component is implemented across the entire namel3ss stack:

- **Parser**: Native `.ai` syntax support
- **AST**: Strongly typed node definitions
- **IR**: Runtime-agnostic specifications
- **Codegen**: React/TypeScript component generation
- **Frontend**: Modern React components with proper data handling

### Design Principles

- **Production Quality**: No demo data, no shortcuts
- **Type Safety**: Full TypeScript support with proper interfaces
- **Accessibility**: ARIA-ready markup and keyboard navigation
- **Performance**: Optimized rendering with React hooks
- **Flexibility**: Extensive configuration options
- **Integration**: Seamless data binding with datasets and queries

---

## Components

### ShowDataTable

Professional data tables with sorting, filtering, pagination, and actions.

#### Syntax

```namel3ss
show data_table "Title" from dataset dataset_name:
  columns:
    - field: column_name
      header: "Column Header"
      width: "100px"              # optional
      sortable: true              # optional, default: true
      format: currency            # optional: currency, percentage, date, datetime, number
      align: left                 # optional: left, center, right
  
  toolbar:                        # optional
    searchable: true
    search_fields: ["field1", "field2"]
    
    filters:
      - field: status
        label: "Status"
        type: select              # select, multiselect, date, range
        options:
          - value: active
            label: "Active"
    
    bulk_actions:
      - label: "Delete Selected"
        action: bulk_delete
        icon: trash
        requires_selection: true  # optional
  
  row_actions:                    # optional
    - label: "View"
      action: view_item
      icon: eye
      condition: "is_visible == true"  # optional
  
  rows_per_page: 20               # optional, default: 10
  
  empty_state:                    # optional
    icon: table
    title: "No data"
    message: "No records to display"
    action_label: "Add Record"   # optional
    action_link: "/add"           # optional
```

#### Column Formats

- `currency`: Formats as USD currency ($1,234.56)
- `percentage`: Formats as percentage (45.2%)
- `date`: Formats as date (Jan 15, 2025)
- `datetime`: Formats as date and time (Jan 15, 2025, 3:30 PM)
- `number`: Formats with thousands separators (1,234,567)
- Custom formats via `format: { type: "number", decimals: 2 }`

#### Row Actions

Row actions are displayed as buttons in an "Actions" column. They support:
- Conditional display based on row data
- Icons for visual clarity
- Custom action handlers

#### Toolbar Features

**Search**: Full-text search across specified fields  
**Filters**: Dynamic filtering with multiple types  
**Bulk Actions**: Operate on selected rows

#### Example

```namel3ss
show data_table "Customer Orders" from dataset recent_orders:
  columns:
    - field: order_number
      header: "Order #"
      sortable: true
    - field: customer_name
      header: "Customer"
    - field: amount
      header: "Amount"
      format: currency
      align: right
    - field: status
      header: "Status"
  
  toolbar:
    searchable: true
    search_fields: ["order_number", "customer_name"]
    filters:
      - field: status
        label: "Order Status"
        type: select
        options:
          - value: pending
            label: "Pending"
          - value: completed
            label: "Completed"
  
  row_actions:
    - label: "View"
      action: view_order
      icon: eye
    - label: "Cancel"
      action: cancel_order
      icon: x
      condition: "status == 'pending'"
  
  rows_per_page: 25
```

---

### ShowDataList

Activity feeds and item lists with rich metadata display.

#### Syntax

```namel3ss
show data_list "Title" from dataset dataset_name:
  item:
    avatar:                       # optional
      field: avatar_url           # optional
      fallback: "?"               # optional
      size: sm | md | lg          # optional, default: md
    
    title:
      field: title_field          # either field or text required
      text: "{{ template }}"      # template with {{ field }} syntax
    
    subtitle:                     # optional
      field: subtitle_field
      text: "{{ template }}"
    
    metadata:                     # optional
      - field: timestamp
        icon: clock
        format: relative          # relative, date, datetime
      - text: "{{ count }} items"
        icon: package
    
    badge:                        # optional
      field: status
      text: "{{ status }}"
      style: status_badge
      transform: humanize         # optional
    
    actions:                      # optional
      - label: "View"
        action: view_item
        icon: eye
        condition: "is_available"  # optional
  
  empty_state:                    # optional
    icon: list
    title: "No items"
    message: "No items to display"
```

#### Avatar Configuration

Avatars can display images or fallback initials:
- **Image**: From data field
- **Fallback**: Text/emoji when no image available
- **Sizes**: `sm` (32px), `md` (40px), `lg` (48px)

#### Metadata

Metadata items display supplementary information:
- Support field values or templates
- Icons for visual categorization
- Formatting for timestamps

#### Example

```namel3ss
show data_list "Recent Activities" from dataset user_activities:
  item:
    avatar:
      field: user_avatar
      fallback: "U"
      size: md
    
    title:
      text: "{{ user_name }} {{ action_type }}"
    
    subtitle:
      field: description
    
    metadata:
      - field: created_at
        icon: clock
        format: relative
      - field: location
        icon: map-pin
    
    badge:
      field: priority
      style: priority_badge
    
    actions:
      - label: "View Details"
        action: view_activity
        icon: eye
```

---

### ShowStatSummary

KPI cards with metrics, trends, and sparkline charts.

#### Syntax

```namel3ss
show stat_summary from dataset dataset_name:
  title: "Card Title"             # optional
  label: "Metric Label"           # required
  
  value:                          # required
    field: value_field            # either field or text
    text: "{{ template }}"
    format: currency              # currency, percentage, number, compact
  
  delta:                          # optional - change indicator
    field: delta_field
    format: currency              # format for delta value
    show_sign: true               # show +/- prefix
  
  trend:                          # optional - trend direction
    field: trend_field            # 1 = up, -1 = down, 0 = neutral
    up_is_good: true              # whether up trend is positive
  
  sparkline:                      # optional - mini chart
    data: history_field           # array of {value: number}
    color: "#3b82f6"
    height: "40px"
  
  comparison: "vs last month"     # optional - comparison text
```

#### Value Formats

- `currency`: $1,234.56
- `percentage`: 45.2%
- `number`: 1,234,567
- `compact`: 1.2M, 45.3K

#### Trend Indicators

Trend direction visualizes performance:
- **Up** (↑): Positive change
- **Down** (↓): Negative change
- **Neutral** (→): No change
- Color indicates good/bad based on `up_is_good`

#### Sparklines

Mini line charts showing historical data:
- Requires array of objects: `[{value: 100}, {value: 120}, ...]`
- Rendered with Recharts
- Configurable color and height

#### Example

```namel3ss
show stat_summary from dataset revenue_stats:
  title: "Revenue"
  label: "Monthly Revenue"
  
  value:
    field: total_revenue
    format: currency
  
  delta:
    field: revenue_change
    format: percentage
    show_sign: true
  
  trend:
    field: trend_direction
    up_is_good: true
  
  sparkline:
    data: daily_revenue
    color: "#10b981"
    height: "40px"
  
  comparison: "vs previous month"
```

---

### ShowTimeline

Chronological event displays with icons and grouping.

#### Syntax

```namel3ss
show timeline "Title" from dataset dataset_name:
  group_by_date: true             # optional, groups by date
  
  items:
    - timestamp: timestamp_field  # required
      icon: check                 # optional
      status: success             # optional: success, error, warning, info
      
      title:                      # required
        field: title_field
        text: "{{ template }}"
      
      description:                # optional
        field: description_field
        text: "{{ template }}"
  
  empty_state:                    # optional
    icon: clock
    title: "No events"
    message: "No timeline events"
```

#### Status Colors

Status determines the timeline marker color:
- `success`: Green (#10b981)
- `error`: Red (#ef4444)
- `warning`: Orange (#f59e0b)
- `info`: Blue (#3b82f6)
- Default: Gray (#6b7280)

#### Date Grouping

When `group_by_date: true`, events are grouped under date headers:
- Automatic grouping by calendar date
- Date headers formatted as "Month Day, Year"
- Events within each date chronologically ordered

#### Example

```namel3ss
show timeline "Order History" from dataset order_events:
  group_by_date: true
  
  items:
    - timestamp: event_time
      icon: "📦"
      status: success
      
      title:
        text: "{{ event_type | humanize }}"
      
      description:
        text: "Order #{{ order_number }}: {{ details }}"
  
  empty_state:
    icon: clock
    title: "No activity"
    message: "No recent order activity"
```

---

### ShowAvatarGroup

User/entity displays with status indicators and overflow handling.

#### Syntax

```namel3ss
show avatar_group "Title" from dataset dataset_name:
  items:
    - image_field: avatar_url     # optional
      name_field: full_name       # required
      status_field: online_status # optional
      tooltip_template: "{{ name }} - {{ role }}"  # optional
  
  max_visible: 5                  # optional, default: 5
  size: sm | md | lg              # optional, default: md
  show_status: true               # optional, shows status indicator
```

#### Status Indicators

Small colored dots indicating user status:
- `online`: Green
- `offline`: Gray
- `busy`: Red
- `away`: Orange

#### Overflow Handling

When more items than `max_visible`:
- Shows first N avatars
- Displays "+N more" overflow indicator
- Tooltip on overflow shows count

#### Example

```namel3ss
show avatar_group "Team Members" from dataset team:
  items:
    - image_field: avatar_url
      name_field: full_name
      status_field: status
      tooltip_template: "{{ full_name }} - {{ role }}"
  
  max_visible: 8
  size: md
  show_status: true
```

---

### ShowDataChart

Multi-series charts with Recharts integration.

#### Syntax

```namel3ss
show data_chart "Title" from dataset dataset_name:
  chart:
    type: line | bar | pie | area | scatter  # required
    x_axis: field_name            # required for non-pie charts
    y_axis: field_name            # optional
    legend: true                  # optional, default: true
    grid: true                    # optional, shows grid lines
    height: "400px"               # optional, default: 300px
    
    series:                       # required
      - data_key: value_field
        label: "Series Label"
        color: "#3b82f6"          # optional
        type: line | bar | area   # optional, for mixed charts
  
  empty_state:                    # optional
    icon: chart
    title: "No data"
    message: "No chart data available"
```

#### Chart Types

**Line Chart**: Continuous data trends
```namel3ss
chart:
  type: line
  x_axis: date
  series:
    - data_key: revenue
      label: "Revenue"
      color: "#10b981"
```

**Bar Chart**: Categorical comparisons
```namel3ss
chart:
  type: bar
  x_axis: category
  series:
    - data_key: value
      label: "Value"
```

**Pie Chart**: Part-to-whole relationships
```namel3ss
chart:
  type: pie
  series:
    - data_key: percentage
      label: "Share"
```

**Area Chart**: Cumulative trends
```namel3ss
chart:
  type: area
  x_axis: date
  series:
    - data_key: total
      label: "Total"
```

**Scatter Chart**: Correlation analysis
```namel3ss
chart:
  type: scatter
  x_axis: metric1
  series:
    - data_key: metric2
      label: "Correlation"
```

#### Multi-Series Charts

Display multiple data series on one chart:
```namel3ss
show data_chart "Performance Metrics" from dataset metrics:
  chart:
    type: line
    x_axis: date
    legend: true
    grid: true
    
    series:
      - data_key: cpu_usage
        label: "CPU %"
        color: "#ef4444"
      - data_key: memory_usage
        label: "Memory %"
        color: "#f59e0b"
      - data_key: disk_usage
        label: "Disk %"
        color: "#10b981"
```

#### Example

```namel3ss
show data_chart "Revenue Trend" from dataset sales:
  chart:
    type: line
    x_axis: month
    legend: true
    grid: true
    height: "400px"
    
    series:
      - data_key: revenue
        label: "Revenue"
        color: "#3b82f6"
      - data_key: target
        label: "Target"
        color: "#10b981"
  
  empty_state:
    icon: chart
    title: "No sales data"
    message: "No revenue data for this period"
```

---

## Common Features

### Empty States

All components support empty state configuration:

```namel3ss
empty_state:
  icon: icon_name          # emoji or icon name
  title: "No Data"         # required
  message: "Description"   # optional
  action_label: "Action"   # optional
  action_link: "/path"     # optional
```

Empty states are displayed when:
- Dataset returns no rows
- Query returns empty result
- Data array is empty

### Templates

Use template syntax for dynamic text:

```namel3ss
text: "{{ field_name }}"
text: "Order #{{ order_id }} by {{ customer }}"
```

Templates support:
- Field interpolation: `{{ field }}`
- Transformations: `{{ status | humanize }}`
- Multiple fields in one template

### Transformations

Apply transformations to field values:

- `humanize`: Converts `snake_case` to `Title Case`
- `relative`: Formats timestamps relatively ("2 hours ago")
- `truncate: N`: Truncates text to N characters
- `format: "pattern"`: Custom date/number formatting

### Conditional Logic

Use conditions to show/hide elements:

```namel3ss
condition: "status == 'active'"
condition: "count > 0"
condition: "is_visible == true"
```

Conditions support:
- Equality: `==`, `!=`
- Comparison: `>`, `<`, `>=`, `<=`
- Boolean: `&&`, `||`, `!`
- Field references from row data

---

## Integration Patterns

### With Datasets

Components bind directly to datasets:

```namel3ss
dataset orders:
  sql: |
    SELECT * FROM orders
    WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'

show data_table "Orders" from dataset orders:
  columns:
    - field: id
      header: "ID"
```

### With Queries

Components can use inline queries:

```namel3ss
show data_list "Recent Users" from query:
  sql: |
    SELECT id, name, email
    FROM users
    ORDER BY created_at DESC
    LIMIT 10
  
  item:
    title:
      field: name
```

### Multiple Components

Combine components for rich dashboards:

```namel3ss
page dashboard:
  path: "/dashboard"
  title: "Dashboard"
  
  # KPI Cards
  show stat_summary from dataset revenue:
    label: "Revenue"
    value:
      field: total
      format: currency
  
  # Chart
  show data_chart "Trend" from dataset sales:
    chart:
      type: line
      x_axis: date
      series:
        - data_key: amount
          label: "Sales"
  
  # Detailed Table
  show data_table "Orders" from dataset orders:
    columns:
      - field: order_number
        header: "Order"
```

### With Actions

Define custom actions for user interaction:

```namel3ss
show data_table "Items" from dataset items:
  columns:
    - field: name
      header: "Name"
  
  row_actions:
    - label: "Edit"
      action: edit_item      # Custom action handler
      icon: edit
    - label: "Delete"
      action: delete_item
      icon: trash
      condition: "can_delete == true"
```

Actions trigger backend endpoints or frontend handlers.

---

## Examples

### Analytics Dashboard

Complete analytics dashboard with all components:

```namel3ss
page analytics:
  path: "/analytics"
  title: "Analytics Dashboard"
  
  # KPI Row
  show stat_summary from dataset revenue:
    label: "Revenue"
    value:
      field: total
      format: currency
    delta:
      field: change
      format: percentage
    sparkline:
      data: history
      color: "#10b981"
  
  show stat_summary from dataset customers:
    label: "Customers"
    value:
      field: count
      format: number
    delta:
      field: growth
      format: number
  
  # Charts
  show data_chart "Sales Trend" from dataset sales:
    chart:
      type: line
      x_axis: date
      series:
        - data_key: revenue
          label: "Revenue"
        - data_key: orders
          label: "Orders"
  
  show data_chart "By Category" from dataset categories:
    chart:
      type: pie
      series:
        - data_key: amount
          label: "Amount"
  
  # Detailed Table
  show data_table "Recent Orders" from dataset orders:
    columns:
      - field: order_number
        header: "Order #"
      - field: customer
        header: "Customer"
      - field: amount
        header: "Amount"
        format: currency
    
    toolbar:
      searchable: true
      search_fields: ["order_number", "customer"]
    
    row_actions:
      - label: "View"
        action: view_order
        icon: eye
  
  # Activity Feed
  show data_list "Recent Activity" from dataset activity:
    item:
      avatar:
        field: user_avatar
        size: md
      title:
        text: "{{ user }} {{ action }}"
      metadata:
        - field: timestamp
          format: relative
  
  # Timeline
  show timeline "Order Events" from dataset events:
    group_by_date: true
    items:
      - timestamp: event_time
        icon: "📦"
        title:
          field: event_type
        description:
          field: details
  
  # Team
  show avatar_group "Team Online" from dataset team:
    items:
      - image_field: avatar
        name_field: name
        status_field: status
    show_status: true
    max_visible: 8
```

### Order Management

Order management interface:

```namel3ss
page orders:
  path: "/orders"
  title: "Order Management"
  
  show data_table "All Orders" from dataset orders:
    columns:
      - field: order_number
        header: "Order #"
        sortable: true
      - field: customer_name
        header: "Customer"
        sortable: true
      - field: amount
        header: "Amount"
        format: currency
        sortable: true
      - field: status
        header: "Status"
        sortable: true
      - field: created_at
        header: "Date"
        format: datetime
        sortable: true
    
    toolbar:
      searchable: true
      search_fields: ["order_number", "customer_name"]
      
      filters:
        - field: status
          label: "Status"
          type: select
          options:
            - value: pending
              label: "Pending"
            - value: processing
              label: "Processing"
            - value: completed
              label: "Completed"
      
      bulk_actions:
        - label: "Export"
          action: export_orders
          icon: download
        - label: "Process Selected"
          action: bulk_process
          icon: play
    
    row_actions:
      - label: "View"
        action: view_order
        icon: eye
      - label: "Process"
        action: process_order
        icon: check
        condition: "status == 'pending'"
      - label: "Cancel"
        action: cancel_order
        icon: x
        condition: "status != 'cancelled'"
    
    rows_per_page: 25
```

---

## Best Practices

### Performance

1. **Limit Result Sets**: Use `LIMIT` in SQL queries
2. **Index Database**: Ensure proper indexing for sortable columns
3. **Pagination**: Use appropriate `rows_per_page` values
4. **Caching**: Cache expensive queries with `refresh_interval`

### User Experience

1. **Empty States**: Always provide meaningful empty states
2. **Loading States**: Indicate when data is loading
3. **Error Handling**: Show clear error messages
4. **Responsive Design**: Components adapt to screen sizes

### Data Quality

1. **Validation**: Validate data before display
2. **Null Handling**: Handle null/undefined values gracefully
3. **Formatting**: Use appropriate formats for data types
4. **Consistency**: Maintain consistent formatting across components

### Accessibility

1. **Labels**: Provide clear labels and headers
2. **Keyboard Nav**: Support keyboard navigation
3. **ARIA**: Components include ARIA attributes
4. **Color**: Don't rely solely on color for information

---

## Migration Guide

### From ShowTable

Old syntax:
```namel3ss
show table from dataset orders:
  columns:
    - id
    - name
```

New syntax:
```namel3ss
show data_table "Orders" from dataset orders:
  columns:
    - field: id
      header: "ID"
    - field: name
      header: "Name"
```

### From ShowCard/ShowList

Card and List components remain, but new DataList provides richer metadata:

```namel3ss
# Old: ShowList
show list "Items" from dataset items:
  columns:
    - field: name

# New: ShowDataList
show data_list "Items" from dataset items:
  item:
    title:
      field: name
    metadata:
      - field: created_at
        format: relative
```

---

## Troubleshooting

### Component Not Rendering

**Problem**: Component doesn't appear  
**Solution**: 
- Verify dataset returns data
- Check syntax for required fields
- Ensure proper indentation

### Data Not Displaying

**Problem**: Data loads but doesn't display  
**Solution**:
- Verify field names match database columns
- Check data types and formats
- Review browser console for errors

### Actions Not Working

**Problem**: Click actions don't fire  
**Solution**:
- Verify action names are defined
- Check condition syntax
- Ensure action handlers are registered

### Performance Issues

**Problem**: Slow rendering or interactions  
**Solution**:
- Reduce `rows_per_page`
- Add database indexes
- Optimize SQL queries
- Enable caching

---

## Reference

### Component Summary

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| ShowDataTable | Tabular data | Sorting, filtering, pagination, actions |
| ShowDataList | Activity feeds | Avatars, metadata, badges, actions |
| ShowStatSummary | KPI metrics | Delta, trend, sparkline, comparison |
| ShowTimeline | Events/history | Chronological, icons, grouping |
| ShowAvatarGroup | User display | Status, overflow, tooltips |
| ShowDataChart | Visualization | Multi-series, 5 chart types, legend |

### Supported Formats

- **Currency**: USD with cents ($1,234.56)
- **Percentage**: With decimal (45.2%)
- **Number**: Thousands separator (1,234,567)
- **Compact**: K/M notation (1.2M, 45K)
- **Date**: Localized date format
- **Datetime**: Localized date and time
- **Relative**: Time ago ("2 hours ago")

### Chart Types

- **line**: Continuous trends
- **bar**: Categorical comparison
- **pie**: Part-to-whole
- **area**: Cumulative trends
- **scatter**: Correlation

---

## Support

For issues, questions, or feature requests:
- GitHub Issues: [namel3ss-programming-language/issues](https://github.com/namel3ss-Ai/namel3ss-programming-language/issues)
- Documentation: [docs/](../docs/)
- Examples: [examples/data-display-dashboard.ai](../examples/data-display-dashboard.ai)

---

**Document Version**: 1.0  
**Component Version**: 0.7.0  
**Last Updated**: November 26, 2025
