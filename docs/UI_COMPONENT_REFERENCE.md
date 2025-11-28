# UI Component Reference

**Complete catalog of supported components in Namel3ss**

This document lists ALL components available in Namel3ss 0.6.1. If a component is not listed here, it is not supported.

---

## Table of Contents

1. [Basic Display Components](#basic-display-components)
2. [Data Display Components](#data-display-components)
3. [Form Components](#form-components)
4. [Layout Components](#layout-components)
5. [Navigation & Chrome Components](#navigation--chrome-components)
6. [Feedback Components](#feedback-components)
7. [AI Semantic Components](#ai-semantic-components)
8. [Unsupported Components](#unsupported-components)

---

## Basic Display Components

### `show text`

Display formatted text content.

**Syntax:**
```namel3ss
page "Example" at "/":
    show text "Hello, world!"
    show text "**Bold** and *italic*" style { color: "blue" }
```

**Properties:**
- `text`: String content (supports markdown)
- `style`: Optional CSS properties

**Use Cases:**
- Headers and titles
- Descriptions and paragraphs
- Static content display

---

### `show table`

Display tabular data from a dataset or frame.

**Syntax:**
```namel3ss
page "Users" at "/users":
    show table "User List":
        from users
        columns: ["name", "email", "created_at"]
        sort by: "created_at desc"
```

**Properties:**
- `title`: Table heading
- `from`: Data source (dataset or frame)
- `columns`: Column names to display
- `filter_by`: Filter expression
- `sort_by`: Sort expression
- `style`: Optional CSS properties
- `binding`: Data binding configuration

**Use Cases:**
- Basic data tables
- Simple lists with columns
- Read-only data display

---

### `show chart`

Display data visualizations.

**Syntax:**
```namel3ss
page "Analytics" at "/analytics":
    show chart "Sales Trends":
        from sales_data
        type: "line"
        x: "month"
        y: "revenue"
```

**Properties:**
- `heading`: Chart title
- `from`: Data source
- `chart_type`: "bar", "line", "area", "pie", "scatter"
- `x`: X-axis field
- `y`: Y-axis field
- `color`: Color field for series
- `encodings`: Advanced Recharts configuration
- `binding`: Data binding configuration

**Use Cases:**
- Basic charts and graphs
- Simple visualizations

**Note:** For advanced multi-series charts, use `show data_chart` instead.

---

## Data Display Components

### `show card`

Display items in a card layout with sections, badges, and actions.

**Syntax:**
```namel3ss
page "Projects" at "/projects":
    show card "Project Cards":
        from projects
        header:
            title: "{{name}}"
            badges:
                - text: "{{status}}"
                  style: "badge-{{status}}"
        sections:
            - type: "info_grid"
              items:
                  - label: "Owner"
                    value: "{{owner}}"
                  - label: "Due Date"
                    value: "{{due_date}}"
        actions:
            - label: "View Details"
              link: "/projects/{{id}}"
```

**Properties:**
- `title`: Card list title
- `from`: Data source
- `header`: Card header configuration (title, subtitle, badges)
- `sections`: Array of card sections (info_grid, text_section, etc.)
- `actions`: Action buttons
- `footer`: Footer content
- `empty_state`: Message when no data

**Use Cases:**
- Project/item cards
- Rich content display
- Multi-section layouts

---

### `show list`

Generic list display that delegates to card layout.

**Syntax:**
```namel3ss
page "Items" at "/items":
    show list "Item List":
        from items
        item_config:
            # Card configuration
```

**Properties:**
- Same as `show card`
- `list_type`: Optional semantic type
- `columns`: Number of columns for grid layout

**Use Cases:**
- Simple card lists
- Grid layouts

**Note:** Currently delegates to `CardWidget`. Use `show card` for production.

---

### `show data_table`

Production-grade data table with toolbar, search, filters, sorting, and row actions.

**Syntax:**
```namel3ss
page "Jobs" at "/jobs":
    show data_table "File Processing Jobs":
        from FileProcessingJob
        columns:
            - field: "id"
              header: "Job ID"
              width: 100
            - field: "filename"
              header: "File"
              sortable: true
            - field: "status"
              header: "Status"
              render:
                  type: "badge"
                  style_map:
                      completed: "success"
                      failed: "error"
                      pending: "warning"
        toolbar:
            search:
                enabled: true
                placeholder: "Search jobs..."
            actions:
                - label: "New Job"
                  action: "create_job"
        row_actions:
            - label: "View Details"
              link: "/jobs/{{id}}"
            - label: "Retry"
              action: "retry_job"
              condition: "{{status == 'failed'}}"
        pagination:
            page_size: 25
            enabled: true
```

**Properties:**
- `title`: Table title
- `from`: Data source
- `columns`: Column definitions with renderers
- `toolbar`: Search, filters, and actions
- `row_actions`: Per-row action buttons
- `pagination`: Pagination configuration
- `empty_state`: Message when no data
- `enable_export`: Export to CSV/JSON

**Use Cases:**
- Admin interfaces
- Job management dashboards
- Data management UIs
- Any production table needs

---

### `show data_list`

Activity feeds and item lists with avatars, metadata, and badges.

**Syntax:**
```namel3ss
page "Activity" at "/activity":
    show data_list "Recent Activity":
        from activity_feed
        item:
            avatar:
                field: "user_avatar"
                fallback: "{{user_initials}}"
            title: "{{user_name}}"
            subtitle: "{{action_description}}"
            metadata:
                - field: "timestamp"
                  format: "relative"
                  icon: "clock"
                - field: "category"
                  icon: "tag"
            badge:
                field: "priority"
                style: "badge-{{priority}}"
        pagination:
            page_size: 20
```

**Properties:**
- `title`: List title
- `from`: Data source
- `item`: Item configuration (avatar, title, subtitle, metadata, badge)
- `actions`: Item action buttons
- `pagination`: Pagination settings
- `empty_state`: Message when no data

**Use Cases:**
- Activity feeds
- Notification lists
- User/entity lists
- Timeline views

---

### `show stat_summary`

Display key metrics and statistics in a grid.

**Syntax:**
```namel3ss
page "Dashboard" at "/":
    show stat_summary "Key Metrics":
        stats:
            - label: "Total Users"
              value_binding: "metrics.total_users"
              change: "+12%"
              trend: "up"
              icon: "users"
            - label: "Revenue"
              value_binding: "metrics.revenue"
              format: "currency"
              change: "+8.2%"
              trend: "up"
              icon: "dollar-sign"
```

**Properties:**
- `title`: Summary title
- `stats`: Array of stat definitions
  - `label`: Stat label
  - `value_binding`: Data binding
  - `format`: Format type (number, currency, percentage, etc.)
  - `change`: Change indicator (e.g., "+12%")
  - `trend`: "up", "down", "neutral"
  - `icon`: Optional icon
- `layout`: "grid" or "horizontal"

**Use Cases:**
- Dashboard KPIs
- Metric summaries
- Analytics overviews

---

### `show timeline`

Display chronological events with timestamps and descriptions.

**Syntax:**
```namel3ss
page "History" at "/history":
    show timeline "Project History":
        from project_events
        event:
            timestamp_field: "created_at"
            title_field: "event_title"
            description_field: "event_description"
            icon_field: "event_type"
            actor_field: "user_name"
        group_by: "date"
        show_relative_times: true
```

**Properties:**
- `title`: Timeline title
- `from`: Data source
- `event`: Event configuration
- `group_by`: Grouping strategy ("date", "month", "none")
- `show_relative_times`: Show "2 hours ago" vs absolute times
- `empty_state`: Message when no events

**Use Cases:**
- Activity timelines
- Event histories
- Audit logs
- Change tracking

---

### `show avatar_group`

Display groups of user avatars with status indicators.

**Syntax:**
```namel3ss
page "Team" at "/team":
    show avatar_group "Project Team":
        from team_members
        items:
            avatar_field: "profile_image"
            name_field: "name"
            status_field: "online_status"
        max_visible: 5
        size: "md"
        show_status: true
```

**Properties:**
- `title`: Group title
- `from`: Data source
- `items`: Avatar item configuration
- `max_visible`: Max avatars before "+N more"
- `size`: "sm", "md", "lg"
- `show_status`: Show online/offline indicators

**Use Cases:**
- Team member displays
- Participant lists
- Collaborator views

---

### `show data_chart`

Advanced multi-series charts with full Recharts configuration.

**Syntax:**
```namel3ss
page "Analytics" at "/analytics":
    show data_chart "Revenue Breakdown":
        from revenue_data
        chart_type: "line"
        series:
            - data_key: "revenue"
              label: "Revenue"
              color: "#3b82f6"
              type: "line"
            - data_key: "expenses"
              label: "Expenses"
              color: "#ef4444"
              type: "line"
        x_axis:
            data_key: "month"
            label: "Month"
        y_axis:
            label: "Amount ($)"
        legend: true
        tooltip: true
```

**Properties:**
- `title`: Chart title
- `from`: Data source
- `chart_type`: "line", "bar", "area", "pie", "scatter"
- `series`: Array of series configurations
- `x_axis`: X-axis configuration
- `y_axis`: Y-axis configuration
- `legend`: Show legend
- `tooltip`: Show tooltips

**Use Cases:**
- Multi-series charts
- Complex visualizations
- Analytics dashboards

---

## Form Components

### `show form`

Create forms with validation and submission handling.

**Syntax:**
```namel3ss
page "Create User" at "/users/new":
    show form "New User":
        fields:
            - name: "username"
              component: "text_input"
              label: "Username"
              required: true
              min_length: 3
            - name: "email"
              component: "email_input"
              label: "Email"
              required: true
            - name: "role"
              component: "select"
              label: "Role"
              options: ["admin", "user", "guest"]
            - name: "avatar"
              component: "file_input"
              label: "Profile Picture"
              accept: "image/*"
        on submit:
            call create_user with {
                username: username,
                email: email,
                role: role,
                avatar: avatar
            }
            go to page "/users"
```

**Field Components:**
- `text_input`: Single-line text
- `email_input`: Email with validation
- `password_input`: Password field
- `number_input`: Numeric input
- `textarea`: Multi-line text
- `select`: Dropdown selection
- `multiselect`: Multiple selection
- `checkbox`: Boolean checkbox
- `radio`: Radio button group
- `date_input`: Date picker
- `time_input`: Time picker
- `file_input`: File upload
- `slider`: Range slider
- `switch`: Toggle switch

**Validation:**
- `required`: Field is required
- `min_length`, `max_length`: String length
- `pattern`: Regex validation
- `min_value`, `max_value`: Numeric bounds
- `custom_validator`: Python function

**Use Cases:**
- User input forms
- Create/edit interfaces
- Multi-step wizards
- File uploads

**See:** [FORMS_REFERENCE.md](./FORMS_REFERENCE.md) for complete form documentation.

---

## Layout Components

### `stack`

Stack children vertically or horizontally.

**Syntax:**
```namel3ss
page "Layout" at "/layout":
    stack direction="column" gap="medium":
        show text "First"
        show text "Second"
        show text "Third"
```

**Properties:**
- `direction`: "row" or "column"
- `gap`: "small", "medium", "large", or CSS value
- `align`: "start", "center", "end", "stretch"
- `justify`: "start", "center", "end", "space-between"
- `wrap`: Enable wrapping

---

### `grid`

Create responsive grid layouts.

**Syntax:**
```namel3ss
page "Grid" at "/grid":
    grid columns=3 gap="large":
        show card "Card 1": ...
        show card "Card 2": ...
        show card "Card 3": ...
```

**Properties:**
- `columns`: Number of columns
- `min_column_width`: Minimum column width (responsive)
- `gap`: Gap size
- `responsive`: Auto-adjust columns

---

### `split`

Split view with resizable panes.

**Syntax:**
```namel3ss
page "Split" at "/split":
    split direction="horizontal" ratio=30:
        # Left pane (30%)
        show list "Items": ...
        # Right pane (70%)
        show text "Details": ...
```

**Properties:**
- `direction`: "horizontal" or "vertical"
- `ratio`: Split ratio (percentage)
- `resizable`: Allow user resizing

---

### `tabs`

Tabbed content areas.

**Syntax:**
```namel3ss
page "Tabs" at "/tabs":
    tabs:
        tab "Overview":
            show text "Overview content"
        tab "Details":
            show text "Detail content"
        tab "Settings":
            show text "Settings content"
```

**Properties:**
- Tab labels
- Tab content
- Default tab selection

---

### `accordion`

Collapsible accordion sections.

**Syntax:**
```namel3ss
page "FAQ" at "/faq":
    accordion:
        section "Question 1":
            show text "Answer 1"
        section "Question 2":
            show text "Answer 2"
```

**Properties:**
- Section titles
- Section content
- Allow multiple open sections

---

## Navigation & Chrome Components

### `sidebar`

Application sidebar with navigation.

**Syntax:**
```namel3ss
page "Home" at "/":
    sidebar:
        logo: "/logo.svg"
        title: "My App"
        nav:
            - label: "Dashboard"
              link: "/"
              icon: "home"
            - label: "Users"
              link: "/users"
              icon: "users"
            - label: "Settings"
              link: "/settings"
              icon: "settings"
```

**Properties:**
- `logo`: Logo image URL
- `title`: Application title
- `nav`: Navigation items
- `position`: "left" or "right"
- `collapsible`: Allow collapse

---

### `navbar`

Top navigation bar.

**Syntax:**
```namel3ss
page "Home" at "/":
    navbar:
        title: "My App"
        actions:
            - label: "Profile"
              link: "/profile"
            - label: "Logout"
              action: "logout"
```

**Properties:**
- `title`: Navbar title
- `actions`: Action buttons
- `show_breadcrumbs`: Show breadcrumb trail

---

### `breadcrumbs`

Breadcrumb navigation trail.

**Syntax:**
```namel3ss
page "User Detail" at "/users/123":
    breadcrumbs:
        - label: "Home"
          link: "/"
        - label: "Users"
          link: "/users"
        - label: "User 123"
```

**Properties:**
- Breadcrumb items (label, link)

---

### `command_palette`

Keyboard-driven command palette (Cmd+K style).

**Syntax:**
```namel3ss
page "Home" at "/":
    command_palette:
        sources:
            - type: "pages"
              label: "Pages"
            - type: "actions"
              label: "Actions"
        shortcut: "cmd+k"
```

**Properties:**
- `sources`: Command sources
- `shortcut`: Keyboard shortcut

---

## Feedback Components

### `modal`

Modal dialogs.

**Syntax:**
```namel3ss
page "Home" at "/":
    modal id="confirm_delete":
        title: "Confirm Deletion"
        description: "Are you sure you want to delete this item?"
        content:
            show text "This action cannot be undone."
        actions:
            - label: "Cancel"
              action: "close_modal"
            - label: "Delete"
              action: "delete_item"
              style: "danger"
```

**Properties:**
- `id`: Modal identifier
- `title`: Modal title
- `description`: Optional description
- `content`: Modal body content
- `actions`: Action buttons
- `size`: "sm", "md", "lg", "xl"

---

### `toast`

Toast notifications.

**Syntax:**
```namel3ss
on submit:
    # ... form logic
    show toast "User created successfully" type="success"
```

**Properties:**
- `message`: Toast message
- `type`: "success", "error", "warning", "info"
- `duration`: Auto-dismiss duration (ms)
- `position`: "top", "top-right", "bottom", "bottom-right"

---

## AI Semantic Components

### `chat_thread`

Multi-message AI conversation display.

**Syntax:**
```namel3ss
page "Chat" at "/chat":
    chat_thread:
        messages_binding: "conversation.messages"
        group_by: "role"
        show_timestamps: true
        show_avatar: true
        streaming_enabled: true
```

**Properties:**
- `messages_binding`: Binding to message array
- `group_by`: "role", "speaker", "timestamp", "none"
- `show_timestamps`: Show message times
- `show_avatar`: Show user avatars
- `streaming_enabled`: Support streaming responses
- `max_height`: Constrain height

**Use Cases:**
- AI chat interfaces
- Conversation histories
- Agent interactions

---

### `agent_panel`

Display agent state, metrics, and status.

**Syntax:**
```namel3ss
page "Agent" at "/agent":
    agent_panel:
        agent_binding: "current_agent"
        metrics_binding: "run.metrics"
        show_status: true
        show_metrics: true
        show_tokens: true
        show_cost: true
```

**Properties:**
- `agent_binding`: Binding to agent
- `metrics_binding`: Binding to metrics
- `show_status`: Show agent status
- `show_metrics`: Show performance metrics
- `show_tokens`: Show token usage
- `show_cost`: Show cost estimates
- `show_latency`: Show response time

**Use Cases:**
- Agent monitoring
- Performance tracking
- Cost analysis

---

### `tool_call_view`

Display tool invocations and results.

**Syntax:**
```namel3ss
page "Tools" at "/tools":
    tool_call_view:
        calls_binding: "run.tool_calls"
        show_inputs: true
        show_outputs: true
        show_timing: true
        expandable: true
```

**Properties:**
- `calls_binding`: Binding to tool call logs
- `show_inputs`: Show input parameters
- `show_outputs`: Show output results
- `show_timing`: Show execution time
- `show_status`: Show success/failure
- `expandable`: Collapsible details

**Use Cases:**
- Debugging tool calls
- Monitoring integrations
- Understanding agent behavior

---

### `log_view`

Tail and inspect logs and traces.

**Syntax:**
```namel3ss
page "Logs" at "/logs":
    log_view:
        logs_binding: "run.logs"
        level_filter: ["info", "warn", "error"]
        max_height: "600px"
        virtualized: true
```

**Properties:**
- `logs_binding`: Binding to log entries
- `level_filter`: Filter by log level
- `max_height`: Constrain height
- `virtualized`: Use virtualization for performance
- `enable_copy`: Copy log entries
- `enable_download`: Download logs

**Use Cases:**
- Runtime debugging
- Error investigation
- Performance analysis

---

### `evaluation_result`

Display evaluation metrics, distributions, and error analysis.

**Syntax:**
```namel3ss
page "Evals" at "/evals":
    evaluation_result:
        eval_run_binding: "eval.run_123"
        show_summary: true
        show_histograms: true
        show_error_table: true
```

**Properties:**
- `eval_run_binding`: Binding to eval run
- `show_summary`: Show aggregate metrics
- `show_histograms`: Show score distributions
- `show_error_table`: Show per-example errors
- `metrics_to_show`: Filter metrics
- `comparison_run_binding`: Compare runs

**Use Cases:**
- Model evaluation
- Quality assurance
- Performance comparison

---

### `diff_view`

Compare model outputs, prompts, or documents.

**Syntax:**
```namel3ss
page "Diff" at "/diff":
    diff_view:
        left_binding: "version.v1.output"
        right_binding: "version.v2.output"
        mode: "split"
        content_type: "code"
        language: "python"
        show_line_numbers: true
```

**Properties:**
- `left_binding`: "Before" content
- `right_binding`: "After" content
- `mode`: "unified" or "split"
- `content_type`: "text", "code", "markdown"
- `language`: Syntax highlighting language
- `ignore_whitespace`: Ignore whitespace changes

**Use Cases:**
- Model output comparison
- Prompt versioning
- Document changes

---

## Unsupported Components

The following components are **NOT supported** in Namel3ss. Each has clear alternatives with complete examples showing how to achieve the same functionality.

---

### ❌ `progress_bar` / `progress`

**Why Not Supported:** Progress bars require complex real-time updates and styling state management. The alternatives provide the same information in more accessible, dashboard-friendly formats.

**Alternatives:**

#### 1. `show stat_summary` (Recommended)
**Best for:** Dashboard KPIs, job completion status, metrics

Display progress as a KPI stat with percentage formatting:

```namel3ss
show stat_summary "Job Progress":
    stats:
        - label: "Completion"
          value_binding: "job.progress_pct"
          format: "percentage"
        - label: "Status"
          value_binding: "job.status"
          icon: "check-circle"
        - label: "Files Processed"
          value_binding: "job.files_done"
          format: "number"
```

#### 2. `show data_chart`
**Best for:** Progress history, trends, multiple metrics over time

Visualize progress over time with a chart:

```namel3ss
show data_chart "Progress Over Time" from dataset progress_history:
    chart_type: "line"
    series:
        - data_key: "completion_pct"
          label: "Completion"
          color: "#10b981"
    x_axis:
        data_key: "timestamp"
        format: "datetime"
```

#### 3. `show text`
**Best for:** Simple inline status, text-based updates

Show progress as formatted text:

```namel3ss
show text "Progress: {{job.progress_pct}}% Complete" style {
    color: "green"
    font-weight: "bold"
    font-size: "1.2rem"
}

# Or with conditional formatting:
show text "Status: {{job.status | upper}}"
```

📚 **Documentation:** [DATA_DISPLAY_COMPONENTS.md](./DATA_DISPLAY_COMPONENTS.md), [UI_COMPONENTS_AND_STYLING.md](./UI_COMPONENTS_AND_STYLING.md)

---

### ❌ `code_block` / `code`

**Why Not Supported:** Code blocks with rich IDE features (code folding, search, copy buttons) add significant complexity. Markdown code blocks provide syntax highlighting, and diff_view handles comparisons effectively.

**Alternatives:**

#### 1. `show text` with Markdown (Recommended)
**Best for:** Displaying code snippets, static examples, documentation

Use markdown code fences for syntax highlighting:

```namel3ss
show text """
```python
def process_file(filename):
    \"\"\"Process a file and return uppercase content.\"\"\"
    with open(filename, 'r') as f:
        data = f.read()
    return data.upper()
```
"""

# Dynamic code from a binding:
show text """
```{{code_language}}
{{code_content}}
```
"""
```

#### 2. `diff_view`
**Best for:** Code reviews, version comparisons, showing changes

Compare code side-by-side with syntax highlighting:

```namel3ss
diff_view:
    left_binding: "original_code"
    right_binding: "modified_code"
    content_type: "code"
    language: "python"
    mode: "split"
    show_line_numbers: true
    highlight_changes: true
```

#### 3. `show text` with Styling
**Best for:** Terminal output, logs, plain text code, non-highlighted code

Display code in a monospace container:

```namel3ss
show text "{{code_output}}" style {
    font-family: "monospace"
    background: "#1e1e1e"
    color: "#d4d4d4"
    padding: "1rem"
    border-radius: "4px"
    white-space: "pre-wrap"
    overflow-x: "auto"
}
```

📚 **Documentation:** [FEEDBACK_COMPONENTS_GUIDE.md](./FEEDBACK_COMPONENTS_GUIDE.md), [UI_COMPONENTS_AND_STYLING.md](./UI_COMPONENTS_AND_STYLING.md)

---

### ❌ `json_view` / `json`

**Why Not Supported:** Interactive JSON trees with expand/collapse and search require complex state management. The alternatives cover 95% of use cases with simpler, more maintainable implementations.

**Alternatives:**

#### 1. `show text` with `to_json` Filter (Recommended)
**Best for:** API responses, configuration display, debugging, formatted JSON output

Format JSON data with the built-in filter:

```namel3ss
show text "{{api_response | to_json}}" style {
    font-family: "monospace"
    white-space: "pre-wrap"
    background: "#f5f5f5"
    padding: "1rem"
    border-radius: "4px"
    border: "1px solid #e5e5e5"
    overflow-x: "auto"
}

# With custom formatting:
show text "Response:\n{{data | to_json(indent=2)}}"
```

#### 2. `show data_table`
**Best for:** Tabular JSON data, lists of objects, data grids, explorable data

Display JSON data as a structured table:

```namel3ss
# Convert JSON array to dataset first
chain parse_response:
    step "convert":
        from_query: """
            SELECT 
                json_extract(value, '$.id') as id,
                json_extract(value, '$.name') as name,
                json_extract(value, '$.status') as status
            FROM json_each(@response)
        """

show data_table "API Response" from dataset response_items:
    columns:
        - field: "id"
          header: "ID"
          width: "80px"
        - field: "name"
          header: "Name"
          sortable: true
          searchable: true
        - field: "status"
          header: "Status"
          render:
              type: "badge"
              variant: "{{status == 'active' ? 'success' : 'default'}}"
```

#### 3. `show card` with `info_grid`
**Best for:** Single objects, nested data, record details, structured displays

Display JSON fields as key-value pairs:

```namel3ss
show card "Response Details":
    header:
        title: "API Response"
        subtitle: "{{response.endpoint}}"
    sections:
        - type: "info_grid"
          items:
              - label: "Status"
                value: "{{response.status}}"
                format: "badge"
              - label: "Message"
                value: "{{response.message}}"
              - label: "Timestamp"
                value: "{{response.timestamp}}"
                format: "datetime"
              - label: "Request ID"
                value: "{{response.request_id}}"
                copyable: true
        - type: "text_section"
          title: "Response Data"
          content: "{{response.data | to_json}}"
```

📚 **Documentation:** [DATA_DISPLAY_COMPONENTS.md](./DATA_DISPLAY_COMPONENTS.md), [STANDARD_LIBRARY.md](./STANDARD_LIBRARY.md), [EXPRESSION_LANGUAGE.md](./EXPRESSION_LANGUAGE.md)

---

### ❌ `tree_view` / `tree`

**Why Not Supported:** Tree views with expand/collapse/drag-drop are complex interactive widgets. Accordion provides the same hierarchical structure with simpler, more accessible semantics.

**Alternatives:**

#### 1. `accordion` (Recommended)
**Best for:** File trees, nested menus, hierarchical data, collapsible sections

Collapsible hierarchical sections:

```namel3ss
accordion:
    section "Documents":
        show text "Total: 156 files"
        
        accordion:
            section "Projects":
                show data_list "Files" from dataset project_files:
                    item:
                        title: "{{filename}}"
                        subtitle: "{{path}}"
                        metadata:
                            - field: "size"
                              format: "bytes"
                            - field: "modified"
                              format: "relative"
                
            section "Reports":
                show data_list "Reports" from dataset reports:
                    item:
                        title: "{{name}}"
                        badge:
                            field: "status"
                            style: "badge-{{status}}"
    
    section "Images":
        show data_list "Images" from dataset images:
            item:
                title: "{{filename}}"
                subtitle: "{{dimensions}}"
```

#### 2. `show data_list` with Nesting
**Best for:** Activity feeds, threaded comments, nested lists, hierarchical relationships

List items with visual hierarchy:

```namel3ss
show data_list "File System" from dataset files:
    item:
        title: "{{name}}"
        subtitle: "{{path}}"
        leading_icon: "{{type == 'folder' ? 'folder' : 'file'}}"
        metadata:
            - field: "size"
              format: "bytes"
            - field: "modified"
              format: "relative"
        badge:
            field: "type"
            style: "badge-{{type}}"
        actions:
            - label: "Open"
              on_click:
                  navigate: "/view?file={{path}}"
            - label: "Download"
              on_click:
                  download: "{{path}}"

# For nested comments/threads:
show data_list "Discussion" from dataset comments:
    item:
        title: "{{author}}"
        subtitle: "{{content}}"
        metadata:
            - field: "timestamp"
              format: "relative"
        # Indentation shows nesting level
        indent_level: "{{depth}}"
```

#### 3. `show card` with Nested Sections
**Best for:** Structured records, grouped data, categorized information

Card sections for grouped hierarchical data:

```namel3ss
show card "Project Structure":
    header:
        title: "File Organization"
        subtitle: "{{total_files}} files"
    
    sections:
        - type: "text_section"
          title: "Source Code"
          content: "Main application code"
        
        - type: "info_grid"
          items:
              - label: "Components"
                value: "12 files"
                icon: "folder"
              - label: "Tests"
                value: "45 files"
                icon: "check-circle"
        
        - type: "text_section"
          title: "Documentation"
          content: "Project documentation and guides"
        
        - type: "info_grid"
          items:
              - label: "Guides"
                value: "8 files"
              - label: "API Docs"
                value: "23 files"
```

📚 **Documentation:** [LAYOUT_PRIMITIVES.md](./LAYOUT_PRIMITIVES.md), [DATA_DISPLAY_COMPONENTS.md](./DATA_DISPLAY_COMPONENTS.md), [CHROME_COMPONENTS_GUIDE.md](./CHROME_COMPONENTS_GUIDE.md)

---

### ⚠️ Note: `tabs` Layout vs Component

**`tabs` is supported as a layout component**, but not as a `show tabs` display component.

```namel3ss
# ✅ Supported - Layout syntax:
tabs:
    tab "Overview":
        show text "Overview content"
    tab "Details":
        show table "Details" from dataset items

# ❌ NOT supported - Component syntax:
show tabs "My Tabs": ...
```

See [LAYOUT_PRIMITIVES.md](./LAYOUT_PRIMITIVES.md) for complete tabs documentation.

---

## Component Selection Guide

**When to use each component:**

| Use Case | Recommended Component |
|----------|----------------------|
| Simple text display | `show text` |
| Basic data table | `show table` |
| Production data table with features | `show data_table` |
| Simple chart | `show chart` |
| Multi-series chart | `show data_chart` |
| Card-based list | `show card` |
| Activity feed | `show data_list` |
| KPI metrics | `show stat_summary` |
| Event history | `show timeline` |
| Team members | `show avatar_group` |
| Forms | `show form` |
| Vertical/horizontal layout | `stack` |
| Grid layout | `grid` |
| Two-pane layout | `split` |
| Tabbed content | `tabs` |
| Collapsible sections | `accordion` |
| Navigation | `sidebar`, `navbar`, `breadcrumbs` |
| Quick actions | `command_palette` |
| Dialogs | `modal` |
| Notifications | `toast` |
| AI chat | `chat_thread` |
| Agent monitoring | `agent_panel` |
| Tool debugging | `tool_call_view` |
| Log inspection | `log_view` |
| Evaluation results | `evaluation_result` |
| Text comparison | `diff_view` |

---

## Next Steps

- **[FORMS_REFERENCE.md](./FORMS_REFERENCE.md)** - Complete form documentation
- **[DATA_DISPLAY_COMPONENTS.md](./DATA_DISPLAY_COMPONENTS.md)** - Deep dive into data components
- **[CHROME_COMPONENTS_GUIDE.md](./CHROME_COMPONENTS_GUIDE.md)** - Navigation and chrome patterns
- **[UI_COMPONENTS_AND_STYLING.md](./UI_COMPONENTS_AND_STYLING.md)** - Styling and theming guide

---

**Last Updated:** November 28, 2025  
**Version:** 0.6.1
