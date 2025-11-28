# Navigation Guide

**Routing, navigation, and state management patterns in Namel3ss**

This guide covers how to build multi-page applications with navigation in Namel3ss 0.6.1.

---

## Table of Contents

1. [Page Declaration and Routes](#page-declaration-and-routes)
2. [Navigation Between Pages](#navigation-between-pages)
3. [Dynamic Content Without Route Parameters](#dynamic-content-without-route-parameters)
4. [Navigation Components](#navigation-components)
5. [State Management](#state-management)
6. [Common Patterns](#common-patterns)
7. [Limitations](#limitations)

---

## Page Declaration and Routes

### Basic Page Syntax

Pages are declared with a route path:

```namel3ss
page "Dashboard" at "/":
    show text "Welcome to the dashboard!"

page "Users" at "/users":
    show text "User list"

page "Settings" at "/settings":
    show text "Application settings"
```

**Properties:**
- **Title**: Display name ("Dashboard", "Users", etc.)
- **Route**: URL path ("/", "/users", "/settings")
- **Colon syntax**: Pages use `:` not `{}`

### Route Syntax Rules

**✅ Supported:**
```namel3ss
page "Home" at "/"
page "About" at "/about"
page "Contact" at "/contact"
page "User Management" at "/users"
page "Job Details" at "/jobs"
```

**❌ NOT Supported:**
```namel3ss
page "User Detail" at "/users/:id"  # No route parameters
page "Job Detail" at "/jobs/:jobId"  # No dynamic segments
```

### Why No Route Parameters?

Namel3ss uses **static routing** for simplicity and performance:
- All routes are known at compile time
- No client-side route matching needed
- Better IDE support and validation
- Clearer application structure

For dynamic content, use **state-based approaches** (see below).

---

## Navigation Between Pages

### Using `go to page`

Navigate to pages in action handlers:

```namel3ss
page "Users" at "/users":
    show data_table "All Users" from dataset users:
        columns:
            - field: "name"
              header: "Name"
            - field: "email"
              header: "Email"
        row_actions:
            - label: "View Details"
              action: "view_user"
    
    action "view_user":
        # Store user ID in session/memory before navigating
        set session.selected_user_id to row.id
        go to page "/user-detail"

page "User Detail" at "/user-detail":
    # Load user based on session state
    show text "User: {{session.selected_user_id}}"
```

### Link Components

Use navigation in chrome components:

```namel3ss
page "Home" at "/":
    sidebar:
        title: "My App"
        nav:
            - label: "Dashboard"
              link: "/"
              icon: "home"
            - label: "Users"
              link: "/users"
              icon: "users"
            - label: "Jobs"
              link: "/jobs"
              icon: "briefcase"
    
    navbar:
        title: "Dashboard"
        actions:
            - label: "Profile"
              link: "/profile"
            - label: "Settings"
              link: "/settings"
```

### Breadcrumbs

Show navigation hierarchy:

```namel3ss
page "Job Detail" at "/job-detail":
    breadcrumbs:
        - label: "Home"
          link: "/"
        - label: "Jobs"
          link: "/jobs"
        - label: "Job {{session.selected_job_id}}"
```

---

## Dynamic Content Without Route Parameters

Since route parameters like `/users/:id` are not supported, use these patterns instead:

### Pattern 1: Session State

Store identifiers in session memory before navigating:

```namel3ss
page "Users" at "/users":
    show data_table "All Users" from dataset users:
        row_actions:
            - label: "View"
              action: "view_user"
    
    action "view_user":
        set session.selected_user_id to row.id
        go to page "/user-detail"

page "User Detail" at "/user-detail":
    show data_table "User Info" from dataset users:
        filter_by: "id = {{session.selected_user_id}}"
```

**Pros:**
- Simple and straightforward
- Works with any data type
- Persists across page reloads

**Cons:**
- Not shareable via URL
- No browser back/forward support

### Pattern 2: Global Memory

Use application-level state:

```namel3ss
memory:
    scope: global
    storage:
        current_user_id: text

page "Users" at "/users":
    action "view_user":
        set global.current_user_id to row.id
        go to page "/user-detail"

page "User Detail" at "/user-detail":
    show text "User: {{global.current_user_id}}"
```

**Pros:**
- Shared across all sessions
- Good for admin interfaces

**Cons:**
- Same limitations as session state
- Can conflict in multi-user scenarios

### Pattern 3: Query Filters

Filter data on the detail page:

```namel3ss
page "Jobs" at "/jobs":
    show data_table "All Jobs" from dataset FileProcessingJob:
        columns:
            - field: "id"
              header: "Job ID"
            - field: "filename"
              header: "File"
            - field: "status"
              header: "Status"
        row_actions:
            - label: "View Details"
              action: "view_job"
    
    action "view_job":
        set session.selected_job to row
        go to page "/job-detail"

page "Job Detail" at "/job-detail":
    # Display full job details
    show card "Job Information":
        header:
            title: "{{session.selected_job.filename}}"
            badges:
                - text: "{{session.selected_job.status}}"
                  style: "badge-{{session.selected_job.status}}"
        sections:
            - type: "info_grid"
              items:
                  - label: "Job ID"
                    value: "{{session.selected_job.id}}"
                  - label: "Created"
                    value: "{{session.selected_job.created_at}}"
                  - label: "Status"
                    value: "{{session.selected_job.status}}"
```

**Pros:**
- Full object available on detail page
- No additional database queries
- Type-safe data passing

**Cons:**
- Still not URL-shareable
- Requires storing full object in session

### Pattern 4: Modal Overlays

Instead of navigating to a new page, show details in a modal:

```namel3ss
page "Jobs" at "/jobs":
    show data_table "All Jobs" from dataset FileProcessingJob:
        row_actions:
            - label: "View Details"
              action: "show_job_modal"
    
    modal id="job_detail_modal":
        title: "Job Details"
        content:
            show card "Job Info":
                # Display selected job details
                header:
                    title: "{{session.selected_job.filename}}"
        actions:
            - label: "Close"
              action: "close_modal"
    
    action "show_job_modal":
        set session.selected_job to row
        show modal "job_detail_modal"
```

**Pros:**
- No page navigation needed
- Better UX for quick views
- Preserves list context

**Cons:**
- Not suitable for complex detail pages
- No direct linking

---

## Navigation Components

### Sidebar

Primary navigation menu:

```namel3ss
page "Home" at "/":
    sidebar:
        logo: "/logo.svg"
        title: "My Application"
        nav:
            - label: "Dashboard"
              link: "/"
              icon: "home"
            - label: "Users"
              link: "/users"
              icon: "users"
            - label: "Jobs"
              link: "/jobs"
              icon: "briefcase"
            - label: "Settings"
              link: "/settings"
              icon: "settings"
        position: "left"
        collapsible: true
```

**Properties:**
- `logo`: Logo image URL
- `title`: Application title
- `nav`: Array of navigation items
- `position`: "left" or "right"
- `collapsible`: Allow collapse

### Navbar

Top navigation bar:

```namel3ss
page "Home" at "/":
    navbar:
        title: "Dashboard"
        actions:
            - label: "Notifications"
              action: "show_notifications"
            - label: "Profile"
              link: "/profile"
            - label: "Logout"
              action: "logout"
```

**Properties:**
- `title`: Navbar title
- `actions`: Action buttons
- `show_breadcrumbs`: Include breadcrumbs

### Breadcrumbs

Navigation trail:

```namel3ss
page "Job Detail" at "/job-detail":
    breadcrumbs:
        - label: "Home"
          link: "/"
        - label: "Jobs"
          link: "/jobs"
        - label: "Job {{session.selected_job_id}}"
          # Last item is current page (no link)
```

### Command Palette

Keyboard-driven navigation:

```namel3ss
page "Home" at "/":
    command_palette:
        sources:
            - type: "pages"
              label: "Navigate to..."
            - type: "actions"
              label: "Run action..."
        shortcut: "cmd+k"
```

**Features:**
- Cmd+K (Mac) or Ctrl+K (Windows/Linux)
- Fuzzy search through pages
- Quick action execution

---

## State Management

### Session Memory

Per-user session state:

```namel3ss
memory:
    scope: session
    storage:
        selected_user_id: text
        search_query: text
        filters: object
```

**Use Cases:**
- Storing selected items
- User preferences
- Temporary state

### Global Memory

Application-wide state:

```namel3ss
memory:
    scope: global
    storage:
        total_users: int
        last_refresh: timestamp
```

**Use Cases:**
- Counters and metrics
- Cache data
- Shared configuration

### Conversation Memory

AI conversation context:

```namel3ss
memory:
    scope: conversation
    storage:
        chat_history: list
        user_context: object
```

**Use Cases:**
- Chat interfaces
- AI interactions
- Contextual data

---

## Common Patterns

### Master-Detail Pattern

List view with detail modal:

```namel3ss
page "Users" at "/users":
    show data_table "All Users" from dataset users:
        columns:
            - field: "name"
            - field: "email"
        row_actions:
            - label: "Details"
              action: "show_user_detail"
    
    modal id="user_detail":
        title: "User Details"
        content:
            show card "User Info":
                header:
                    title: "{{session.selected_user.name}}"
                sections:
                    - type: "info_grid"
                      items:
                          - label: "Email"
                            value: "{{session.selected_user.email}}"
                          - label: "Role"
                            value: "{{session.selected_user.role}}"
        actions:
            - label: "Close"
              action: "close_modal"
    
    action "show_user_detail":
        set session.selected_user to row
        show modal "user_detail"
```

### Tabbed Navigation

Use tabs for related pages:

```namel3ss
page "Settings" at "/settings":
    tabs:
        tab "Profile":
            show form "Edit Profile":
                fields:
                    - name: "name"
                      component: "text_input"
                on submit:
                    update users set { name: name }
        
        tab "Security":
            show form "Change Password":
                fields:
                    - name: "current_password"
                      component: "password_input"
                    - name: "new_password"
                      component: "password_input"
        
        tab "Notifications":
            show text "Notification preferences"
```

### Wizard Pattern

Multi-step form with state:

```namel3ss
memory:
    scope: session
    storage:
        wizard_step: int
        wizard_data: object

page "Job Wizard" at "/jobs/new":
    tabs:
        tab "Step 1: Upload File":
            show form "Upload":
                fields:
                    - name: "file"
                      component: "file_input"
                on submit:
                    set session.wizard_data.file to file
                    set session.wizard_step to 2
        
        tab "Step 2: Configure":
            show form "Options":
                fields:
                    - name: "format"
                      component: "select"
                      options: ["pdf", "csv", "json"]
                on submit:
                    set session.wizard_data.format to format
                    call create_job with session.wizard_data
                    go to page "/jobs"
```

### Search and Filter

Persistent search state:

```namel3ss
page "Users" at "/users":
    show data_table "All Users" from dataset users:
        toolbar:
            search:
                enabled: true
                placeholder: "Search users..."
                binding: "session.user_search"
            filters:
                - label: "Role"
                  field: "role"
                  options: ["admin", "user", "guest"]
                  binding: "session.role_filter"
        filter_by: "name LIKE '%{{session.user_search}}%' AND role = '{{session.role_filter}}'"
```

---

## Limitations

### ❌ Dynamic Route Parameters

**Not Supported:**
```namel3ss
page "User Detail" at "/users/:id"  # ❌ No :id parameter
page "Job Detail" at "/jobs/:jobId"  # ❌ No :jobId parameter
```

**Alternative:** Use session state (see [Pattern 1](#pattern-1-session-state))

### ❌ Nested Routes

**Not Supported:**
```namel3ss
page "User Detail" at "/users/:id/posts/:postId"  # ❌ No nested params
```

**Alternative:** Use flat routes with state:
```namel3ss
page "User Posts" at "/user-posts"
# Use session.selected_user_id and session.selected_post_id
```

### ❌ Query Parameters

**Not Supported:**
```namel3ss
page "Search" at "/search?q=:query"  # ❌ No query params
```

**Alternative:** Use session state for search terms:
```namel3ss
page "Search" at "/search"
# Use session.search_query
```

### ❌ Programmatic useNavigate

**Not Supported:**
```javascript
const navigate = useNavigate();
navigate(`/users/${userId}`);  // ❌ React Router hooks not exposed
```

**Alternative:** Use Namel3ss actions:
```namel3ss
action "view_user":
    set session.selected_user_id to user_id
    go to page "/user-detail"
```

### ❌ URL-Shareable Links

Because identifiers are stored in session/memory, URLs cannot be shared directly.

**Workaround:** For shareable links, consider:
1. Using query parameters in external links (handle in frontend customization)
2. Building a separate landing page that accepts URL params and sets session state
3. Using deep linking with custom URL schemes (advanced)

---

## Best Practices

### 1. Use Consistent Navigation Patterns

Choose one pattern (session state, modals, etc.) and use it throughout your app.

### 2. Store Full Objects

When navigating to detail pages, store the full object in session:

```namel3ss
action "view_item":
    set session.selected_item to row  # Full object
    go to page "/item-detail"
```

This avoids additional database queries and type issues.

### 3. Clear State on Navigation

Reset session state when navigating away:

```namel3ss
action "go_back_to_list":
    set session.selected_item to null
    go to page "/items"
```

### 4. Use Modals for Quick Views

Reserve separate pages for complex detail views. Use modals for quick information display.

### 5. Provide Breadcrumbs

Help users understand where they are:

```namel3ss
breadcrumbs:
    - label: "Home"
      link: "/"
    - label: "{{section}}"
      link: "/{{section}}"
    - label: "Current Page"
```

### 6. Test Navigation Flows

Ensure all navigation paths work correctly:
- Click through all links
- Test browser back button behavior
- Verify state is maintained/cleared appropriately

---

## Examples

### Complete Navigation Example

```namel3ss
app "Job Management"

dataset "FileProcessingJob":
    schema:
        id: uuid
        filename: text
        status: text
        created_at: timestamp

memory:
    scope: session
    storage:
        selected_job: object

page "Dashboard" at "/":
    sidebar:
        title: "Job Manager"
        nav:
            - label: "Dashboard"
              link: "/"
            - label: "Jobs"
              link: "/jobs"
            - label: "Settings"
              link: "/settings"
    
    show stat_summary "Overview":
        stats:
            - label: "Total Jobs"
              value_binding: "metrics.total_jobs"
            - label: "Active"
              value_binding: "metrics.active_jobs"

page "Jobs" at "/jobs":
    sidebar:
        title: "Job Manager"
        nav:
            - label: "Dashboard"
              link: "/"
            - label: "Jobs"
              link: "/jobs"
            - label: "Settings"
              link: "/settings"
    
    breadcrumbs:
        - label: "Home"
          link: "/"
        - label: "Jobs"
    
    show data_table "All Jobs" from dataset FileProcessingJob:
        columns:
            - field: "filename"
              header: "File"
            - field: "status"
              header: "Status"
              render:
                  type: "badge"
            - field: "created_at"
              header: "Created"
        row_actions:
            - label: "View Details"
              action: "view_job"
            - label: "Retry"
              action: "retry_job"
              condition: "{{status == 'failed'}}"
        toolbar:
            actions:
                - label: "New Job"
                  link: "/jobs/new"
    
    action "view_job":
        set session.selected_job to row
        go to page "/job-detail"
    
    action "retry_job":
        call retry_job_function with { job_id: row.id }
        show toast "Job retried" type="success"

page "Job Detail" at "/job-detail":
    sidebar:
        title: "Job Manager"
        nav:
            - label: "Dashboard"
              link: "/"
            - label: "Jobs"
              link: "/jobs"
            - label: "Settings"
              link: "/settings"
    
    breadcrumbs:
        - label: "Home"
          link: "/"
        - label: "Jobs"
          link: "/jobs"
        - label: "Job {{session.selected_job.id}}"
    
    show card "Job Information":
        header:
            title: "{{session.selected_job.filename}}"
            badges:
                - text: "{{session.selected_job.status}}"
                  style: "badge-{{session.selected_job.status}}"
        sections:
            - type: "info_grid"
              items:
                  - label: "Job ID"
                    value: "{{session.selected_job.id}}"
                  - label: "File"
                    value: "{{session.selected_job.filename}}"
                  - label: "Status"
                    value: "{{session.selected_job.status}}"
                  - label: "Created"
                    value: "{{session.selected_job.created_at}}"
        actions:
            - label: "Back to List"
              link: "/jobs"
            - label: "Retry"
              action: "retry_this_job"
              condition: "{{session.selected_job.status == 'failed'}}"
    
    action "retry_this_job":
        call retry_job_function with { job_id: session.selected_job.id }
        go to page "/jobs"

page "New Job" at "/jobs/new":
    sidebar:
        title: "Job Manager"
        nav:
            - label: "Dashboard"
              link: "/"
            - label: "Jobs"
              link: "/jobs"
            - label: "Settings"
              link: "/settings"
    
    breadcrumbs:
        - label: "Home"
          link: "/"
        - label: "Jobs"
          link: "/jobs"
        - label: "New Job"
    
    show form "Create Job":
        fields:
            - name: "file"
              component: "file_input"
              label: "File"
              required: true
              accept: ".pdf,.csv,.json"
            - name: "priority"
              component: "select"
              label: "Priority"
              options: ["low", "normal", "high"]
        on submit:
            call create_job with { file: file, priority: priority }
            show toast "Job created" type="success"
            go to page "/jobs"
```

---

## Next Steps

- **[UI_COMPONENT_REFERENCE.md](./UI_COMPONENT_REFERENCE.md)** - Complete component catalog
- **[FORMS_REFERENCE.md](./FORMS_REFERENCE.md)** - Form components and validation
- **[CHROME_COMPONENTS_GUIDE.md](./CHROME_COMPONENTS_GUIDE.md)** - Sidebar, navbar, modals
- **[API_AND_NAVIGATION_PATTERNS.md](./API_AND_NAVIGATION_PATTERNS.md)** - External API integration

---

**Last Updated:** November 28, 2025  
**Version:** 0.6.1
