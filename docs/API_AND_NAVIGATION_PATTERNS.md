# API Integration & Navigation Patterns

**Complete guide to external APIs, routing, actions, and widget embedding in Namel3ss**

This guide covers how to integrate external APIs, implement navigation and routing, use the action system, and embed widgets in other applications. These features enable building interactive, data-driven applications.

---

## Table of Contents

1. [API Integration](#api-integration)
   - [REST API Connectors](#rest-api-connectors)
   - [GraphQL Connectors](#graphql-connectors)
   - [Response Processing](#response-processing)
   - [Error Handling](#error-handling)
2. [Navigation & Routing](#navigation--routing)
   - [Page Routes](#page-routes)
   - [Programmatic Navigation](#programmatic-navigation)
   - [Navigation Components](#navigation-components)
3. [Action System](#action-system)
   - [Action Definitions](#action-definitions)
   - [Action Operations](#action-operations)
   - [Navbar & Toolbar Actions](#navbar--toolbar-actions)
4. [Widget Embedding](#widget-embedding)
   - [Iframe-Safe Widgets](#iframe-safe-widgets)
   - [Embedding Script](#embedding-script)
   - [Cross-Origin Communication](#cross-origin-communication)

---

## API Integration

Namel3ss provides built-in connectors for integrating with external REST and GraphQL APIs. Connectors handle authentication, retries, response parsing, and error handling automatically.

### REST API Connectors

REST connectors fetch data from HTTP endpoints and make it available to your application.

#### Basic REST Connector

```namel3ss
# Define a REST API connector
connector "userAPI":
  type: rest
  options:
    endpoint: "https://api.example.com/users"
    method: get
    headers:
      Authorization: "Bearer {{api_key}}"
      Content-Type: "application/json"
```

#### REST Connector with POST

```namel3ss
connector "createUser":
  type: rest
  options:
    endpoint: "https://api.example.com/users"
    method: post
    headers:
      Authorization: "Bearer {{api_key}}"
      Content-Type: "application/json"
    payload:
      name: "{{user_name}}"
      email: "{{user_email}}"
```

#### REST Connector Configuration Options

| Option | Type | Description |
|--------|------|-------------|
| `endpoint` | string | Full URL of the API endpoint |
| `method` | string | HTTP method: `get`, `post`, `put`, `patch`, `delete` |
| `headers` | object | HTTP headers (supports `{{variable}}` interpolation) |
| `payload` | object | Request body for POST/PUT/PATCH (supports interpolation) |
| `result_path` | string | Path to extract data from nested response (e.g., `data.users`) |
| `retry_attempts` | integer | Number of retry attempts on failure (default: 3) |
| `timeout` | integer | Request timeout in seconds (default: 30) |

### GraphQL Connectors

GraphQL connectors execute queries and mutations against GraphQL endpoints.

#### Basic GraphQL Query

```namel3ss
connector "githubRepos":
  type: graphql
  options:
    endpoint: "https://api.github.com/graphql"
    query: """
      query GetRepos($owner: String!) {
        user(login: $owner) {
          repositories(first: 10) {
            nodes {
              name
              description
              stargazerCount
            }
          }
        }
      }
    """
    variables:
      owner: "{{github_username}}"
    headers:
      Authorization: "Bearer {{github_token}}"
```

#### GraphQL Mutation

```namel3ss
connector "createIssue":
  type: graphql
  options:
    endpoint: "https://api.github.com/graphql"
    query: """
      mutation CreateIssue($repoId: ID!, $title: String!, $body: String!) {
        createIssue(input: {
          repositoryId: $repoId
          title: $title
          body: $body
        }) {
          issue {
            id
            number
            url
          }
        }
      }
    """
    variables:
      repoId: "{{repository_id}}"
      title: "{{issue_title}}"
      body: "{{issue_body}}"
    headers:
      Authorization: "Bearer {{github_token}}"
    root: "data.createIssue.issue"
```

#### GraphQL Connector Options

| Option | Type | Description |
|--------|------|-------------|
| `endpoint` | string | GraphQL endpoint URL |
| `query` | string | GraphQL query or mutation (use triple quotes for multiline) |
| `variables` | object | Query variables (supports `{{variable}}` interpolation) |
| `headers` | object | HTTP headers |
| `root` | string | Path to extract result from response (e.g., `data.user.repositories.nodes`) |

### Response Processing

Connectors return data that can be bound to UI components. The runtime automatically normalizes responses into arrays of records.

#### Using Connector Data in Pages

```namel3ss
page "Users" at "/users":
  # Fetch data from connector
  show list from connector userAPI:
    item:
      show text "{{name}}"
      show text "{{email}}"
      show text "{{role}}"
```

#### Response Normalization

The runtime normalizes different response structures:

- **Array responses**: Used directly
- **Object with array field**: Extracts array using `result_path`
- **Single object**: Wrapped in array `[object]`
- **Nested structures**: Traversed using dot notation

Example transformations:

```javascript
// Input: { data: { users: [...] } }
// With result_path: "data.users"
// Output: [...]

// Input: { items: [...], total: 10 }
// With result_path: "items"
// Output: [...]

// Input: { user: { name: "Alice" } }
// With result_path: "user"
// Output: [{ name: "Alice" }]
```

### Error Handling

Connectors include automatic error handling with retries and fallbacks.

#### Retry Configuration

```namel3ss
connector "unreliableAPI":
  type: rest
  options:
    endpoint: "https://api.example.com/data"
    method: get
    retry_attempts: 5
    timeout: 10
```

#### Using Connector Data with Error States

```namel3ss
page "DataView" at "/data":
  show list from connector unreliableAPI:
    loading: "Fetching data..."
    empty: "No data available"
    error: "Failed to load data. Please try again."
    item:
      show text "{{value}}"
```

---

## Navigation & Routing

Namel3ss uses file-based routing with automatic React Router integration. Navigation can be declarative or programmatic.

### Page Routes

Each page defines a route using the `at` keyword.

#### Basic Routing

```namel3ss
app "Multi-Page App"

# Home page
page "Home" at "/":
  show text "Welcome to the app!"

# About page
page "About" at "/about":
  show text "About us"

# User profile with parameter
page "UserProfile" at "/users/:userId":
  show text "Profile for user: {{userId}}"
```

#### Route Parameters

Routes support dynamic parameters using `:paramName` syntax:

```namel3ss
page "ProductDetail" at "/products/:productId":
  # Access route parameter
  show text "Product ID: {{productId}}"
  
  # Use in connector
  show list from connector getProduct where { id: "{{productId}}" }:
    item:
      show text "{{name}}"
      show text "Price: ${{price}}"
```

### Programmatic Navigation

Navigate between pages using actions and operations.

#### GoToPage Operation

```namel3ss
page "Dashboard" at "/dashboard":
  show text "Dashboard"
  
  # Action that navigates to another page
  action "View Profile" when clicks "Go to Profile":
    go to page "UserProfile"
```

#### Navigation with Toast

```namel3ss
action "Save Settings" when clicks "Save":
  # Save operation
  update settings set { theme: "{{selected_theme}}" }
  
  # Show confirmation
  show toast "Settings saved!"
  
  # Navigate to dashboard
  go to page "Dashboard"
```

### Navigation Components

Built-in chrome components provide navigation UI.

#### Navbar with Navigation

```namel3ss
page "Home" at "/":
  navbar:
    logo: "🏠"
    title: "My App"
    actions:
      action "Home" icon "🏠" type "button"
      action "Settings" icon "⚙️" type "button"
      action "User" icon "👤" type "menu":
        item "Profile" at "/profile"
        item "Logout" action "logout"
```

#### Sidebar Navigation

```namel3ss
page "Dashboard" at "/dashboard":
  sidebar:
    title: "Dashboard"
    items:
      item "Overview" at "/" icon "📊"
      item "Analytics" at "/analytics" icon "📈"
      item "Reports" at "/reports" icon "📄"
      section "Settings":
        item "Profile" at "/profile" icon "👤"
        item "Security" at "/security" icon "🔒"
```

#### Breadcrumbs

```namel3ss
page "ProductDetail" at "/products/:productId":
  breadcrumbs:
    item "Home" at "/"
    item "Products" at "/products"
    item "Product {{productId}}"
```

#### Command Palette for Quick Navigation

```namel3ss
page "Home" at "/":
  command palette:
    shortcut: "Ctrl+K"
    source "pages" from routes
    source "actions" from available_actions
```

The command palette provides keyboard-driven navigation:
- **Ctrl+K** (or Cmd+K on Mac) opens the palette
- Type to filter available routes and actions
- Arrow keys to navigate, Enter to execute
- Escape to close

---

## Action System

Actions define interactive behaviors triggered by user interactions. They can perform operations like updating data, showing toasts, running chains, or navigating.

### Action Definitions

Actions use the `action` keyword with a trigger condition.

#### Basic Action Syntax

```namel3ss
action "ActionName" when clicks "Button Label":
  # Operations go here
  show toast "Action executed!"
```

#### Trigger Conditions

| Trigger | Description | Example |
|---------|-------------|---------|
| `clicks "Label"` | Button click | `when clicks "Submit"` |
| `submits form` | Form submission | `when submits form` |
| Custom event | Custom JavaScript event | `when event "custom:event"` |

### Action Operations

Actions can perform multiple operations in sequence.

#### Available Operations

**1. Show Toast**

```namel3ss
action "Notify" when clicks "Show Notification":
  show toast "Operation successful!"
```

**2. Go To Page**

```namel3ss
action "Navigate" when clicks "Go to Dashboard":
  go to page "Dashboard"
```

**3. Update Data**

```namel3ss
action "UpdateStatus" when clicks "Activate":
  update users set { status: "active" } where { id: "{{user_id}}" }
  show toast "User activated!"
```

**4. Run Chain**

```namel3ss
action "GenerateReport" when clicks "Generate":
  run chain reportGenerator with { user_id: "{{user_id}}" }
  show toast "Report generated!"
```

**5. Run Prompt**

```namel3ss
action "Analyze" when clicks "Analyze":
  run prompt analyzer with { text: "{{input_text}}" }
  show toast "Analysis complete!"
```

**6. Ask Connector**

```namel3ss
action "FetchData" when clicks "Refresh":
  ask connector dataAPI with { page: "{{current_page}}" }
  show toast "Data refreshed!"
```

**7. Call Python**

```namel3ss
action "Process" when clicks "Process Data":
  call python custom.processors.process_data with { data: "{{raw_data}}" }
  show toast "Processing complete!"
```

#### Combining Operations

Actions can chain multiple operations:

```namel3ss
action "CompleteTask" when clicks "Complete":
  # 1. Update database
  update tasks set { status: "completed", completed_at: "{{now}}" } where { id: "{{task_id}}" }
  
  # 2. Show confirmation
  show toast "Task completed!"
  
  # 3. Navigate to list
  go to page "TaskList"
```

### Navbar & Toolbar Actions

Actions can be placed in chrome components for global access.

#### Navbar Actions

```namel3ss
page "Home" at "/":
  navbar:
    logo: "🏠"
    title: "My App"
    actions:
      action "Theme" icon "🎨" type "toggle"
      action "Notifications" icon "🔔" type "button" badge {count: 3}
      action "User" icon "👤" type "menu":
        item "Profile" at "/profile"
        item "Settings" at "/settings"
        item "Logout" action "logout"
```

#### Action Types

| Type | Description | Use Case |
|------|-------------|----------|
| `button` | Single-click action | Refresh, Save, Export |
| `menu` | Dropdown menu with items | User menu, More options |
| `toggle` | On/off switch | Theme toggle, Enable feature |

#### Toolbar Actions

```namel3ss
page "Dashboard" at "/dashboard":
  show data_table "Reports" from dataset reports:
    toolbar:
      action "Export" icon "📥" type "button"
      action "Refresh" icon "🔄" type "button"
      action "Filter" icon "🔍" type "button"
```

---

## Widget Embedding

Namel3ss applications can be embedded in other websites as widgets using iframes.

### Iframe-Safe Widgets

Pages can be configured for iframe embedding with cross-origin support.

#### Embeddable Widget Page

```namel3ss
app "Support Chat Widget"

page "ChatWidget" at "/widget":
  # Minimal layout for embedding
  # (No navbar, sidebar, or chrome components)
  
  show text "Customer Support" style {
    fontSize: "20px"
    fontWeight: "bold"
    marginBottom: "16px"
  }
  
  # Chat messages
  show list from memory "messages":
    item:
      show text "{{message}}" style {
        padding: "8px"
        background: "{{is_user ? '#e3f2fd' : '#f5f5f5'}}"
        borderRadius: "8px"
        marginBottom: "8px"
      }
  
  # Input form
  show form "Send Message":
    field "message" type text placeholder "Type your message..."
    button "Send"
    on submit:
      # Add to memory
      # Trigger chain for response
      show toast "Message sent!"
```

### Embedding Script

Generate an embedding script for the widget:

#### HTML Embedding Code

```html
<!DOCTYPE html>
<html>
<head>
  <title>Website with Chat Widget</title>
  <style>
    /* Widget container styling */
    #support-widget {
      position: fixed;
      bottom: 20px;
      right: 20px;
      width: 350px;
      height: 500px;
      border: none;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
      overflow: hidden;
      z-index: 9999;
    }
    
    /* Minimize button */
    #widget-toggle {
      position: fixed;
      bottom: 20px;
      right: 20px;
      width: 60px;
      height: 60px;
      border-radius: 50%;
      background: #1976d2;
      color: white;
      border: none;
      cursor: pointer;
      font-size: 24px;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
      z-index: 9998;
      display: none;
    }
    
    #widget-toggle.visible {
      display: block;
    }
    
    #support-widget.minimized {
      display: none;
    }
  </style>
</head>
<body>
  <h1>My Website</h1>
  <p>Website content goes here...</p>
  
  <!-- Chat widget iframe -->
  <iframe 
    id="support-widget"
    src="https://your-namel3ss-app.com/widget"
    allow="microphone; camera"
  ></iframe>
  
  <!-- Toggle button -->
  <button id="widget-toggle" onclick="toggleWidget()">💬</button>
  
  <script>
    // Widget toggle functionality
    const widgetFrame = document.getElementById('support-widget');
    const toggleBtn = document.getElementById('widget-toggle');
    
    function toggleWidget() {
      if (widgetFrame.classList.contains('minimized')) {
        widgetFrame.classList.remove('minimized');
        toggleBtn.classList.remove('visible');
      } else {
        widgetFrame.classList.add('minimized');
        toggleBtn.classList.add('visible');
      }
    }
    
    // Listen for messages from widget
    window.addEventListener('message', (event) => {
      // Verify origin
      if (event.origin !== 'https://your-namel3ss-app.com') {
        return;
      }
      
      // Handle widget events
      if (event.data.type === 'minimize') {
        toggleWidget();
      } else if (event.data.type === 'notification') {
        console.log('Widget notification:', event.data.message);
      }
    });
  </script>
</body>
</html>
```

### Cross-Origin Communication

Widgets can communicate with parent pages using `postMessage`.

#### Sending Messages from Widget

```javascript
// In widget JavaScript
window.parent.postMessage({
  type: 'notification',
  message: 'New message received',
  count: 3
}, '*'); // Replace '*' with specific origin in production
```

#### Receiving Messages in Widget

```javascript
// In widget JavaScript
window.addEventListener('message', (event) => {
  // Verify origin
  if (event.origin !== 'https://parent-site.com') {
    return;
  }
  
  // Handle parent messages
  if (event.data.type === 'userInfo') {
    console.log('Received user info:', event.data);
  }
});
```

#### Security Considerations

1. **Always validate origin**: Check `event.origin` before processing messages
2. **Use specific origins**: Replace `'*'` with actual domain in production
3. **Sanitize data**: Never trust incoming data without validation
4. **HTTPS only**: Use HTTPS for all cross-origin communication
5. **CSP headers**: Configure Content-Security-Policy headers appropriately

---

## Complete Examples

### Example 1: API Dashboard with Navigation

```namel3ss
app "API Dashboard"

# Users API connector
connector "githubUsers":
  type: rest
  options:
    endpoint: "https://api.github.com/users"
    method: get

# Single user API
connector "githubUser":
  type: rest
  options:
    endpoint: "https://api.github.com/users/{{username}}"
    method: get

# Home page with user list
page "Home" at "/":
  navbar:
    logo: "🐙"
    title: "GitHub Explorer"
    actions:
      action "Home" icon "🏠" type "button"
      action "About" icon "ℹ️" type "button"
  
  show text "GitHub Users" style {
    fontSize: "24px"
    fontWeight: "bold"
    marginBottom: "16px"
  }
  
  show list from connector githubUsers:
    loading: "Loading users..."
    empty: "No users found"
    error: "Failed to load users"
    item:
      show text "{{login}}" style {
        fontSize: "18px"
        fontWeight: "bold"
      }
      show text "Type: {{type}}"
      
      action "View Profile" when clicks "View Profile":
        go to page "UserDetail"

# User detail page
page "UserDetail" at "/users/:username":
  navbar:
    logo: "🐙"
    title: "GitHub Explorer"
  
  breadcrumbs:
    item "Home" at "/"
    item "User: {{username}}"
  
  show text "User Profile" style {
    fontSize: "24px"
    marginBottom: "16px"
  }
  
  show list from connector githubUser where { username: "{{username}}" }:
    item:
      show text "Username: {{login}}"
      show text "Name: {{name}}"
      show text "Bio: {{bio}}"
      show text "Public Repos: {{public_repos}}"
```

### Example 2: Multi-Step Form with Actions

```namel3ss
app "User Onboarding"

# Memory for form state
memory "user_data":
  storage: session
  schema:
    email: text
    name: text
    preferences: object

page "Onboarding" at "/onboarding":
  show text "Welcome! Let's get you set up." style {
    fontSize: "24px"
    marginBottom: "20px"
  }
  
  # Step 1: Basic Info
  show form "Basic Information":
    field "email" type email label "Email Address" required
    field "name" type text label "Full Name" required
    button "Next"
    on submit:
      # Save to memory
      show toast "Information saved!"
      go to page "Preferences"

page "Preferences" at "/preferences":
  show text "Set Your Preferences" style {
    fontSize: "24px"
    marginBottom: "20px"
  }
  
  # Step 2: Preferences
  show form "User Preferences":
    field "theme" type select label "Theme":
      option "Light"
      option "Dark"
      option "Auto"
    field "notifications" type checkbox label "Enable notifications"
    button "Complete Setup"
    on submit:
      # Save to memory
      show toast "Setup complete! Welcome aboard!"
      go to page "Dashboard"

page "Dashboard" at "/dashboard":
  navbar:
    logo: "👋"
    title: "Welcome, {{user_data.name}}"
  
  show text "Your Dashboard" style {
    fontSize: "24px"
    marginBottom: "20px"
  }
  
  show text "Email: {{user_data.email}}"
  show text "Theme: {{user_data.preferences.theme}}"
```

### Example 3: GraphQL Integration

```namel3ss
app "GitHub GraphQL Explorer"

connector "githubRepos":
  type: graphql
  options:
    endpoint: "https://api.github.com/graphql"
    query: """
      query GetRepos($owner: String!, $first: Int!) {
        user(login: $owner) {
          repositories(first: $first, orderBy: {field: STARGAZERS, direction: DESC}) {
            nodes {
              name
              description
              stargazerCount
              forkCount
              url
            }
          }
        }
      }
    """
    variables:
      owner: "{{github_username}}"
      first: 10
    headers:
      Authorization: "Bearer {{github_token}}"
    root: "data.user.repositories.nodes"

page "Repositories" at "/repos":
  navbar:
    logo: "🔍"
    title: "GitHub Repos"
  
  show text "Top Repositories" style {
    fontSize: "24px"
    marginBottom: "20px"
  }
  
  show list from connector githubRepos:
    loading: "Loading repositories..."
    empty: "No repositories found"
    error: "Failed to load repositories. Check your token."
    item:
      show text "{{name}}" style {
        fontSize: "18px"
        fontWeight: "bold"
      }
      show text "{{description}}"
      show text "⭐ {{stargazerCount}} stars | 🍴 {{forkCount}} forks"
      
      action "View on GitHub" when clicks "Open":
        # Note: External navigation would be handled via custom JavaScript
        show toast "Opening repository..."
```

---

## Best Practices

### API Integration

1. **Use environment variables** for API keys and secrets
2. **Set appropriate timeouts** to prevent hanging requests
3. **Implement retry logic** for unreliable APIs
4. **Use result_path** to extract nested data efficiently
5. **Handle error states** with loading/empty/error messages

### Navigation

1. **Keep routes RESTful** and predictable (`/users/:id` not `/user-detail`)
2. **Use breadcrumbs** for deep navigation hierarchies
3. **Provide clear navigation** in navbar/sidebar
4. **Use command palette** for power users
5. **Validate route parameters** before using in connectors

### Actions

1. **Provide user feedback** with toasts after actions
2. **Chain operations logically** (update → toast → navigate)
3. **Use meaningful action names** that describe behavior
4. **Handle errors gracefully** in action operations
5. **Keep actions focused** on a single responsibility

### Widget Embedding

1. **Keep widgets minimal** without unnecessary chrome
2. **Validate postMessage origins** for security
3. **Use HTTPS** for all cross-origin communication
4. **Test in multiple browsers** and iframe contexts
5. **Provide minimize/maximize** controls

---

## Next Steps

- **[Backend & Deployment Guide](BACKEND_AND_DEPLOYMENT_GUIDE.md)**: Session management, database operations, error handling
- **[Complete Examples](COMPLETE_CHAT_WIDGET_EXAMPLE.md)**: Production-ready chat widget implementation
- **[UI Components Guide](UI_COMPONENTS_AND_STYLING.md)**: Styling, conditional rendering, layouts
- **[Real-time Guide](REALTIME_AND_FORMS_GUIDE.md)**: WebSocket communication, collaborative features

---

## Summary

This guide covered:
- ✅ REST and GraphQL API connectors with authentication
- ✅ Response processing and normalization
- ✅ Page routing with parameters
- ✅ Programmatic navigation with actions
- ✅ Navigation components (navbar, sidebar, breadcrumbs, command palette)
- ✅ Action system with multiple operation types
- ✅ Widget embedding with cross-origin communication
- ✅ Complete working examples

You now have comprehensive knowledge of API integration, navigation, and action patterns in Namel3ss!
