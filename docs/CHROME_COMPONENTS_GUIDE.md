# Chrome Components - Developer Guide

## Quick Start

Add navigation and app chrome to your namel3ss application with simple declarative syntax.

## Basic Example

```namel3ss
app "My Application"

page "Dashboard" at "/":
    # Add sidebar navigation
    sidebar:
        item "Dashboard" at "/" icon "📊"
        item "Settings" at "/settings" icon "⚙️"
    
    # Add top navbar
    navbar:
        title: "My App"
        action "User" icon "👤" type "menu":
            item "Profile" at "/profile"
            item "Logout" action "logout"
    
    # Add breadcrumbs
    breadcrumbs:
        item "Home" at "/"
    
    show text "Welcome to Dashboard"
```

## Components

### Sidebar

Hierarchical navigation with unlimited nesting.

**Basic Sidebar**:
```namel3ss
sidebar:
    item "Dashboard" at "/" icon "📊"
    item "Analytics" at "/analytics" icon "📈"
    item "Settings" at "/settings" icon "⚙️"
```

**Nested Items**:
```namel3ss
sidebar:
    item "Reports" at "/reports" icon "📋":
        item "Sales" at "/reports/sales"
        item "Revenue" at "/reports/revenue"
        item "Custom" at "/reports/custom"
```

**Sections**:
```namel3ss
sidebar:
    section "Admin":
        item "Users" at "/admin/users"
        item "Roles" at "/admin/roles"
        collapsible: true
        collapsed by default: false
```

**With Badges**:
```namel3ss
sidebar:
    item "Messages" at "/messages" icon "✉️" badge {text: "New"}
    item "Alerts" at "/alerts" icon "🔔" badge {count: 5}
```

**Configuration Options**:
```namel3ss
sidebar:
    width: normal              # narrow | normal | wide
    position: left             # left | right
    collapsible: true
    collapsed by default: false
```

### Navbar

Top-level navigation with branding and actions.

**Basic Navbar**:
```namel3ss
navbar:
    logo: "/assets/logo.png"
    title: "My Application"
```

**With Actions**:
```namel3ss
navbar:
    logo: "/assets/logo.png"
    title: "My App"
    
    action "Theme" icon "🎨" type "toggle"
    action "Notifications" icon "🔔" type "button"
    action "Help" icon "❓" type "button"
```

**With Menu**:
```namel3ss
navbar:
    title: "My App"
    
    action "User" icon "👤" type "menu":
        item "Profile" at "/profile"
        item "Settings" at "/settings"
        item "Logout" action "logout"
```

**Configuration**:
```namel3ss
navbar:
    position: top              # top | bottom
    sticky: true               # Stay visible on scroll
```

### Breadcrumbs

Show navigation path to current page.

**Manual Breadcrumbs**:
```namel3ss
breadcrumbs:
    item "Home" at "/"
    item "Reports" at "/reports"
    item "Sales"
    separator: "/"
```

**Auto-Derived Breadcrumbs**:
```namel3ss
breadcrumbs:
    auto_derive: true
    separator: ">"
```

Automatically generates breadcrumbs from the URL path.

**Custom Separator**:
```namel3ss
breadcrumbs:
    separator: " > "           # Default: "/"
```

### Command Palette

Keyboard-driven command launcher with search.

**Basic Command Palette**:
```namel3ss
command palette:
    shortcut: "Ctrl+K"
    placeholder: "Search or jump to..."
```

**With API Sources**:
```namel3ss
command palette:
    shortcut: "Ctrl+K"
    
    source "documents" from "/api/search/documents" label "Search Documents"
    source "users" from "/api/search/users" label "Find Users"
    
    placeholder: "Quick search..."
    max results: 10
```

**Configuration**:
```namel3ss
command palette:
    shortcut: "Ctrl+K"         # Keyboard shortcut to open
    placeholder: "Search..."   # Placeholder text
    max results: 10            # Limit displayed results
```

## Complete Examples

### Full-Featured Dashboard

```namel3ss
app "Dashboard App"

page "Dashboard" at "/":
    sidebar:
        item "Dashboard" at "/" icon "📊"
        item "Analytics" at "/analytics" icon "📈"
        item "Reports" at "/reports" icon "📋":
            item "Sales" at "/reports/sales"
            item "Revenue" at "/reports/revenue"
        
        section "Settings":
            item "Profile" at "/profile" icon "👤"
            item "Security" at "/security" icon "🔒"
            collapsible: true
        
        width: normal
        collapsible: true
    
    navbar:
        logo: "/logo.png"
        title: "Dashboard"
        
        action "Theme" icon "🎨" type "toggle"
        action "Notifications" icon "🔔" type "button"
        action "User" icon "👤" type "menu":
            item "Profile" at "/profile"
            item "Settings" at "/settings"
            item "Logout" action "logout"
        
        sticky: true
    
    breadcrumbs:
        item "Home" at "/"
    
    command palette:
        shortcut: "Ctrl+K"
        source "documents" from "/api/search/docs" label "Documents"
        placeholder: "Quick search..."
    
    show text "Welcome to Dashboard"

page "Analytics" at "/analytics":
    sidebar:
        item "Dashboard" at "/" icon "📊"
        item "Analytics" at "/analytics" icon "📈"
    
    navbar:
        title: "Analytics"
    
    breadcrumbs:
        item "Home" at "/"
        item "Analytics"
    
    show text "Analytics Page"
```

### Report Detail with Auto Breadcrumbs

```namel3ss
page "Sales Report" at "/reports/sales":
    sidebar:
        item "Dashboard" at "/" icon "📊"
        item "Reports" at "/reports" icon "📋":
            item "Sales" at "/reports/sales"
            item "Revenue" at "/reports/revenue"
    
    navbar:
        title: "Sales Report"
        action "Export" icon "📥" type "button"
        action "Refresh" icon "🔄" type "button"
    
    breadcrumbs:
        auto_derive: true
        separator: " / "
    
    show text "Sales Report Details"
```

## Best Practices

### 1. Consistent Navigation

Use the same sidebar structure across all pages in your app:

```namel3ss
# Define navigation once, use everywhere
page "Page1" at "/page1":
    sidebar:
        item "Home" at "/"
        item "Page1" at "/page1"
        item "Page2" at "/page2"
    
    navbar:
        title: "My App"
    
    # Page content...

page "Page2" at "/page2":
    sidebar:
        item "Home" at "/"
        item "Page1" at "/page1"
        item "Page2" at "/page2"
    
    navbar:
        title: "My App"
    
    # Page content...
```

### 2. Icon Usage

Use emoji or icon identifiers consistently:

```namel3ss
sidebar:
    item "Dashboard" at "/" icon "📊"      # Emoji
    item "Users" at "/users" icon "👥"
    item "Settings" at "/settings" icon "⚙️"
```

### 3. Section Organization

Group related items in collapsible sections:

```namel3ss
sidebar:
    # Main navigation
    item "Dashboard" at "/"
    item "Analytics" at "/analytics"
    
    # Admin section
    section "Administration":
        item "Users" at "/admin/users"
        item "Roles" at "/admin/roles"
        item "Audit Log" at "/admin/audit"
        collapsible: true
        collapsed by default: true
```

### 4. Breadcrumb Strategy

Use manual breadcrumbs for custom labels, auto-derive for simple paths:

```namel3ss
# Custom labels
breadcrumbs:
    item "Home" at "/"
    item "Customer Reports" at "/reports"
    item "Q4 2025"

# Or auto-derive from URL
breadcrumbs:
    auto_derive: true
```

### 5. Command Palette Integration

Connect command palette to your backend search:

```namel3ss
command palette:
    shortcut: "Ctrl+K"
    
    # Backend-powered search
    source "documents" from "/api/search/documents" label "Documents"
    source "users" from "/api/search/users" label "Users"
    source "projects" from "/api/search/projects" label "Projects"
    
    max results: 20
```

## Keyboard Shortcuts

### Sidebar
- **Arrow Up/Down**: Navigate items
- **Home/End**: Jump to first/last item
- **Enter**: Follow link

### Navbar
- **Escape**: Close open menus
- **Tab**: Navigate between actions

### Breadcrumbs
- **Tab**: Navigate between crumbs

### Command Palette
- **Ctrl+K**: Open/close palette (configurable)
- **Arrow Up/Down**: Navigate results
- **Enter**: Execute command
- **Escape**: Close palette

## Accessibility

All chrome components are built with accessibility in mind:

- **ARIA Labels**: Proper roles and labels for screen readers
- **Semantic HTML**: Native elements (<nav>, <ol>, <button>)
- **Keyboard Navigation**: Full keyboard support
- **Focus Management**: Proper focus traps and indicators

## Styling

Chrome components include base CSS classes for customization:

### Sidebar Classes
- `.sidebar` - Main container
- `.sidebar-narrow`, `.sidebar-wide` - Width variants
- `.nav-item` - Individual items
- `.nav-item.active` - Active/current item
- `.nav-section` - Section container
- `.nav-badge` - Badge indicator

### Navbar Classes
- `.navbar` - Main container
- `.navbar-sticky` - Sticky variant
- `.navbar-logo` - Logo image
- `.navbar-title` - App title
- `.navbar-action` - Action button

### Breadcrumbs Classes
- `.breadcrumbs` - Main container
- `.breadcrumb-item` - Individual crumb
- `.breadcrumb-separator` - Separator element

### Command Palette Classes
- `.command-palette` - Dialog overlay
- `.command-palette-input` - Search input
- `.command-results` - Results container
- `.command-item` - Individual result

## Advanced Patterns

### Conditional Items

Show items based on user permissions:

```namel3ss
sidebar:
    item "Dashboard" at "/"
    item "Admin Panel" at "/admin" condition "user.isAdmin"
    item "Reports" at "/reports" condition "user.canViewReports"
```

### Dynamic Badges

Use badges to show counts or status:

```namel3ss
sidebar:
    item "Inbox" at "/inbox" icon "✉️" badge {count: 5}
    item "Alerts" at "/alerts" icon "🔔" badge {text: "New"}
    item "Beta Feature" at "/beta" badge {text: "Beta", variant: "info"}
```

### Multi-Level Nesting

Create deep navigation hierarchies:

```namel3ss
sidebar:
    item "Products" at "/products":
        item "Electronics" at "/products/electronics":
            item "Computers" at "/products/electronics/computers"
            item "Phones" at "/products/electronics/phones"
        item "Clothing" at "/products/clothing":
            item "Men" at "/products/clothing/men"
            item "Women" at "/products/clothing/women"
```

## Troubleshooting

### Sidebar not showing
- Check that sidebar is inside a page block
- Verify route paths are correct
- Ensure proper indentation

### Breadcrumbs empty with auto_derive
- Check that pages have proper route definitions
- Verify URL structure matches expected paths

### Command palette not opening
- Verify shortcut key format: "Ctrl+K", "Cmd+K", etc.
- Check that API endpoints return correct format
- Ensure command palette is inside a page block

## Migration from Previous Versions

If you're updating from earlier namel3ss versions without chrome components:

**Before**:
```namel3ss
page "Home" at "/":
    show text "Home"
    show link "Settings" to "/settings"
```

**After**:
```namel3ss
page "Home" at "/":
    sidebar:
        item "Home" at "/"
        item "Settings" at "/settings"
    
    navbar:
        title: "My App"
    
    show text "Home"
```

## Related Documentation

- [Language Reference](LANGUAGE_REFERENCE.md)
- [Component Guide](docs/component-guide.md)
- [Examples](examples/)

## Support

For issues or questions:
1. Check [Examples](examples/chrome_demo.ai)
2. Review [Test Cases](tests/test_chrome_parser.py)
3. See [Implementation Guide](CHROME_IMPLEMENTATION_FINAL.md)
