# Chrome Components - Migration & Upgrade Guide

## Overview

This guide helps you add navigation and app chrome components to existing namel3ss applications.

## What's New

Chrome components provide first-class navigation features:

- **Sidebar**: Hierarchical navigation with nesting
- **Navbar**: Top-level branding and actions
- **Breadcrumbs**: Path navigation with auto-derivation
- **Command Palette**: Keyboard-driven search and commands

## Compatibility

- **namel3ss Version**: Latest (with chrome components support)
- **Breaking Changes**: None - chrome components are additive
- **Backward Compatible**: Yes - existing apps work unchanged

## Migration Steps

### Step 1: Update Your Application

Add chrome components to your existing pages:

**Before**:
```namel3ss
app "My App"

page "Home" at "/":
    show text "Welcome"
    show link "About" to "/about"

page "About" at "/about":
    show text "About Us"
    show link "Home" to "/"
```

**After**:
```namel3ss
app "My App"

page "Home" at "/":
    sidebar:
        item "Home" at "/"
        item "About" at "/about"
    
    navbar:
        title: "My App"
    
    breadcrumbs:
        item "Home" at "/"
    
    show text "Welcome"

page "About" at "/about":
    sidebar:
        item "Home" at "/"
        item "About" at "/about"
    
    navbar:
        title: "My App"
    
    breadcrumbs:
        item "Home" at "/"
        item "About"
    
    show text "About Us"
```

### Step 2: Consolidate Navigation

Extract repeated sidebar/navbar into a consistent structure:

```namel3ss
# Recommended: Use identical chrome structure across related pages

page "Page1" at "/page1":
    # Chrome components
    sidebar:
        item "Page1" at "/page1"
        item "Page2" at "/page2"
        item "Page3" at "/page3"
    
    navbar:
        title: "My App"
    
    # Page content
    show text "Page 1 content"

page "Page2" at "/page2":
    # Same chrome structure
    sidebar:
        item "Page1" at "/page1"
        item "Page2" at "/page2"
        item "Page3" at "/page3"
    
    navbar:
        title: "My App"
    
    # Page content
    show text "Page 2 content"
```

### Step 3: Replace Manual Links

Convert manual navigation links to sidebar items:

**Before**:
```namel3ss
page "Dashboard" at "/":
    show text "Navigation:"
    show link "Analytics" to "/analytics"
    show link "Reports" to "/reports"
    show link "Settings" to "/settings"
```

**After**:
```namel3ss
page "Dashboard" at "/":
    sidebar:
        item "Dashboard" at "/" icon "📊"
        item "Analytics" at "/analytics" icon "📈"
        item "Reports" at "/reports" icon "📋"
        item "Settings" at "/settings" icon "⚙️"
    
    show text "Dashboard content"
```

### Step 4: Add User Actions

Convert user-related buttons to navbar actions:

**Before**:
```namel3ss
page "Home" at "/":
    show button "Toggle Theme" action "toggle_theme"
    show button "Logout" action "logout"
```

**After**:
```namel3ss
page "Home" at "/":
    navbar:
        title: "My App"
        action "Theme" icon "🎨" type "toggle"
        action "User" icon "👤" type "menu":
            item "Profile" at "/profile"
            item "Logout" action "logout"
```

### Step 5: Add Breadcrumbs

Add breadcrumb navigation to show user location:

```namel3ss
page "Report Detail" at "/reports/sales":
    breadcrumbs:
        auto_derive: true
        separator: " / "
    
    # Or manual:
    # breadcrumbs:
    #     item "Home" at "/"
    #     item "Reports" at "/reports"
    #     item "Sales"
```

## Common Patterns

### Pattern 1: Dashboard Layout

Convert dashboard with manual nav to chrome:

**Before**:
```namel3ss
page "Dashboard" at "/":
    show section "Navigation":
        show link "Home" to "/"
        show link "Analytics" to "/analytics"
        show link "Settings" to "/settings"
    
    show section "Content":
        show text "Dashboard"
```

**After**:
```namel3ss
page "Dashboard" at "/":
    sidebar:
        item "Home" at "/"
        item "Analytics" at "/analytics"
        item "Settings" at "/settings"
    
    navbar:
        title: "Dashboard"
    
    show text "Dashboard"
```

### Pattern 2: Admin Section

Convert admin menus to sections:

**Before**:
```namel3ss
page "Admin" at "/admin":
    show text "Admin Menu:"
    show link "Users" to "/admin/users"
    show link "Roles" to "/admin/roles"
    show link "Audit" to "/admin/audit"
```

**After**:
```namel3ss
page "Admin" at "/admin":
    sidebar:
        item "Dashboard" at "/"
        
        section "Administration":
            item "Users" at "/admin/users"
            item "Roles" at "/admin/roles"
            item "Audit Log" at "/admin/audit"
            collapsible: true
```

### Pattern 3: Multi-Level Navigation

Convert nested menus to hierarchical sidebar:

**Before**:
```namel3ss
page "Reports" at "/reports":
    show text "Reports:"
    show link "All Reports" to "/reports"
    show text "Sales:"
    show link "- Monthly" to "/reports/sales/monthly"
    show link "- Quarterly" to "/reports/sales/quarterly"
```

**After**:
```namel3ss
page "Reports" at "/reports":
    sidebar:
        item "Reports" at "/reports":
            item "Sales Reports" at "/reports/sales":
                item "Monthly" at "/reports/sales/monthly"
                item "Quarterly" at "/reports/sales/quarterly"
            item "Revenue Reports" at "/reports/revenue"
```

## Incremental Migration

You don't have to migrate all at once. Chrome components work alongside existing content:

### Phase 1: Add Sidebar

```namel3ss
page "Home" at "/":
    sidebar:
        item "Home" at "/"
        item "About" at "/about"
    
    # Keep existing content
    show text "Welcome"
    show button "Get Started" action "start"
```

### Phase 2: Add Navbar

```namel3ss
page "Home" at "/":
    sidebar:
        item "Home" at "/"
        item "About" at "/about"
    
    navbar:
        title: "My App"
    
    # Keep existing content
    show text "Welcome"
```

### Phase 3: Add Breadcrumbs & Command Palette

```namel3ss
page "Home" at "/":
    sidebar:
        item "Home" at "/"
        item "About" at "/about"
    
    navbar:
        title: "My App"
    
    breadcrumbs:
        item "Home" at "/"
    
    command palette:
        shortcut: "Ctrl+K"
    
    show text "Welcome"
```

## Testing Your Migration

After migrating, verify:

1. **Build succeeds**: `namel3ss build app.ai --target react-vite`
2. **All routes work**: Check navigation links function
3. **Chrome renders**: Verify sidebar, navbar, breadcrumbs appear
4. **Keyboard works**: Test Ctrl+K, arrow keys, etc.

### Quick Test

```bash
# Build your app
namel3ss build your_app.ai --target react-vite --out dist

# Check for chrome components
ls dist/src/components/Sidebar.tsx
ls dist/src/components/Navbar.tsx
ls dist/src/components/Breadcrumbs.tsx
ls dist/src/components/CommandPalette.tsx
```

## Troubleshooting

### Issue: Chrome components not generating

**Problem**: Components don't appear in build output

**Solution**: Ensure chrome blocks are inside page declarations:

```namel3ss
# ✅ Correct
page "Home" at "/":
    sidebar:
        item "Home" at "/"

# ❌ Wrong - outside page
sidebar:
    item "Home" at "/"

page "Home" at "/":
    show text "Home"
```

### Issue: Routes don't work

**Problem**: Sidebar links lead to 404

**Solution**: Ensure all sidebar routes have corresponding pages:

```namel3ss
# Sidebar references these routes
sidebar:
    item "Home" at "/"
    item "About" at "/about"
    item "Contact" at "/contact"

# Must have matching pages
page "Home" at "/"
page "About" at "/about"
page "Contact" at "/contact"
```

### Issue: Indentation errors

**Problem**: Parse errors about indentation

**Solution**: Use consistent 4-space indentation:

```namel3ss
# ✅ Correct
page "Home" at "/":
    sidebar:
        item "Home" at "/"
        item "About" at "/about"

# ❌ Wrong - inconsistent spaces
page "Home" at "/":
  sidebar:
      item "Home" at "/"
   item "About" at "/about"
```

### Issue: Nested items not rendering

**Problem**: Child items don't appear under parent

**Solution**: Ensure child items are indented under parent with colon:

```namel3ss
# ✅ Correct
sidebar:
    item "Reports" at "/reports":
        item "Sales" at "/reports/sales"
        item "Revenue" at "/reports/revenue"

# ❌ Wrong - missing colon after parent
sidebar:
    item "Reports" at "/reports"
        item "Sales" at "/reports/sales"
```

## Advanced Migration

### Custom Component Integration

Chrome components work with other namel3ss features:

```namel3ss
page "Dashboard" at "/" reactive:
    # Chrome components
    sidebar:
        item "Dashboard" at "/"
        item "Analytics" at "/analytics"
    
    navbar:
        title: "Dashboard"
        action "Refresh" icon "🔄" type "button"
    
    # Data-driven content
    connect to dataset "metrics"
    
    show stat summary from metrics:
        metric "Total Users" from "user_count"
        metric "Revenue" from "revenue_total"
```

### API Integration

Connect command palette to your backend:

```namel3ss
page "Home" at "/":
    command palette:
        shortcut: "Ctrl+K"
        
        # Your API endpoints
        source "products" from "/api/search/products" label "Search Products"
        source "orders" from "/api/search/orders" label "Search Orders"
        source "customers" from "/api/search/customers" label "Search Customers"
```

## Performance Considerations

Chrome components are rendered on every page. For optimal performance:

1. **Consistent Structure**: Use same sidebar across pages to enable React optimization
2. **Minimal Nesting**: Limit sidebar nesting to 3-4 levels
3. **Lazy Loading**: Command palette only loads when opened
4. **Route Validation**: Build-time validation prevents broken links

## Rollback

If you need to revert:

1. Remove chrome component blocks from pages
2. Rebuild your application
3. Chrome components won't be generated

Your app will work as before - chrome is completely additive.

## Next Steps

After migration:

1. **Customize Styling**: Add CSS for your brand
2. **Add Icons**: Enhance with meaningful icons
3. **Use Badges**: Show counts and notifications
4. **Setup Command Palette**: Connect to search APIs
5. **Add Sections**: Organize large navigation hierarchies

## Resources

- [Developer Guide](CHROME_COMPONENTS_GUIDE.md) - Complete syntax reference
- [Examples](../examples/chrome_demo.ai) - Full demo application
- [Tests](../tests/test_chrome_parser.py) - Test examples for syntax
- [Implementation](CHROME_IMPLEMENTATION_FINAL.md) - Technical details

## Support

Migration issues? Check:

1. **Syntax**: Review [Developer Guide](CHROME_COMPONENTS_GUIDE.md)
2. **Examples**: See [chrome_demo.ai](../examples/chrome_demo.ai)
3. **Tests**: Reference [test files](../tests/test_chrome_parser.py)
4. **Build Logs**: Check error messages for specific issues

All chrome components are backward compatible and designed for smooth integration with existing applications.
