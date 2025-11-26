# Feedback Components - Quick Reference

## Modal Dialogs

### Basic Modal
```namel3ss
modal "modal_id":
  title: "Modal Title"
  description: "Modal description"
```

### With Actions
```namel3ss
modal "confirm":
  title: "Confirm Action"
  actions:
    action "Cancel" variant "ghost"
    action "Confirm" variant "primary" action "do_action"
```

### With Content
```namel3ss
modal "details":
  title: "Details"
  size: lg
  content:
    show text "Line 1 of content"
    show text "Line 2 of content"
```

### Properties
| Property | Type | Default | Options |
|----------|------|---------|---------|
| title | string | required | - |
| description | string | optional | - |
| size | string | "md" | sm, md, lg, xl, full |
| dismissible | boolean | true | true, false |
| trigger | string | optional | event name |
| content | Component[] | [] | show text, etc. |
| actions | ModalAction[] | [] | - |

### Action Properties
| Property | Type | Default | Options |
|----------|------|---------|---------|
| label | string | required | - |
| action | string | optional | action ID |
| variant | string | "default" | default, primary, destructive, ghost, link |
| close | boolean | true | true, false |

---

## Toast Notifications

### Basic Toast
```namel3ss
toast "toast_id":
  title: "Toast Title"
  variant: success
```

### With Action
```namel3ss
toast "success":
  title: "Success"
  description: "Action completed"
  variant: success
  duration: 3000
  action_label: "Undo"
  action: "undo_action"
```

### Properties
| Property | Type | Default | Options |
|----------|------|---------|---------|
| title | string | required | - |
| description | string | optional | - |
| variant | string | "default" | default, success, error, warning, info |
| duration | integer | 3000 | 0 (persistent), 2000-5000 |
| position | string | "top-right" | top, top-right, top-left, bottom, bottom-right, bottom-left |
| action_label | string | optional | - |
| action | string | optional | action ID |
| trigger | string | optional | event name |

---

## Common Patterns

### Confirmation Flow
```namel3ss
# Modal asks for confirmation
modal "confirm_delete":
  title: "Delete Item?"
  actions:
    action "Cancel" variant "ghost"
    action "Delete" variant "destructive" action "do_delete"

# Toast shows result
toast "deleted":
  title: "Deleted"
  variant: success
  trigger: "show_deleted"
```

### Form Validation
```namel3ss
modal "edit_form":
  title: "Edit Item"
  actions:
    action "Validate" variant "primary" action "validate" close false
    action "Save" variant "primary" action "save"

toast "validation_error":
  title: "Validation Failed"
  variant: error
  duration: 5000
  trigger: "show_validation_error"
```

### Error Handling
```namel3ss
# Persistent error toast
toast "error":
  title: "Connection Lost"
  variant: error
  duration: 0
  action_label: "Retry"
  action: "retry"

# Detailed error modal
modal "error_details":
  title: "Error Details"
  content:
    show text "Additional error information"
  actions:
    action "Close" variant "default"
```

---

## Size Guide

### Modal Sizes
- **sm** (400px): Small confirmations, simple forms
- **md** (600px): Default, general purpose dialogs
- **lg** (800px): Forms with multiple fields
- **xl** (1000px): Complex forms, detailed content
- **full**: Full-screen overlays, immersive experiences

### Toast Positions
- **top**: Centered at top (mobile-friendly)
- **top-right**: Default, desktop standard
- **top-left**: Alternative desktop position
- **bottom**: Centered at bottom (mobile-friendly)
- **bottom-right**: Bottom desktop position
- **bottom-left**: Bottom desktop position

---

## Variant Guide

### Modal Action Variants
- **default**: Gray, secondary actions (Cancel, Close)
- **primary**: Blue, main actions (Confirm, Save)
- **destructive**: Red, dangerous actions (Delete, Remove)
- **ghost**: Transparent, low-priority actions
- **link**: Text-only, navigation actions

### Toast Variants (with Icons)
- **default**: No icon, gray
- **success**: CheckCircle, green
- **error**: XCircle, red
- **warning**: AlertCircle, yellow
- **info**: Info, blue

---

## Duration Guide

- **2000ms**: Quick feedback (saves, updates)
- **3000ms**: Default, general notifications
- **4000ms**: Important messages
- **5000ms**: Messages requiring attention
- **0**: Persistent (errors, warnings requiring action)

---

## Event System

### Trigger Modal/Toast
```javascript
window.dispatchEvent(new CustomEvent('namel3ss:action', {
    detail: { action: 'show_modal_id' }
}));
```

### Handle Action
```javascript
window.addEventListener('namel3ss:action', (event) => {
    if (event.detail.action === 'confirm_action') {
        // Handle action
    }
});
```

---

## Best Practices

### Modal DO
✅ Clear, action-oriented titles  
✅ Sufficient context in description  
✅ Destructive variant for dangerous actions  
✅ 2-3 buttons maximum  
✅ Always provide cancel option  

### Modal DON'T
❌ Non-critical information  
❌ Chaining multiple modals  
❌ Non-dismissible without reason  
❌ Cryptic action labels  

### Toast DO
✅ Match variant to message type  
✅ Short, scannable titles  
✅ Consistent positioning  
✅ Appropriate durations  

### Toast DON'T
❌ Critical errors (use modal)  
❌ Multiple simultaneous toasts  
❌ Persistent for routine messages  
❌ Long descriptions  

---

## Accessibility

### Modal
- ESC closes (if dismissible)
- Tab cycles through actions
- Focus trapped inside
- ARIA roles applied

### Toast
- Announced to screen readers
- Color + icon (not color alone)
- Sufficient reading time
- Keyboard dismissible

---

## Mobile Tips

### Modal
- Use `lg` or `xl` for mobile (full width)
- Larger touch targets (44x44px)
- Scrollable content

### Toast
- Use `top` or `bottom` positions
- Increase duration (+1000ms)
- Full width with padding

---

## Documentation

**Full Guide**: `docs/FEEDBACK_COMPONENTS_GUIDE.md`  
**Examples**: `examples/feedback_demo.ai`  
**Summary**: `FEEDBACK_COMPONENTS_SUMMARY.md`
