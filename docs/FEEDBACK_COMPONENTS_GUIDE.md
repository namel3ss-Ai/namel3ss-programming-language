# Feedback Components - Developer Guide

## Overview

Feedback components provide essential user interaction patterns for confirmations, notifications, and alerts. Namel3ss includes two core feedback components:

- **Modal**: Dialog overlays for confirmations, forms, and complex interactions
- **Toast**: Notification messages for success, errors, warnings, and info

## Quick Start

```namel3ss
app "Feedback Example"

page "Dashboard" at "/":
    # Simple confirmation modal
    modal "confirm_action":
        title: "Confirm Action"
        description: "Are you sure?"
        trigger: "show_confirm"
        actions:
            action "Cancel" variant "ghost"
            action "Confirm" variant "primary" action "do_confirm"
    
    # Success notification
    toast "success":
        title: "Saved!"
        variant: success
        duration: 3000
        trigger: "show_success"
    
    show text "Your content here"
```

---

## Modal Component

Modals are dialog overlays that require user interaction. They're perfect for confirmations, forms, detailed information, and critical decisions.

### Basic Modal

```namel3ss
modal "basic_modal":
    title: "Modal Title"
    description: "Optional description text"
```

**Properties:**
- `title` (required): Main heading displayed in the modal
- `description` (optional): Subtitle or additional context

### Modal Sizes

Control the width of your modal with the `size` property.

```namel3ss
# Small modal (400px max-width)
modal "small":
    title: "Quick Tip"
    size: sm

# Medium modal (600px max-width) - Default
modal "medium":
    title: "Standard Dialog"
    size: md

# Large modal (800px max-width)
modal "large":
    title: "Detailed Form"
    size: lg

# Extra large modal (1000px max-width)
modal "extra_large":
    title: "Complex Content"
    size: xl

# Full screen modal
modal "fullscreen":
    title: "Full Application"
    size: full
```

**Size Options:**
- `sm`: Small (400px) - Quick prompts, tips
- `md`: Medium (600px) - Standard dialogs *(default)*
- `lg`: Large (800px) - Forms, detailed content
- `xl`: Extra Large (1000px) - Complex interfaces
- `full`: Full screen - Immersive experiences

### Modal Content

Add rich content inside modals using nested statements.

```namel3ss
modal "info_modal":
    title: "Important Information"
    description: "Please read carefully"
    content:
        show text "First paragraph of content."
        show text "Second paragraph with more details."
        show text "⚠️ Warning: This action is irreversible."
```

**Supported Content:**
- `show text`: Text paragraphs with styling
- Additional components can be added as the system expands

### Modal Actions

Action buttons appear in the modal footer and handle user responses.

**Basic Actions:**
```namel3ss
modal "simple":
    title: "Simple Modal"
    actions:
        action "OK"
```

**Multiple Actions:**
```namel3ss
modal "confirm":
    title: "Confirm Action"
    actions:
        action "Cancel" variant "ghost"
        action "Confirm" variant "primary" action "do_confirm"
```

**Action Properties:**
- `label` (required): Button text
- `variant` (optional): Button style - `default`, `primary`, `destructive`, `ghost`, `link`
- `action` (optional): Action identifier to trigger on click
- `close` (optional): Whether to close modal after click (default: `true`)

### Action Variants

```namel3ss
modal "variants_demo":
    title: "Action Variants"
    actions:
        # Default - neutral gray button
        action "Default" variant "default"
        
        # Primary - blue, emphasizes main action
        action "Save" variant "primary" action "save"
        
        # Destructive - red, for dangerous actions
        action "Delete" variant "destructive" action "delete"
        
        # Ghost - transparent, secondary actions
        action "Cancel" variant "ghost"
        
        # Link - text-only button
        action "Learn More" variant "link" action "show_help"
```

### Non-Closing Actions

Keep the modal open after clicking an action (useful for validation, saving drafts).

```namel3ss
modal "form_modal":
    title: "Edit Profile"
    actions:
        action "Cancel" variant "ghost"
        action "Save Draft" variant "default" action "save_draft" close: false
        action "Save & Close" variant "primary" action "save_profile"
```

### Dismissible Control

Control whether users can close the modal by clicking outside or pressing ESC.

```namel3ss
# User can dismiss (default)
modal "dismissible":
    title: "Optional Action"
    dismissible: true

# User must use action buttons
modal "required":
    title: "Required Action"
    dismissible: false
    actions:
        action "Accept" variant "primary" action "accept"
```

### Trigger-Based Opening

Modals open in response to custom events.

```namel3ss
modal "triggered":
    title: "Event Triggered"
    trigger: "open_modal"
    actions:
        action "Close"
```

**Opening the modal from your code:**
```javascript
// Dispatch custom event to open modal
window.dispatchEvent(new CustomEvent('namel3ss:action', {
    detail: { action: 'open_modal' }
}));
```

### Complete Modal Example

```namel3ss
modal "complete_example":
    title: "Delete Account"
    description: "This action is permanent and cannot be undone"
    size: lg
    dismissible: true
    trigger: "show_delete_account"
    content:
        show text "⚠️ Warning: Deleting your account will:"
        show text "• Permanently remove all your data"
        show text "• Cancel any active subscriptions"
        show text "• Delete all uploaded files"
        show text "• Remove you from all shared projects"
    actions:
        action "Cancel" variant "ghost"
        action "I understand, delete my account" variant "destructive" action "delete_account"
```

---

## Toast Component

Toasts are non-blocking notifications that appear temporarily to provide feedback on actions, system status, or important information.

### Basic Toast

```namel3ss
toast "basic":
    title: "Notification"
```

### Toast Variants

Toasts use color-coded variants to convey different types of messages.

```namel3ss
# Default - neutral gray
toast "default":
    title: "General notification"
    variant: default

# Success - green with checkmark icon
toast "success":
    title: "Success!"
    description: "Operation completed successfully"
    variant: success

# Error - red with X icon
toast "error":
    title: "Error"
    description: "Something went wrong"
    variant: error

# Warning - yellow with alert icon
toast "warning":
    title: "Warning"
    description: "Please review this action"
    variant: warning

# Info - blue with info icon
toast "info":
    title: "Information"
    description: "Here's something you should know"
    variant: info
```

**Variant Icons:**
- `default`: No icon
- `success`: ✓ CheckCircle (green)
- `error`: ✗ XCircle (red)
- `warning`: ⚠ AlertCircle (yellow)
- `info`: ℹ Info (blue)

### Toast Duration

Control how long toasts remain visible (in milliseconds).

```namel3ss
# Quick notification (2 seconds)
toast "quick":
    title: "Quick message"
    duration: 2000

# Standard notification (3 seconds) - Default
toast "standard":
    title: "Standard message"
    duration: 3000

# Longer notification (5 seconds)
toast "longer":
    title: "Important message"
    duration: 5000

# Persistent toast (stays until manually dismissed)
toast "persistent":
    title: "Critical Alert"
    duration: 0
```

**Duration Guidelines:**
- `2000ms`: Quick confirmations (copied, saved)
- `3000ms`: Standard feedback *(default)*
- `4000-5000ms`: Important messages requiring attention
- `0`: Persistent - stays until user dismisses

### Toast Positioning

Place toasts in any corner or edge of the screen.

```namel3ss
# Top center
toast "top":
    title: "Top notification"
    position: top

# Top right (most common) - Default
toast "top_right":
    title: "Top right notification"
    position: top-right

# Top left
toast "top_left":
    title: "Top left notification"
    position: top-left

# Bottom center
toast "bottom":
    title: "Bottom notification"
    position: bottom

# Bottom right
toast "bottom_right":
    title: "Bottom right notification"
    position: bottom-right

# Bottom left
toast "bottom_left":
    title: "Bottom left notification"
    position: bottom-left
```

**Position Guidelines:**
- `top-right`: Default, least intrusive *(default)*
- `top`: Important system-wide messages
- `bottom-right`: Alternative if top is occupied
- `bottom`: Mobile-friendly, easy thumb access
- Corners: Consistent with app layout

### Toast Actions

Add an action button for user interaction.

```namel3ss
toast "with_action":
    title: "Item Deleted"
    description: "The file has been removed"
    variant: success
    action_label: "Undo"
    action: "undo_delete"
```

**Common Action Patterns:**
- **Undo**: Reverse destructive actions
- **View**: Navigate to related content
- **Retry**: Attempt failed operations again
- **Dismiss**: Explicitly acknowledge the message

### Trigger-Based Toasts

Toasts appear in response to custom events.

```namel3ss
toast "triggered":
    title: "Triggered Toast"
    trigger: "show_toast"
```

**Showing the toast from your code:**
```javascript
// Dispatch custom event to show toast
window.dispatchEvent(new CustomEvent('namel3ss:action', {
    detail: { action: 'show_toast' }
}));
```

### Complete Toast Example

```namel3ss
toast "complete_example":
    title: "Profile Updated"
    description: "Your changes have been saved successfully"
    variant: success
    duration: 4000
    position: top-right
    action_label: "View Profile"
    action: "navigate_to_profile"
    trigger: "show_profile_updated"
```

---

## Usage Patterns

### Confirmation Pattern

Use modals for destructive actions that need user confirmation.

```namel3ss
page "Items" at "/items":
    # Confirmation modal
    modal "confirm_delete":
        title: "Delete Item?"
        description: "This action cannot be undone"
        size: md
        trigger: "show_delete_confirm"
        content:
            show text "The item will be permanently removed from the database."
        actions:
            action "Cancel" variant "ghost"
            action "Delete" variant "destructive" action "delete_item"
    
    # Success notification
    toast "delete_success":
        title: "Item Deleted"
        description: "The item has been removed"
        variant: success
        duration: 3000
        action_label: "Undo"
        action: "undo_delete"
        trigger: "show_delete_success"
```

**Flow:**
1. User clicks delete button
2. Trigger `show_delete_confirm` event → Modal appears
3. User clicks "Delete" → Performs `delete_item` action
4. Backend confirms deletion → Trigger `show_delete_success` event → Toast appears

### Form Validation Pattern

Use modals for complex forms, toasts for validation feedback.

```namel3ss
page "Profile" at "/profile":
    modal "edit_profile":
        title: "Edit Profile"
        size: lg
        trigger: "show_edit_profile"
        content:
            show text "Update your information below"
        actions:
            action "Cancel" variant "ghost"
            action "Validate" variant "default" action "validate_profile" close: false
            action "Save" variant "primary" action "save_profile"
    
    # Validation error
    toast "validation_error":
        title: "Validation Failed"
        description: "Please check the highlighted fields"
        variant: error
        duration: 4000
        trigger: "show_validation_error"
    
    # Save success
    toast "save_success":
        title: "Profile Saved"
        variant: success
        duration: 3000
        trigger: "show_save_success"
```

### Multi-Step Process Pattern

Guide users through complex workflows with feedback at each step.

```namel3ss
page "Onboarding" at "/onboarding":
    # Welcome modal
    modal "welcome":
        title: "Welcome!"
        description: "Let's get you started"
        size: md
        trigger: "show_welcome"
        actions:
            action "Get Started" variant "primary" action "start_onboarding"
    
    # Progress toasts
    toast "step_1_complete":
        title: "Profile Created"
        variant: success
        duration: 2000
        trigger: "show_step_1_complete"
    
    toast "step_2_complete":
        title: "Preferences Saved"
        variant: success
        duration: 2000
        trigger: "show_step_2_complete"
    
    # Completion modal
    modal "complete":
        title: "All Set!"
        description: "You're ready to go"
        size: md
        trigger: "show_complete"
        actions:
            action "Go to Dashboard" variant "primary" action "navigate_dashboard"
```

### Error Handling Pattern

Provide clear feedback for errors with recovery options.

```namel3ss
page "Data" at "/data":
    # Connection error (persistent)
    toast "connection_error":
        title: "Connection Lost"
        description: "Unable to reach the server"
        variant: error
        duration: 0
        position: top
        action_label: "Retry"
        action: "retry_connection"
        trigger: "show_connection_error"
    
    # Partial failure modal
    modal "batch_error":
        title: "Some Items Failed"
        description: "3 of 10 items could not be processed"
        size: lg
        trigger: "show_batch_error"
        content:
            show text "Failed items:"
            show text "• item-001.pdf - File too large"
            show text "• item-005.jpg - Invalid format"
            show text "• item-008.doc - Permission denied"
        actions:
            action "Review Errors" variant "default" action "review_errors"
            action "Continue" variant "primary"
```

---

## Best Practices

### Modal Best Practices

**DO:**
- Use modals for important decisions that require user attention
- Keep modal titles clear and action-oriented
- Provide sufficient context in the description and content
- Use destructive variant for dangerous actions
- Include a clear way to cancel (ghost button or dismissible)
- Limit to 2-3 action buttons maximum

**DON'T:**
- Use modals for non-critical information (use toasts instead)
- Chain multiple modals in sequence (combine or use a wizard)
- Make modals non-dismissible without good reason
- Use cryptic action labels like "OK" for important decisions
- Overload with too much content (consider splitting into steps)

### Toast Best Practices

**DO:**
- Use toasts for transient feedback that doesn't require action
- Match variant to message type (success for success, error for errors)
- Keep titles short and descriptive (3-5 words)
- Use descriptions for additional context
- Position consistently across your app
- Set appropriate durations (longer for more important messages)

**DON'T:**
- Use toasts for critical errors (use modal with actions)
- Display multiple toasts simultaneously (queue them or stack)
- Use duration: 0 for routine messages (only for critical alerts)
- Make toast descriptions too long (2 lines max)
- Use toasts for information users need to reference later

### Accessibility Guidelines

**Modals:**
- Always include meaningful title and description
- Ensure keyboard navigation works (Tab, ESC)
- First action button receives focus when opened
- ESC key closes dismissible modals
- Focus returns to trigger element on close

**Toasts:**
- Use appropriate ARIA roles and labels
- Don't rely on color alone (include icons)
- Ensure sufficient contrast for text
- Allow keyboard users to dismiss persistent toasts
- Don't interrupt screen reader announcements

### Mobile Considerations

**Modals:**
- Use smaller sizes (`sm` or `md`) for mobile
- Stack action buttons vertically on narrow screens
- Ensure modal content scrolls on small devices
- Make backdrop tap-to-dismiss opt-in on mobile

**Toasts:**
- Use `bottom` or `top` positions (centered) for mobile
- Increase duration by 1-2 seconds for mobile users
- Make action buttons larger (easier to tap)
- Test toast stacking on various screen sizes

---

## Event Integration

### Triggering from Backend Actions

```javascript
// In your backend action response
window.dispatchEvent(new CustomEvent('namel3ss:action', {
    detail: { 
        action: 'show_success_toast',
        data: { itemName: 'Document.pdf' }
    }
}));
```

### Handling Action Events

```javascript
// Listen for modal/toast actions
window.addEventListener('namel3ss:action', (event) => {
    const { action, data } = event.detail;
    
    switch(action) {
        case 'delete_item':
            // Perform deletion
            // Then show success toast
            window.dispatchEvent(new CustomEvent('namel3ss:action', {
                detail: { action: 'show_delete_success' }
            }));
            break;
        
        case 'undo_delete':
            // Restore deleted item
            break;
    }
});
```

### Programmatic Control

```javascript
// Show modal programmatically
const modal = document.querySelector('[data-modal-id="confirm_delete"]');
if (modal) {
    modal.dispatchEvent(new CustomEvent('open'));
}

// Close modal programmatically
modal.dispatchEvent(new CustomEvent('close'));

// Show toast programmatically
const toast = document.querySelector('[data-toast-id="success"]');
if (toast) {
    toast.dispatchEvent(new CustomEvent('show'));
}
```

---

## Complete Examples

### E-commerce Checkout

```namel3ss
page "Checkout" at "/checkout":
    # Confirm order modal
    modal "confirm_order":
        title: "Review Your Order"
        description: "Please confirm your purchase"
        size: lg
        trigger: "show_confirm_order"
        content:
            show text "Total: $149.99"
            show text "Shipping: $9.99"
            show text "Tax: $15.00"
            show text "Grand Total: $174.98"
        actions:
            action "Edit Cart" variant "ghost" action "edit_cart"
            action "Place Order" variant "primary" action "place_order"
    
    # Order processing toast
    toast "processing_order":
        title: "Processing Order..."
        description: "Please wait while we confirm your payment"
        variant: info
        duration: 0
        trigger: "show_processing"
    
    # Success confirmation
    toast "order_confirmed":
        title: "Order Confirmed!"
        description: "You'll receive an email confirmation shortly"
        variant: success
        duration: 5000
        action_label: "View Order"
        action: "view_order_details"
        trigger: "show_order_confirmed"
    
    # Payment failed modal
    modal "payment_failed":
        title: "Payment Failed"
        description: "We couldn't process your payment"
        size: md
        dismissible: false
        trigger: "show_payment_failed"
        content:
            show text "Your card was declined."
            show text "Please check your payment information and try again."
        actions:
            action "Update Payment" variant "primary" action "update_payment"
            action "Cancel Order" variant "ghost" action "cancel_order"
```

### User Management Dashboard

```namel3ss
page "Users" at "/admin/users":
    # Bulk delete confirmation
    modal "bulk_delete":
        title: "Delete Multiple Users"
        description: "You are about to delete 12 user accounts"
        size: md
        trigger: "show_bulk_delete"
        content:
            show text "⚠️ This will permanently remove:"
            show text "• All user data and preferences"
            show text "• Associated files and documents"
            show text "• Activity history and logs"
        actions:
            action "Cancel" variant "ghost"
            action "Delete 12 Users" variant "destructive" action "confirm_bulk_delete"
    
    # Processing progress
    toast "bulk_progress":
        title: "Deleting Users..."
        description: "Processing 5 of 12"
        variant: info
        duration: 0
        trigger: "show_bulk_progress"
    
    # Completion notification
    toast "bulk_complete":
        title: "Users Deleted"
        description: "12 user accounts have been removed"
        variant: success
        duration: 4000
        action_label: "Undo"
        action: "undo_bulk_delete"
        trigger: "show_bulk_complete"
    
    # Individual disable
    toast "user_disabled":
        title: "User Disabled"
        variant: warning
        duration: 3000
        action_label: "Re-enable"
        action: "reenable_user"
        trigger: "show_user_disabled"
```

---

## Styling & Customization

### Modal Styling

Modals use Tailwind CSS classes and can be customized:

```typescript
// Modal sizes (defined in Modal.tsx)
const sizeClasses = {
    sm: 'max-w-md',      // 448px
    md: 'max-w-2xl',     // 672px
    lg: 'max-w-4xl',     // 896px
    xl: 'max-w-6xl',     // 1152px
    full: 'max-w-full'   // Full width with padding
};

// Action button variants
const actionVariants = {
    default: 'bg-gray-100 hover:bg-gray-200 text-gray-900',
    primary: 'bg-blue-600 hover:bg-blue-700 text-white',
    destructive: 'bg-red-600 hover:bg-red-700 text-white',
    ghost: 'hover:bg-gray-100 text-gray-700',
    link: 'text-blue-600 hover:underline'
};
```

### Toast Styling

Toast variants with icons and colors:

```typescript
// Toast variant styles (defined in Toast.tsx)
const variantStyles = {
    default: {
        bg: 'bg-white',
        text: 'text-gray-900',
        border: 'border-gray-200',
        icon: null
    },
    success: {
        bg: 'bg-green-50',
        text: 'text-green-900',
        border: 'border-green-200',
        icon: 'CheckCircle'
    },
    error: {
        bg: 'bg-red-50',
        text: 'text-red-900',
        border: 'border-red-200',
        icon: 'XCircle'
    },
    warning: {
        bg: 'bg-yellow-50',
        text: 'text-yellow-900',
        border: 'border-yellow-200',
        icon: 'AlertCircle'
    },
    info: {
        bg: 'bg-blue-50',
        text: 'text-blue-900',
        border: 'border-blue-200',
        icon: 'Info'
    }
};
```

---

## API Reference

### Modal Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `id` | string | required | Unique identifier for the modal |
| `title` | string | required | Main heading text |
| `description` | string | optional | Subtitle or context |
| `size` | `sm\|md\|lg\|xl\|full` | `md` | Modal width |
| `dismissible` | boolean | `true` | Allow ESC/backdrop close |
| `trigger` | string | optional | Event name to open modal |
| `content` | Statement[] | `[]` | Nested content blocks |
| `actions` | ModalAction[] | `[]` | Action buttons |

### ModalAction Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `label` | string | required | Button text |
| `action` | string | optional | Action identifier |
| `variant` | `default\|primary\|destructive\|ghost\|link` | `default` | Button style |
| `close` | boolean | `true` | Close modal on click |

### Toast Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `id` | string | required | Unique identifier |
| `title` | string | required | Main notification text |
| `description` | string | optional | Additional details |
| `variant` | `default\|success\|error\|warning\|info` | `default` | Visual style with icon |
| `duration` | number | `3000` | Display time (ms), 0 = persistent |
| `position` | `top\|top-right\|top-left\|bottom\|bottom-right\|bottom-left` | `top-right` | Screen position |
| `action_label` | string | optional | Action button text |
| `action` | string | optional | Action identifier |
| `trigger` | string | optional | Event name to show toast |

---

## Migration Guide

### From Alert/Confirm Dialogs

**Before (browser alerts):**
```javascript
if (confirm('Delete this item?')) {
    deleteItem();
}
alert('Item deleted!');
```

**After (namel3ss feedback components):**
```namel3ss
modal "confirm_delete":
    title: "Delete Item?"
    trigger: "show_delete_confirm"
    actions:
        action "Cancel" variant "ghost"
        action "Delete" variant "destructive" action "delete_item"

toast "delete_success":
    title: "Item Deleted"
    variant: success
    trigger: "show_delete_success"
```

### From Custom Modals

Replace custom modal implementations with declarative syntax for consistency and maintainability.

---

## Troubleshooting

### Modal Not Appearing

**Check:**
1. Modal has a `trigger` property
2. Trigger event is being dispatched correctly
3. Modal ID is unique on the page
4. No JavaScript errors in console

**Debug:**
```javascript
// Test trigger manually
window.dispatchEvent(new CustomEvent('namel3ss:action', {
    detail: { action: 'your_trigger_name' }
}));
```

### Toast Not Dismissing

**Check:**
1. Duration is not set to `0` (persistent)
2. Toast component is receiving proper props
3. Auto-dismiss timer is not being cleared

**Solution:**
```namel3ss
# Ensure duration is set correctly
toast "auto_dismiss":
    title: "Message"
    duration: 3000  # Will auto-dismiss after 3 seconds
```

### Action Not Firing

**Check:**
1. Action name matches event listener
2. Event is propagating correctly
3. No errors in action handler

**Debug:**
```javascript
window.addEventListener('namel3ss:action', (event) => {
    console.log('Action received:', event.detail);
});
```

---

## See Also

- [Chrome Components Guide](./CHROME_COMPONENTS_GUIDE.md) - Navigation components
- [Examples](../examples/feedback_demo.ai) - Complete demo application
- [API Reference](./API_REFERENCE.md) - Full API documentation
