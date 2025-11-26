# Forms and User Input - Complete Reference

## Overview

Namel3ss provides a powerful, declarative form system with 11 semantic field types, comprehensive validation, conditional rendering, and seamless backend integration. Forms are defined using the `show form` statement within pages and automatically generate:

- **Frontend**: React components with validation, state management, and user feedback
- **Backend**: FastAPI endpoints with server-side validation and JSON Schema
- **Type Safety**: Full TypeScript interfaces and Python type hints

## Quick Start

```namel3ss
app "Contact Manager"

page "NewContact" at "/new":
  show form "Add Contact":
    fields:
      - name: full_name
        component: text_input
        label: "Full Name"
        required: true
        min_length: 2
      
      - name: email
        component: text_input
        label: "Email Address"
        required: true
        pattern: "^[^@]+@[^@]+$"
      
      - name: phone
        component: text_input
        label: "Phone Number"
        placeholder: "(555) 555-5555"
      
      - name: notes
        component: textarea
        label: "Notes"
        max_length: 500
    
    submit_button_text: "Add Contact"
    layout: vertical
```

## Form Syntax

### Basic Structure

```namel3ss
show form "Form Title":
  fields:
    - name: field_name
      component: field_type
      label: "Field Label"
      # Field-specific properties...
  
  # Form configuration
  layout: vertical | horizontal
  submit_button_text: "Submit"
  validation_mode: on_blur | on_change | on_submit
  loading_text: "Processing..."
  success_message: "Success!"
  error_message: "Error occurred"
  reset_button: true | false
```

### Field Properties

All fields support these common properties:

| Property | Type | Description |
|----------|------|-------------|
| `name` | string | **Required.** Unique field identifier |
| `component` | string | **Required.** Field type (see Field Types) |
| `label` | string | Display label for the field |
| `placeholder` | string | Placeholder text |
| `help_text` | string | Helper text below field |
| `required` | boolean | Whether field is required |
| `default` | any | Default value |
| `initial_value` | any | Initial value (overrides default) |
| `visible` | expression | Conditional visibility |
| `disabled` | boolean/expression | Whether field is disabled |

## Field Types

### 1. Text Input (`text_input`)

Single-line text input with validation support.

```namel3ss
- name: username
  component: text_input
  label: "Username"
  required: true
  min_length: 3
  max_length: 20
  pattern: "^[a-zA-Z0-9_]+$"
  placeholder: "Enter username"
```

**Validation Properties:**
- `min_length`: Minimum character count
- `max_length`: Maximum character count
- `pattern`: Regex pattern (string)

### 2. Text Area (`textarea`)

Multi-line text input.

```namel3ss
- name: description
  component: textarea
  label: "Description"
  max_length: 1000
  placeholder: "Enter description..."
  help_text: "Maximum 1000 characters"
```

**Validation Properties:**
- `min_length`: Minimum character count
- `max_length`: Maximum character count

### 3. Select (`select`)

Single-choice dropdown.

```namel3ss
- name: country
  component: select
  label: "Country"
  required: true
  options: ["USA", "Canada", "UK", "Germany", "France"]
```

**Configuration:**
- `options`: Array of string options (static)
- `options_binding`: Dataset binding for dynamic options

### 4. Multi-Select (`multiselect`)

Multiple-choice selection.

```namel3ss
- name: interests
  component: multiselect
  label: "Interests"
  options: ["Sports", "Music", "Art", "Technology", "Travel"]
```

**Configuration:**
- `options`: Array of string options
- `options_binding`: Dataset binding for dynamic options

### 5. Checkbox (`checkbox`)

Boolean on/off toggle.

```namel3ss
- name: agree_terms
  component: checkbox
  label: "I agree to the terms and conditions"
  required: true

- name: newsletter
  component: checkbox
  label: "Subscribe to newsletter"
  default: true
```

### 6. Switch (`switch`)

Visual toggle switch (alternative to checkbox).

```namel3ss
- name: notifications_enabled
  component: switch
  label: "Enable Notifications"
  default: true
```

### 7. Radio Group (`radio_group`)

Single choice from multiple options (displayed as radio buttons).

```namel3ss
- name: size
  component: radio_group
  label: "T-Shirt Size"
  options: ["Small", "Medium", "Large", "XL"]
  default: "Medium"
  required: true
```

### 8. Slider (`slider`)

Numeric input with visual slider.

```namel3ss
- name: volume
  component: slider
  label: "Volume"
  min_value: 0
  max_value: 100
  step: 5
  default: 50
  required: true
```

**Validation Properties:**
- `min_value`: Minimum numeric value
- `max_value`: Maximum numeric value
- `step`: Increment step

### 9. Date Picker (`date_picker`)

Date selection (calendar widget).

```namel3ss
- name: birth_date
  component: date_picker
  label: "Date of Birth"
  required: true
```

### 10. DateTime Picker (`datetime_picker`)

Date and time selection.

```namel3ss
- name: appointment
  component: datetime_picker
  label: "Appointment Time"
  required: true
```

### 11. File Upload (`file_upload`)

File selection and upload.

```namel3ss
- name: profile_picture
  component: file_upload
  label: "Profile Picture"
  accept: "image/*"
  max_file_size: 5242880  # 5MB in bytes
  multiple: false

- name: documents
  component: file_upload
  label: "Documents"
  accept: ".pdf,.doc,.docx"
  multiple: true
```

**Configuration:**
- `accept`: MIME types or file extensions
- `max_file_size`: Maximum file size in bytes
- `multiple`: Allow multiple files
- `upload_endpoint`: Custom upload endpoint (optional)

## Validation

### Validation Modes

Control when validation occurs:

```namel3ss
validation_mode: on_blur    # Validate when field loses focus (default)
validation_mode: on_change  # Validate on every keystroke
validation_mode: on_submit  # Validate only on form submission
```

### Validation Rules

#### Text Validation

```namel3ss
- name: email
  component: text_input
  required: true
  min_length: 5
  max_length: 100
  pattern: "^[^@]+@[^@]+\\.[^@]+$"
```

#### Numeric Validation

```namel3ss
- name: age
  component: slider
  required: true
  min_value: 18
  max_value: 120
  step: 1
```

### Custom Error Messages

```namel3ss
show form "Registration":
  fields:
    - name: email
      component: text_input
      label: "Email"
      required: true
  
  error_message: "Registration failed. Please check your input."
  success_message: "Registration successful! Welcome aboard."
```

## Conditional Rendering

### Visible Expressions

Show/hide fields based on other field values:

```namel3ss
show form "Pet Survey":
  fields:
    - name: has_pet
      component: checkbox
      label: "Do you have a pet?"
    
    - name: pet_type
      component: select
      label: "Pet Type"
      options: ["Dog", "Cat", "Bird", "Other"]
      visible: has_pet  # Only shown when has_pet is checked
    
    - name: pet_name
      component: text_input
      label: "Pet Name"
      visible: has_pet
```

### Disabled Expressions

```namel3ss
- name: save_preferences
  component: checkbox
  label: "Save my preferences"
  disabled: false  # Can use boolean or expression
```

## Layout Modes

### Vertical Layout (Default)

Fields stacked vertically - ideal for most forms.

```namel3ss
show form "Contact Form":
  fields:
    - name: name
      component: text_input
      label: "Name"
    - name: email
      component: text_input
      label: "Email"
  layout: vertical
```

### Horizontal Layout

Fields arranged horizontally - good for compact forms or search bars.

```namel3ss
show form "Search":
  fields:
    - name: query
      component: text_input
      label: "Search"
      placeholder: "Type to search..."
  layout: horizontal
  submit_button_text: "Search"
```

## Multiple Forms on a Page

Place multiple independent forms on the same page:

```namel3ss
page "Dashboard" at "/":
  show form "Quick Search":
    fields:
      - name: query
        component: text_input
        label: "Search"
    layout: horizontal
  
  show form "Filters":
    fields:
      - name: category
        component: select
        label: "Category"
        options: ["All", "Active", "Archived"]
      - name: date_range
        component: select
        label: "Date Range"
        options: ["Today", "This Week", "This Month"]
    layout: vertical
```

## Advanced Features

### Dynamic Options with Data Binding

Load options from a dataset:

```namel3ss
dataset "Countries":
  source: "sql"
  query: "SELECT name, code FROM countries ORDER BY name"

page "Signup" at "/signup":
  show form "Registration":
    fields:
      - name: country
        component: select
        label: "Country"
        options_binding: Countries  # Load from dataset
```

### Form Submission Actions

Integrate with backend actions (coming in future release):

```namel3ss
show form "Contact":
  fields:
    - name: email
      component: text_input
      label: "Email"
  
  submit_action: create_contact  # Reference to action
  success_message: "Contact created successfully!"
```

### File Upload with Custom Endpoint

```namel3ss
- name: avatar
  component: file_upload
  label: "Profile Picture"
  accept: "image/*"
  max_file_size: 5242880
  upload_endpoint: "/api/uploads/avatar"
```

## Complete Example

```namel3ss
app "User Management"

page "NewUser" at "/users/new":
  show form "Create User":
    fields:
      # Basic Info
      - name: username
        component: text_input
        label: "Username"
        required: true
        min_length: 3
        max_length: 20
        pattern: "^[a-zA-Z0-9_]+$"
        placeholder: "johndoe"
      
      - name: email
        component: text_input
        label: "Email"
        required: true
        pattern: "^[^@]+@[^@]+\\.[^@]+$"
      
      - name: full_name
        component: text_input
        label: "Full Name"
        required: true
        min_length: 2
      
      # Profile
      - name: bio
        component: textarea
        label: "Biography"
        max_length: 500
        help_text: "Tell us about yourself (max 500 characters)"
      
      - name: country
        component: select
        label: "Country"
        required: true
        options: ["USA", "Canada", "UK", "Germany", "France"]
      
      - name: interests
        component: multiselect
        label: "Interests"
        options: ["Technology", "Sports", "Music", "Travel", "Reading"]
      
      # Preferences
      - name: newsletter
        component: checkbox
        label: "Subscribe to newsletter"
        default: true
      
      - name: notifications
        component: switch
        label: "Enable notifications"
        default: true
      
      # Account Type
      - name: account_type
        component: radio_group
        label: "Account Type"
        options: ["Free", "Pro", "Enterprise"]
        default: "Free"
        required: true
      
      # Additional
      - name: age
        component: slider
        label: "Age"
        min_value: 13
        max_value: 120
        step: 1
        required: true
      
      - name: start_date
        component: date_picker
        label: "Start Date"
        required: true
      
      - name: profile_picture
        component: file_upload
        label: "Profile Picture"
        accept: "image/*"
        max_file_size: 5242880
    
    # Form Configuration
    layout: vertical
    validation_mode: on_blur
    submit_button_text: "Create User"
    reset_button: true
    loading_text: "Creating user..."
    success_message: "User created successfully!"
    error_message: "Failed to create user. Please check your input."
```

## Generated Output

### Frontend (React)

The form automatically generates:
- **FormWidget Component**: Comprehensive React component with all field types
- **Validation Logic**: Client-side validation with real-time feedback
- **State Management**: Form data, errors, and touched state
- **Event Handlers**: onChange, onBlur, onSubmit with proper validation
- **Type Safety**: Full TypeScript interfaces

### Backend (FastAPI)

The form automatically generates:
- **Form Endpoint**: POST endpoint for form submission
- **JSON Schema Validation**: Server-side validation matching frontend
- **Type Hints**: Python type annotations for form data
- **Error Handling**: Structured error responses

## Best Practices

### 1. Field Naming

Use clear, descriptive names:
```namel3ss
# Good
- name: email_address
- name: phone_number
- name: birth_date

# Avoid
- name: field1
- name: input
- name: data
```

### 2. Validation

Always validate on both client and server:
```namel3ss
- name: email
  component: text_input
  required: true              # Client validation
  pattern: "^[^@]+@[^@]+$"   # Regex validation
```

### 3. User Feedback

Provide clear labels and help text:
```namel3ss
- name: password
  component: text_input
  label: "Password"
  placeholder: "Enter password"
  help_text: "Must be at least 8 characters with one number"
  min_length: 8
```

### 4. Required Fields

Mark required fields explicitly:
```namel3ss
- name: email
  component: text_input
  label: "Email"
  required: true  # Clear indication
```

### 5. Appropriate Field Types

Choose the right component for the data:
```namel3ss
# Date data - use date_picker
- name: birth_date
  component: date_picker

# Boolean - use checkbox or switch
- name: agree_terms
  component: checkbox

# Numeric range - use slider
- name: rating
  component: slider
  min_value: 1
  max_value: 5
```

## Migration from Legacy Syntax

If you're using the older `field_type` syntax, here's how to migrate:

### Old Syntax
```namel3ss
show form "Old Form":
  fields:
    - name: email
      field_type: email  # Legacy
```

### New Syntax
```namel3ss
show form "New Form":
  fields:
    - name: email
      component: text_input  # Modern
      pattern: "^[^@]+@[^@]+$"
```

The old syntax is still supported for backward compatibility but the new `component` syntax is recommended for all new forms.

## Troubleshooting

### Form Not Rendering

Check that:
1. Form is inside a `page` block
2. All required fields have `name` and `component`
3. Field names are unique

### Validation Not Working

Verify:
1. `validation_mode` is set appropriately
2. Validation rules are compatible with field type
3. Regex patterns are properly escaped

### Options Not Showing

Ensure:
1. `options` is an array of strings
2. For dynamic options, `options_binding` references an existing dataset

## Summary

Namel3ss forms provide:
- ✅ **11 semantic field types** covering all input scenarios
- ✅ **Comprehensive validation** with client and server-side support
- ✅ **Conditional rendering** for dynamic forms
- ✅ **Multiple layout modes** for flexibility
- ✅ **File upload** with size and type restrictions
- ✅ **Type safety** with full TypeScript and Python types
- ✅ **Auto-generation** of frontend and backend code

For more examples, see `examples/forms_demo.ai` in the repository.
