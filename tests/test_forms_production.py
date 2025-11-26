"""Tests for production-grade form parsing and IR generation."""

import pytest
from namel3ss.parser import Parser
from namel3ss.ast.pages import ShowForm, FormField
from namel3ss.ir.builder import build_frontend_ir
from namel3ss.ir.spec import IRForm, IRFormField


def test_parse_basic_form():
    """Test parsing a basic form with declarative field syntax."""
    source = '''
app "Test Forms"

page "Contact" at "/contact":
  show form "Contact Us":
    fields:
      - name: email
        component: text_input
        label: "Email Address"
        required: true
      - name: message
        component: textarea
        label: "Message"
    layout: vertical
    submit_action: send_message
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Verify page exists
    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.name == "Contact"
    
    # Verify form exists
    assert len(page.body) == 1
    form = page.body[0]
    assert isinstance(form, ShowForm)
    assert form.title == "Contact Us"
    assert form.layout_mode == "vertical"
    assert form.submit_action == "send_message"
    
    # Verify fields
    assert len(form.fields) == 2
    
    email_field = form.fields[0]
    assert email_field.name == "email"
    assert email_field.component == "text_input"
    assert email_field.label == "Email Address"
    assert email_field.required == True
    
    message_field = form.fields[1]
    assert message_field.name == "message"
    assert message_field.component == "textarea"
    assert message_field.label == "Message"


def test_parse_form_with_validation():
    """Test parsing form fields with validation rules."""
    source = '''
app "Validation Test"

page "Register" at "/register":
  show form "Registration":
    fields:
      - name: username
        component: text_input
        required: true
        min_length: 3
        max_length: 20
        pattern: "^[a-zA-Z0-9_]+$"
      - name: age
        component: slider
        min_value: 18
        max_value: 100
        step: 1
    submit_action: register_user
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    form = app.pages[0].body[0]
    
    # Check username field validation
    username = form.fields[0]
    assert username.min_length == 3
    assert username.max_length == 20
    assert username.pattern == "^[a-zA-Z0-9_]+$"
    
    # Check age field validation
    age = form.fields[1]
    assert age.min_value == 18
    assert age.max_value == 100
    assert age.step == 1


def test_parse_form_with_select_options():
    """Test parsing select field with options binding."""
    source = '''
app "Select Test"

page "Profile" at "/profile":
  show form "Edit Profile":
    fields:
      - name: role
        component: select
        label: "Role"
        options_binding: datasets.roles
      - name: tags
        component: multiselect
        label: "Tags"
        options_binding: datasets.tags
        multiple: true
    initial_values_binding: user_profile
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    form = app.pages[0].body[0]
    
    # Check role field
    role = form.fields[0]
    assert role.component == "select"
    assert role.options_binding == "datasets.roles"
    
    # Check tags field
    tags = form.fields[1]
    assert tags.component == "multiselect"
    assert tags.options_binding == "datasets.tags"
    assert tags.multiple == True
    
    # Check initial values binding
    assert form.initial_values_binding == "user_profile"


def test_parse_form_with_file_upload():
    """Test parsing form with file upload field."""
    source = '''
app "Upload Test"

page "Documents" at "/documents":
  show form "Upload Document":
    fields:
      - name: document
        component: file_upload
        label: "Document"
        accept: "application/pdf,application/msword"
        max_file_size: 5242880
        upload_endpoint: "/api/upload"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    form = app.pages[0].body[0]
    
    doc_field = form.fields[0]
    assert doc_field.component == "file_upload"
    assert doc_field.accept == "application/pdf,application/msword"
    assert doc_field.max_file_size == 5242880
    assert doc_field.upload_endpoint == "/api/upload"


def test_parse_form_with_conditional_fields():
    """Test parsing form with conditional rendering."""
    source = '''
app "Conditional Test"

page "Application" at "/apply":
  show form "Job Application":
    fields:
      - name: has_experience
        component: checkbox
        label: "I have experience"
      - name: years_experience
        component: text_input
        label: "Years of Experience"
        visible: has_experience
        disabled: false
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    form = app.pages[0].body[0]
    
    years_field = form.fields[1]
    assert years_field.visible == "has_experience"  # Expression string
    assert years_field.disabled == False  # Boolean value


def test_parse_form_with_all_field_types():
    """Test parsing form with all supported field types."""
    source = '''
app "Comprehensive Form"

page "Demo" at "/demo":
  show form "All Fields":
    fields:
      - { name: text, component: text_input, label: "Text" }
      - { name: textarea, component: textarea, label: "Textarea" }
      - { name: select, component: select, label: "Select" }
      - { name: multiselect, component: multiselect, label: "Multi-Select" }
      - { name: checkbox, component: checkbox, label: "Checkbox" }
      - { name: switch, component: switch, label: "Switch" }
      - { name: radio, component: radio_group, label: "Radio" }
      - { name: slider, component: slider, label: "Slider" }
      - { name: date, component: date_picker, label: "Date" }
      - { name: datetime, component: datetime_picker, label: "DateTime" }
      - { name: file, component: file_upload, label: "File" }
    layout: horizontal
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    form = app.pages[0].body[0]
    
    assert len(form.fields) == 11
    assert form.layout_mode == "horizontal"
    
    field_components = [f.component for f in form.fields]
    expected_components = [
        "text_input", "textarea", "select", "multiselect", 
        "checkbox", "switch", "radio_group", "slider",
        "date_picker", "datetime_picker", "file_upload"
    ]
    assert field_components == expected_components


def test_legacy_form_syntax_compatibility():
    """Test backward compatibility with legacy form syntax."""
    source = '''
app "Legacy"

page "Contact" at "/contact":
  show form "Contact":
    fields: name, email, message
    on submit:
      show toast "Thank you"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    form = app.pages[0].body[0]
    
    # Should parse legacy syntax
    assert len(form.fields) == 3
    assert form.fields[0].name == "name"
    assert form.fields[1].name == "email"
    assert form.fields[2].name == "message"
    assert len(form.on_submit_ops) > 0


def test_form_ir_generation():
    """Test IR generation from form AST."""
    source = '''
app "IR Test"

dataset "users" from "db://users"

page "Register" at "/register":
  show form "User Registration":
    fields:
      - name: email
        component: text_input
        label: "Email"
        required: true
        pattern: "^[^@]+@[^@]+\\.[^@]+$"
      - name: password
        component: text_input
        label: "Password"
        required: true
        min_length: 8
    layout: vertical
    submit_action: create_user
    validation_mode: on_blur
    submit_button_text: "Create Account"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Build IR
    from namel3ss.ir.builder import _build_ir_form_from_ast, _show_form_to_component
    from namel3ss.codegen.backend.state import build_backend_state
    
    state = build_backend_state(app)
    form_ast = app.pages[0].body[0]
    
    # Generate IR
    ir_form = _build_ir_form_from_ast(form_ast, state)
    
    # Verify IR structure
    assert isinstance(ir_form, IRForm)
    assert ir_form.title == "User Registration"
    assert ir_form.layout_mode == "vertical"
    assert ir_form.submit_action == "create_user"
    assert ir_form.validation_mode == "on_blur"
    assert ir_form.submit_button_text == "Create Account"
    
    # Verify fields
    assert len(ir_form.fields) == 2
    
    email_field = ir_form.fields[0]
    assert isinstance(email_field, IRFormField)
    assert email_field.name == "email"
    assert email_field.component == "text_input"
    assert email_field.required == True
    assert "pattern" in email_field.validation
    
    password_field = ir_form.fields[1]
    assert password_field.name == "password"
    assert password_field.validation.get("min_length") == 8
    
    # Verify validation schema generation
    assert "properties" in ir_form.validation_schema
    assert "email" in ir_form.validation_schema["properties"]
    assert "password" in ir_form.validation_schema["properties"]
    assert "required" in ir_form.validation_schema
    assert "email" in ir_form.validation_schema["required"]
    assert "password" in ir_form.validation_schema["required"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
