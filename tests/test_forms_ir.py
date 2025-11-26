"""Tests for form IR generation and conversion."""

import pytest
from namel3ss.parser import Parser
from namel3ss.ir.builder import build_frontend_ir
from namel3ss.ir.spec import IRForm, IRFormField


def test_ir_form_field_generation():
    """Test IRFormField generation from AST."""
    source = '''
app "IR Test"

page "Form" at "/form":
  show form "Test Form":
    fields:
      - name: email
        component: text_input
        label: "Email"
        required: true
        min_length: 5
        max_length: 100
        pattern: "^[^@]+@[^@]+$"
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Build frontend IR
    frontend_ir = build_frontend_ir(app)
    
    # Verify page exists
    assert len(frontend_ir.pages) == 1
    page = frontend_ir.pages[0]
    assert page.name == "Form"
    
    # Verify form component
    assert len(page.components) == 1
    component = page.components[0]
    assert component.type == "form"
    
    # Extract IRForm from component props
    form_spec = component.props.get("form_spec")
    assert form_spec is not None
    assert isinstance(form_spec, IRForm)
    
    # Verify form properties
    assert form_spec.title == "Test Form"
    assert form_spec.layout_mode == "vertical"
    assert len(form_spec.fields) == 1
    
    # Verify field
    field = form_spec.fields[0]
    assert isinstance(field, IRFormField)
    assert field.name == "email"
    assert field.component == "text_input"
    assert field.label == "Email"
    assert field.required == True
    
    # Verify validation
    assert field.validation["min_length"] == 5
    assert field.validation["max_length"] == 100
    assert field.validation["pattern"] == "^[^@]+@[^@]+$"


def test_ir_validation_schema_generation():
    """Test JSON Schema generation from form fields."""
    source = '''
app "Schema Test"

page "Form" at "/form":
  show form "Registration":
    fields:
      - name: username
        component: text_input
        required: true
        min_length: 3
        max_length: 20
      - name: age
        component: slider
        required: true
        min_value: 18
        max_value: 120
      - name: terms
        component: checkbox
        required: true
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props.get("form_spec")
    
    # Verify validation schema
    schema = form_spec.validation_schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema
    
    # Check username validation
    username_schema = schema["properties"]["username"]
    assert username_schema["type"] == "string"
    assert username_schema["minLength"] == 3
    assert username_schema["maxLength"] == 20
    
    # Check age validation
    age_schema = schema["properties"]["age"]
    assert age_schema["type"] == "number"
    assert age_schema["minimum"] == 18
    assert age_schema["maximum"] == 120
    
    # Check terms validation
    terms_schema = schema["properties"]["terms"]
    assert terms_schema["type"] == "boolean"
    
    # Check required fields
    assert set(schema["required"]) == {"username", "age", "terms"}


def test_ir_form_with_select_options():
    """Test IRFormField with select options."""
    source = '''
app "Options Test"

page "Form" at "/form":
  show form "Preferences":
    fields:
      - name: country
        component: select
        label: "Country"
        options: ["USA", "Canada", "UK"]
      - name: interests
        component: multiselect
        label: "Interests"
        options: ["Sports", "Music", "Art"]
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props.get("form_spec")
    
    # Verify country field
    country_field = form_spec.fields[0]
    assert country_field.name == "country"
    assert country_field.component == "select"
    assert country_field.static_options == ["USA", "Canada", "UK"]
    
    # Verify interests field
    interests_field = form_spec.fields[1]
    assert interests_field.name == "interests"
    assert interests_field.component == "multiselect"
    assert interests_field.static_options == ["Sports", "Music", "Art"]


def test_ir_form_with_file_upload():
    """Test IRFormField with file upload configuration."""
    source = '''
app "Upload Test"

page "Form" at "/form":
  show form "File Upload":
    fields:
      - name: avatar
        component: file_upload
        label: "Profile Picture"
        accept: "image/*"
        max_file_size: 5242880
        upload_endpoint: "/api/upload"
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props.get("form_spec")
    
    # Verify file upload field
    field = form_spec.fields[0]
    assert field.name == "avatar"
    assert field.component == "file_upload"
    assert field.component_config["accept"] == "image/*"
    assert field.component_config["max_file_size"] == 5242880
    assert field.component_config["upload_endpoint"] == "/api/upload"


def test_ir_form_with_conditional_fields():
    """Test IRFormField with conditional rendering."""
    source = '''
app "Conditional Test"

page "Form" at "/form":
  show form "Dynamic Form":
    fields:
      - name: has_pet
        component: checkbox
        label: "Do you have a pet?"
      - name: pet_name
        component: text_input
        label: "Pet Name"
        visible: has_pet
        disabled: false
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props.get("form_spec")
    
    # Verify conditional field
    pet_name_field = form_spec.fields[1]
    assert pet_name_field.name == "pet_name"
    assert pet_name_field.visible_expr == "has_pet"
    assert pet_name_field.disabled_expr == "False"


def test_ir_form_with_submit_action():
    """Test IRForm with submit action."""
    source = '''
app "Action Test"

page "Form" at "/form":
  show form "Contact":
    fields:
      - name: message
        component: textarea
        label: "Message"
    layout: vertical
    submit_action: send_message
    submit_button_text: "Send"
    success_message: "Message sent!"
    error_message: "Failed to send"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props.get("form_spec")
    
    # Verify form action properties
    assert form_spec.submit_action == "send_message"
    assert form_spec.submit_button_text == "Send"
    assert form_spec.success_message == "Message sent!"
    assert form_spec.error_message == "Failed to send"


def test_ir_form_validation_mode():
    """Test IRForm validation modes."""
    source = '''
app "Validation Test"

page "Form" at "/form":
  show form "Settings":
    fields:
      - name: setting
        component: text_input
    layout: vertical
    validation_mode: on_change
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props.get("form_spec")
    
    assert form_spec.validation_mode == "on_change"


def test_ir_form_layout_modes():
    """Test IRForm layout modes."""
    source = '''
app "Layout Test"

page "Horizontal" at "/h":
  show form "Form":
    fields:
      - name: field1
        component: text_input
      - name: field2
        component: text_input
    layout: horizontal

page "Vertical" at "/v":
  show form "Form":
    fields:
      - name: field1
        component: text_input
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    
    # Check horizontal layout
    horizontal_form = frontend_ir.pages[0].components[0].props.get("form_spec")
    assert horizontal_form.layout_mode == "horizontal"
    
    # Check vertical layout
    vertical_form = frontend_ir.pages[1].components[0].props.get("form_spec")
    assert vertical_form.layout_mode == "vertical"


def test_ir_multiple_forms_on_page():
    """Test multiple forms on a single page."""
    source = '''
app "Multi Form Test"

page "Forms" at "/forms":
  show form "Form 1":
    fields:
      - name: field1
        component: text_input
    layout: vertical
  
  show form "Form 2":
    fields:
      - name: field2
        component: text_input
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    
    # Verify both forms exist
    page = frontend_ir.pages[0]
    assert len(page.components) == 2
    
    form1 = page.components[0].props.get("form_spec")
    form2 = page.components[1].props.get("form_spec")
    
    assert form1.title == "Form 1"
    assert form2.title == "Form 2"
    assert form1.fields[0].name == "field1"
    assert form2.fields[0].name == "field2"


def test_ir_all_field_types():
    """Test IRFormField generation for all 11 field types."""
    source = '''
app "All Fields Test"

page "Form" at "/form":
  show form "Complete Form":
    fields:
      - name: text
        component: text_input
      - name: textarea
        component: textarea
      - name: select
        component: select
      - name: multiselect
        component: multiselect
      - name: checkbox
        component: checkbox
      - name: switch
        component: switch
      - name: radio
        component: radio_group
      - name: slider
        component: slider
      - name: date
        component: date_picker
      - name: datetime
        component: datetime_picker
      - name: file
        component: file_upload
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props.get("form_spec")
    
    # Verify all field types
    expected_components = [
        "text_input", "textarea", "select", "multiselect",
        "checkbox", "switch", "radio_group", "slider",
        "date_picker", "datetime_picker", "file_upload"
    ]
    
    assert len(form_spec.fields) == 11
    actual_components = [field.component for field in form_spec.fields]
    assert actual_components == expected_components
    
    # Verify JSON types in validation schema
    schema = form_spec.validation_schema
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["checkbox"]["type"] == "boolean"
    assert schema["properties"]["slider"]["type"] == "number"
    assert schema["properties"]["multiselect"]["type"] == "array"
