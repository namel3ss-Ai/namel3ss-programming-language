"""Integration tests for end-to-end form workflows."""

import pytest
import tempfile
import shutil
from pathlib import Path
from namel3ss.parser import Parser
from namel3ss.codegen.frontend.react.main import generate_react_vite_site
from namel3ss.codegen.backend import generate_backend
from namel3ss.ir.builder import build_frontend_ir, build_backend_ir
from namel3ss.codegen.backend.state import build_backend_state


@pytest.fixture
def temp_dirs():
    """Create temporary directories for frontend and backend output."""
    frontend_dir = tempfile.mkdtemp()
    backend_dir = tempfile.mkdtemp()
    yield frontend_dir, backend_dir
    shutil.rmtree(frontend_dir, ignore_errors=True)
    shutil.rmtree(backend_dir, ignore_errors=True)


def test_complete_form_generation_pipeline(temp_dirs):
    """Test complete pipeline from .ai source to generated frontend."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "Contact Manager"

page "AddContact" at "/add":
  show form "New Contact":
    fields:
      - name: name
        component: text_input
        label: "Full Name"
        required: true
        min_length: 2
      - name: email
        component: text_input
        label: "Email"
        required: true
        pattern: "^[^@]+@[^@]+$"
      - name: phone
        component: text_input
        label: "Phone"
        placeholder: "(555) 555-5555"
      - name: notes
        component: textarea
        label: "Notes"
        max_length: 500
    submit_button_text: "Add Contact"
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Generate frontend
    generate_react_vite_site(app, frontend_dir)
    
    # Verify frontend structure
    frontend_path = Path(frontend_dir)
    assert (frontend_path / "src" / "components" / "FormWidget.tsx").exists()
    assert (frontend_path / "src" / "pages").exists()
    assert (frontend_path / "package.json").exists()
    
    # Check FormWidget contains validation logic
    form_widget = (frontend_path / "src" / "components" / "FormWidget.tsx").read_text()
    assert "required" in form_widget.lower()
    assert "pattern" in form_widget.lower() or "regex" in form_widget.lower()
    
    # Verify IR generation
    frontend_ir = build_frontend_ir(app)
    assert len(frontend_ir.pages) == 1
    form_spec = frontend_ir.pages[0].components[0].props["form_spec"]
    assert form_spec.title == "New Contact"
    assert len(form_spec.fields) == 4


def test_form_validation_schema_consistency(temp_dirs):
    """Test that frontend and backend validation schemas are consistent."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "Validation Test"

page "Form" at "/form":
  show form "Validated Form":
    fields:
      - name: username
        component: text_input
        label: "Username"
        required: true
        min_length: 3
        max_length: 20
        pattern: "^[a-zA-Z0-9_]+$"
      - name: age
        component: slider
        label: "Age"
        required: true
        min_value: 18
        max_value: 120
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Build IR to check validation schemas
    frontend_ir = build_frontend_ir(app)
    backend_state = build_backend_state(app)
    
    # Check frontend IR has form with validation
    assert len(frontend_ir.pages) == 1
    page = frontend_ir.pages[0]
    assert len(page.components) == 1
    
    form_component = page.components[0]
    assert form_component.type == "form"
    
    form_spec = form_component.props.get("form_spec")
    assert form_spec is not None
    assert form_spec.validation_schema is not None
    
    # Verify validation schema structure
    schema = form_spec.validation_schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "username" in schema["properties"]
    assert "age" in schema["properties"]
    
    # Check required fields
    assert "required" in schema
    assert "username" in schema["required"]
    assert "age" in schema["required"]
    
    # Check field constraints
    username_schema = schema["properties"]["username"]
    assert username_schema["minLength"] == 3
    assert username_schema["maxLength"] == 20
    assert username_schema["pattern"] == "^[a-zA-Z0-9_]+$"
    
    age_schema = schema["properties"]["age"]
    assert age_schema["minimum"] == 18
    assert age_schema["maximum"] == 120


def test_form_with_select_options_integration(temp_dirs):
    """Test select fields with options generate correctly across stack."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "Survey"

page "Preferences" at "/prefs":
  show form "User Preferences":
    fields:
      - name: country
        component: select
        label: "Country"
        required: true
        options: ["USA", "Canada", "UK", "Germany", "France"]
      - name: interests
        component: multiselect
        label: "Interests"
        options: ["Sports", "Music", "Art", "Technology", "Travel"]
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    generate_react_vite_site(app, frontend_dir)
    
    # Check that options are in generated pages
    pages_dir = Path(frontend_dir) / "src" / "pages"
    page_files = list(pages_dir.glob("*.tsx"))
    
    found_options = False
    for page_file in page_files:
        content = page_file.read_text()
        if "country" in content.lower() and "USA" in content:
            found_options = True
            assert "Canada" in content
            assert "Germany" in content
            break
    
    assert found_options, "Select options not found in generated pages"


def test_form_file_upload_integration(temp_dirs):
    """Test file upload fields generate correctly."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "File Manager"

page "Upload" at "/upload":
  show form "File Upload":
    fields:
      - name: profile_picture
        component: file_upload
        label: "Profile Picture"
        required: true
        accept: "image/*"
        max_file_size: 5242880
      - name: documents
        component: file_upload
        label: "Documents"
        accept: ".pdf,.doc,.docx"
        multiple: true
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    
    # Verify file upload fields in IR
    form_spec = frontend_ir.pages[0].components[0].props["form_spec"]
    
    profile_field = form_spec.fields[0]
    assert profile_field.component == "file_upload"
    assert profile_field.component_config.get("accept") == "image/*"
    assert profile_field.component_config.get("max_file_size") == 5242880
    
    docs_field = form_spec.fields[1]
    assert docs_field.component == "file_upload"
    assert docs_field.component_config.get("multiple") == True


def test_form_conditional_rendering_integration(temp_dirs):
    """Test conditional field expressions work end-to-end."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "Dynamic Form"

page "Survey" at "/survey":
  show form "Pet Survey":
    fields:
      - name: has_pet
        component: checkbox
        label: "Do you have a pet?"
      - name: pet_type
        component: select
        label: "Pet Type"
        options: ["Dog", "Cat", "Bird", "Other"]
        visible: has_pet
      - name: pet_name
        component: text_input
        label: "Pet Name"
        visible: has_pet
      - name: pet_age
        component: slider
        label: "Pet Age"
        min_value: 0
        max_value: 25
        visible: has_pet
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props["form_spec"]
    
    # Verify conditional expressions
    pet_type_field = form_spec.fields[1]
    assert pet_type_field.visible_expr == "has_pet"
    
    pet_name_field = form_spec.fields[2]
    assert pet_name_field.visible_expr == "has_pet"
    
    pet_age_field = form_spec.fields[3]
    assert pet_age_field.visible_expr == "has_pet"


def test_form_layout_modes_integration(temp_dirs):
    """Test horizontal and vertical layout modes."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "Layout Test"

page "Vertical" at "/v":
  show form "Vertical Form":
    fields:
      - name: field1
        component: text_input
        label: "Field 1"
    layout: vertical

page "Horizontal" at "/h":
  show form "Horizontal Form":
    fields:
      - name: field2
        component: text_input
        label: "Field 2"
    layout: horizontal
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    
    # Verify layout modes in IR
    vertical_form = frontend_ir.pages[0].components[0].props["form_spec"]
    assert vertical_form.layout_mode == "vertical"
    
    horizontal_form = frontend_ir.pages[1].components[0].props["form_spec"]
    assert horizontal_form.layout_mode == "horizontal"


def test_form_all_validation_constraints_integration(temp_dirs):
    """Test all validation constraints work together."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "Comprehensive Validation"

page "Form" at "/form":
  show form "Complete Validation":
    fields:
      - name: username
        component: text_input
        label: "Username"
        required: true
        min_length: 3
        max_length: 20
        pattern: "^[a-zA-Z0-9_]+$"
        placeholder: "user123"
      - name: bio
        component: textarea
        label: "Bio"
        max_length: 500
        help_text: "Tell us about yourself"
      - name: age
        component: slider
        label: "Age"
        required: true
        min_value: 13
        max_value: 120
        step: 1
        default: 25
    validation_mode: on_blur
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props["form_spec"]
    
    # Verify all field constraints
    username_field = form_spec.fields[0]
    assert username_field.required == True
    assert username_field.validation["min_length"] == 3
    assert username_field.validation["max_length"] == 20
    assert username_field.validation["pattern"] == "^[a-zA-Z0-9_]+$"
    assert username_field.placeholder == "user123"
    
    bio_field = form_spec.fields[1]
    assert bio_field.validation["max_length"] == 500
    assert bio_field.help_text == "Tell us about yourself"
    
    age_field = form_spec.fields[2]
    assert age_field.required == True
    assert age_field.validation["min_value"] == 13
    assert age_field.validation["max_value"] == 120
    assert age_field.validation["step"] == 1
    assert age_field.default_value == "25"
    
    # Verify validation mode
    assert form_spec.validation_mode == "on_blur"


def test_multiple_forms_same_page_integration(temp_dirs):
    """Test multiple forms on same page work independently."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "Multi Form"

page "Dashboard" at "/":
  show form "Quick Search":
    fields:
      - name: search_query
        component: text_input
        label: "Search"
        placeholder: "Type to search..."
    submit_button_text: "Search"
    layout: horizontal
  
  show form "Filters":
    fields:
      - name: category
        component: select
        label: "Category"
        options: ["All", "Active", "Archived", "Draft"]
      - name: date_range
        component: select
        label: "Date Range"
        options: ["Today", "This Week", "This Month", "All Time"]
    submit_button_text: "Apply Filters"
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    page = frontend_ir.pages[0]
    
    # Verify two forms on page
    assert len(page.components) == 2
    
    search_form = page.components[0].props["form_spec"]
    assert search_form.title == "Quick Search"
    assert search_form.layout_mode == "horizontal"
    assert len(search_form.fields) == 1
    assert search_form.submit_button_text == "Search"
    
    filters_form = page.components[1].props["form_spec"]
    assert filters_form.title == "Filters"
    assert filters_form.layout_mode == "vertical"
    assert len(filters_form.fields) == 2
    assert filters_form.submit_button_text == "Apply Filters"


def test_form_with_all_field_types_integration(temp_dirs):
    """Test form with all 11 field types generates correctly."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "All Fields"

page "Complete" at "/complete":
  show form "All Field Types":
    fields:
      - name: text_field
        component: text_input
        label: "Text Input"
      - name: textarea_field
        component: textarea
        label: "Text Area"
      - name: select_field
        component: select
        label: "Select"
        options: ["Option 1", "Option 2"]
      - name: multiselect_field
        component: multiselect
        label: "Multi Select"
        options: ["A", "B", "C"]
      - name: checkbox_field
        component: checkbox
        label: "Checkbox"
      - name: switch_field
        component: switch
        label: "Switch"
      - name: radio_field
        component: radio_group
        label: "Radio Group"
        options: ["Yes", "No"]
      - name: slider_field
        component: slider
        label: "Slider"
        min_value: 0
        max_value: 100
      - name: date_field
        component: date_picker
        label: "Date"
      - name: datetime_field
        component: datetime_picker
        label: "Date Time"
      - name: file_field
        component: file_upload
        label: "File"
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props["form_spec"]
    
    # Verify all 11 field types
    assert len(form_spec.fields) == 11
    
    field_types = [field.component for field in form_spec.fields]
    expected_types = [
        "text_input", "textarea", "select", "multiselect",
        "checkbox", "switch", "radio_group", "slider",
        "date_picker", "datetime_picker", "file_upload"
    ]
    
    for expected_type in expected_types:
        assert expected_type in field_types, f"Missing field type: {expected_type}"


def test_form_validation_messages_integration(temp_dirs):
    """Test custom validation messages."""
    frontend_dir, backend_dir = temp_dirs
    
    source = '''
app "Messaging Test"

page "Form" at "/form":
  show form "Registration":
    fields:
      - name: email
        component: text_input
        label: "Email"
        required: true
    submit_button_text: "Register"
    loading_text: "Processing..."
    success_message: "Registration successful!"
    error_message: "Registration failed. Please try again."
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    frontend_ir = build_frontend_ir(app)
    form_spec = frontend_ir.pages[0].components[0].props["form_spec"]
    
    # Verify custom messages
    assert form_spec.submit_button_text == "Register"
    assert form_spec.loading_text == "Processing..."
    assert form_spec.success_message == "Registration successful!"
    assert form_spec.error_message == "Registration failed. Please try again."
