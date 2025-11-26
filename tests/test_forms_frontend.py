"""Tests for form frontend code generation."""

import pytest
import tempfile
import shutil
from pathlib import Path
from namel3ss.parser import Parser
from namel3ss.codegen.frontend.react.main import generate_react_vite_site


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test output."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_form_widget_component_generation(temp_output_dir):
    """Test that FormWidget component is generated correctly."""
    source = '''
app "Test"

page "Form" at "/form":
  show form "Test":
    fields:
      - name: test_field
        component: text_input
        label: "Test"
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    generate_react_vite_site(app, temp_output_dir)
    
    # Check FormWidget was generated
    form_widget = Path(temp_output_dir) / "src" / "components" / "FormWidget.tsx"
    assert form_widget.exists(), "FormWidget.tsx not generated"
    
    components_tsx = form_widget.read_text()
    
    # Verify FormWidget is present
    assert "FormWidget" in components_tsx
    
    # Verify field type renderers are present
    field_types = [
        "text_input", "textarea", "select", "multiselect",
        "checkbox", "switch", "radio", "slider",
        "date", "file"
    ]
    
    found_count = sum(1 for ft in field_types if ft in components_tsx)
    assert found_count >= 8, f"Only found {found_count} field types in FormWidget"
    
    # Verify validation logic
    assert "validate" in components_tsx.lower()
    assert "onChange" in components_tsx or "handleChange" in components_tsx
    
    # Verify form submission
    assert "onSubmit" in components_tsx or "handleSubmit" in components_tsx
    
    # Verify state management
    assert "useState" in components_tsx
    assert "formData" in components_tsx or "data" in components_tsx.lower()
    assert "errors" in components_tsx.lower()


def test_form_page_generation(temp_output_dir):
    """Test that form pages generate correctly to files."""
    source = '''
app "Codegen Test"

page "ContactForm" at "/contact":
  show form "Contact Us":
    fields:
      - name: name
        component: text_input
        label: "Name"
        required: true
      - name: email
        component: text_input
        label: "Email"
        required: true
        pattern: "^[^@]+@[^@]+$"
      - name: message
        component: textarea
        label: "Message"
        required: true
        min_length: 10
    submit_button_text: "Send Message"
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Generate frontend
    generate_react_vite_site(app, temp_output_dir)
    
    # Check generated structure
    src_dir = Path(temp_output_dir) / "src"
    assert src_dir.exists()
    
    # Check components were generated
    components_dir = src_dir / "components"
    assert components_dir.exists()
    form_widget = components_dir / "FormWidget.tsx"
    assert form_widget.exists()
    
    # Verify FormWidget content
    form_content = form_widget.read_text()
    assert "FormWidget" in form_content
    assert "text_input" in form_content or "textarea" in form_content
    
    # Check pages were generated
    pages_dir = src_dir / "pages"
    assert pages_dir.exists()
    
    # Check page file exists (lowercase slug)
    page_files = list(pages_dir.glob("*.tsx"))
    assert len(page_files) > 0
    
    # Find the contact form page
    contact_page = None
    for page_file in page_files:
        content = page_file.read_text()
        if "Contact Us" in content or "ContactForm" in content:
            contact_page = content
            break
    
    assert contact_page is not None, "ContactForm page not found"
    assert "FormWidget" in contact_page
    assert "name" in contact_page
    assert "email" in contact_page


def test_form_validation_serialization(temp_output_dir):
    """Test different validation modes serialize correctly."""
    source = '''
app "Validation Test"

page "Form" at "/form":
  show form "Test":
    fields:
      - name: email
        component: text_input
        label: "Email"
        required: true
        pattern: "^[^@]+@[^@]+$"
    validation_mode: on_blur
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    generate_react_vite_site(app, temp_output_dir)
    
    # Find generated page
    pages_dir = Path(temp_output_dir) / "src" / "pages"
    page_files = list(pages_dir.glob("*.tsx"))
    
    found_validation = False
    for page_file in page_files:
        content = page_file.read_text()
        if "email" in content.lower():
            # Check for validation config
            assert "required" in content.lower() or "validation" in content.lower()
            found_validation = True
            break
    
    assert found_validation, "Validation configuration not found in generated pages"


def test_form_select_options_serialization(temp_output_dir):
    """Test select field options are properly serialized."""
    source = '''
app "Select Test"

page "Form" at "/form":
  show form "Preferences":
    fields:
      - name: country
        component: select
        label: "Country"
        options: ["USA", "Canada", "UK"]
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    generate_react_vite_site(app, temp_output_dir)
    
    # Find generated page
    pages_dir = Path(temp_output_dir) / "src" / "pages"
    page_files = list(pages_dir.glob("*.tsx"))
    
    found_options = False
    for page_file in page_files:
        content = page_file.read_text()
        if "country" in content.lower():
            # Check that options are serialized
            assert "USA" in content or "Canada" in content
            found_options = True
            break
    
    assert found_options, "Select options not found in generated pages"


def test_form_file_upload_serialization(temp_output_dir):
    """Test file upload field generates correct configuration."""
    source = '''
app "Upload Test"

page "Form" at "/form":
  show form "File Upload":
    fields:
      - name: avatar
        component: file_upload
        label: "Avatar"
        accept: "image/*"
        max_file_size: 5242880
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    generate_react_vite_site(app, temp_output_dir)
    
    # Find generated page
    pages_dir = Path(temp_output_dir) / "src" / "pages"
    page_files = list(pages_dir.glob("*.tsx"))
    
    found_upload = False
    for page_file in page_files:
        content = page_file.read_text()
        if "avatar" in content.lower() or "file_upload" in content.lower():
            found_upload = True
            break
    
    assert found_upload, "File upload field not found in generated pages"


def test_form_conditional_fields_serialization(temp_output_dir):
    """Test conditional field expressions are serialized."""
    source = '''
app "Conditional Test"

page "Form" at "/form":
  show form "Survey":
    fields:
      - name: has_pet
        component: checkbox
        label: "Do you have a pet?"
      - name: pet_name
        component: text_input
        label: "Pet Name"
        visible: has_pet
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    generate_react_vite_site(app, temp_output_dir)
    
    # Find generated page
    pages_dir = Path(temp_output_dir) / "src" / "pages"
    page_files = list(pages_dir.glob("*.tsx"))
    
    found_conditional = False
    for page_file in page_files:
        content = page_file.read_text()
        if "has_pet" in content and "pet_name" in content:
            found_conditional = True
            break
    
    assert found_conditional, "Conditional fields not found in generated pages"


def test_multiple_forms_on_page(temp_output_dir):
    """Test multiple forms on same page generate correctly."""
    source = '''
app "Multi Form Test"

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
        options: ["All", "Active"]
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    generate_react_vite_site(app, temp_output_dir)
    
    # Find generated page
    pages_dir = Path(temp_output_dir) / "src" / "pages"
    page_files = list(pages_dir.glob("*.tsx"))
    
    found_both_forms = False
    for page_file in page_files:
        content = page_file.read_text()
        if "query" in content and "category" in content:
            # Both forms should be present
            found_both_forms = True
            break
    
    assert found_both_forms, "Multiple forms not found in generated page"


def test_form_all_field_types(temp_output_dir):
    """Test all field types generate correctly."""
    source = '''
app "All Fields Test"

page "Form" at "/form":
  show form "Complete":
    fields:
      - name: text_field
        component: text_input
        label: "Text"
      - name: textarea_field
        component: textarea
        label: "Textarea"
      - name: select_field
        component: select
        label: "Select"
        options: ["A", "B"]
      - name: multiselect_field
        component: multiselect
        label: "Multiselect"
        options: ["X", "Y"]
      - name: checkbox_field
        component: checkbox
        label: "Checkbox"
      - name: switch_field
        component: switch
        label: "Switch"
      - name: radio_field
        component: radio_group
        label: "Radio"
        options: ["1", "2"]
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
        label: "DateTime"
      - name: file_field
        component: file_upload
        label: "File"
    layout: vertical
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    generate_react_vite_site(app, temp_output_dir)
    
    # Verify FormWidget has all field types
    form_widget = Path(temp_output_dir) / "src" / "components" / "FormWidget.tsx"
    assert form_widget.exists()
    
    content = form_widget.read_text()
    
    # Check for field type handling
    field_types = [
        "text_input", "textarea", "select", "multiselect",
        "checkbox", "switch", "radio", "slider",
        "date", "file"
    ]
    
    found_types = []
    for field_type in field_types:
        if field_type in content:
            found_types.append(field_type)
    
    # Should have most field types mentioned
    assert len(found_types) >= 8, f"Only found {len(found_types)} field types in FormWidget: {found_types}"
