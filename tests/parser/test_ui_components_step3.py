"""
Test UI components, forms, and navigation for Step 3 of Processing Tools Upgrade.

Tests:
1. Unsupported components show clear error messages with alternatives
2. File upload forms work correctly
3. Navigation patterns work without route parameters
"""

import pytest
from namel3ss.parser import Parser


# =============================================================================
# Unsupported Component Tests
# =============================================================================


def test_progress_bar_not_supported():
    """Test that progress_bar shows clear error with alternative."""
    source = '''
app "Test"

page "Home" at "/":
    show progress_bar value=75
'''
    
    with pytest.raises(Exception) as exc_info:
        Parser(source).parse()
    
    error_msg = str(exc_info.value).lower()
    assert 'progress' in error_msg or 'not supported' in error_msg
    assert 'stat_summary' in error_msg or 'alternative' in error_msg


def test_code_block_not_supported():
    """Test that code_block shows clear error with alternative."""
    source = '''
app "Test"

page "Home" at "/":
    show code_block language="python"
'''
    
    with pytest.raises(Exception) as exc_info:
        Parser(source).parse()
    
    error_msg = str(exc_info.value).lower()
    assert 'code' in error_msg or 'not supported' in error_msg
    assert 'text' in error_msg or 'alternative' in error_msg


def test_json_view_not_supported():
    """Test that json_view shows clear error with alternative."""
    source = '''
app "Test"

page "Home" at "/":
    show json_view data=response
'''
    
    with pytest.raises(Exception) as exc_info:
        Parser(source).parse()
    
    error_msg = str(exc_info.value).lower()
    assert 'json' in error_msg or 'not supported' in error_msg
    assert 'text' in error_msg or 'data_table' in error_msg or 'alternative' in error_msg


def test_tree_view_not_supported():
    """Test that tree_view shows clear error with alternative."""
    source = '''
app "Test"

page "Home" at "/":
    show tree_view data=tree
'''
    
    with pytest.raises(Exception) as exc_info:
        Parser(source).parse()
    
    error_msg = str(exc_info.value).lower()
    assert 'tree' in error_msg or 'not supported' in error_msg
    assert 'accordion' in error_msg or 'data_list' in error_msg or 'alternative' in error_msg


# =============================================================================
# Supported Component Tests
# =============================================================================


def test_supported_components_parse():
    """Test that all documented supported components parse correctly."""
    source = '''
app "Test"

dataset "users" from inline:
    fields:
        - id: int
        - name: text

page "Dashboard" at "/":
    # Basic components
    show text "Welcome"
    
    show table "Users" from dataset users:
        columns: ["id", "name"]
    
    show chart "Stats" from dataset users:
        type: "bar"
    
    # Data display components
    show card "Items" from dataset users:
        header:
            title: "{{name}}"
    
    show list "List" from dataset users
    
    show data_table "Advanced Table" from dataset users:
        columns:
            - field: "name"
              header: "Name"
    
    show data_list "Activity" from dataset users:
        item:
            title: "{{name}}"
    
    show stat_summary "Metrics":
        stats:
            - label: "Total"
              value_binding: "count"
    
    # Forms
    show form "Create User":
        fields:
            - name: "username"
              component: "text_input"
        on submit:
            show toast "Created" type="success"
    
    # Layout
    stack direction="column":
        show text "First"
        show text "Second"
    
    grid columns=2:
        show text "A"
        show text "B"
    
    tabs:
        tab "Tab 1":
            show text "Content 1"
        tab "Tab 2":
            show text "Content 2"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert app is not None
    assert len(app.pages) == 1
    page = app.pages[0]
    # Should have multiple statements without errors
    assert len(page.body) > 5


# =============================================================================
# File Upload Form Tests
# =============================================================================


def test_file_upload_form_parses():
    """Test that file upload forms parse correctly."""
    source = '''
app "Test"

page "Upload" at "/upload":
    show form "Upload File":
        fields:
            - name: "file"
              component: "file_input"
              label: "Choose File"
              accept: "image/*"
              max_file_size: 5242880
              required: true
            - name: "description"
              component: "textarea"
              label: "Description"
        on submit:
            show toast "Uploaded" type="success"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert app is not None
    page = app.pages[0]
    form = page.body[0]
    
    # Check file field properties
    file_field = form.fields[0]
    assert file_field.name == "file"
    assert file_field.component == "file_input"
    assert file_field.accept == "image/*"
    assert file_field.max_file_size == 5242880
    assert file_field.required is True


def test_multiple_file_upload():
    """Test multiple file upload configuration."""
    source = '''
app "Test"

page "Upload" at "/upload":
    show form "Upload Files":
        fields:
            - name: "files"
              component: "file_input"
              label: "Choose Files"
              multiple: true
              accept: ".pdf,.doc,.docx"
        on submit:
            show toast "Uploaded" type="success"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    form = page.body[0]
    file_field = form.fields[0]
    
    assert file_field.multiple is True
    assert file_field.accept == ".pdf,.doc,.docx"


# =============================================================================
# Navigation Pattern Tests
# =============================================================================


def test_static_routes_parse():
    """Test that static routes parse correctly."""
    source = '''
app "Test"

page "Home" at "/"
page "Users" at "/users"
page "Settings" at "/settings"
page "User Detail" at "/user-detail"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.pages) == 4
    assert app.pages[0].route == "/"
    assert app.pages[1].route == "/users"
    assert app.pages[2].route == "/settings"
    assert app.pages[3].route == "/user-detail"


def test_navigation_with_session_state():
    """Test navigation pattern using session state."""
    source = '''
app "Test"

dataset "users" from inline:
    fields:
        - id: int
        - name: text

memory:
    scope: session
    storage:
        selected_user_id: text

page "Users" at "/users":
    show data_table "All Users" from dataset users:
        columns:
            - field: "name"
              header: "Name"
        row_actions:
            - label: "View Details"
              action: "view_user"
    
    action "view_user":
        set session.selected_user_id to row.id
        go to page "/user-detail"

page "User Detail" at "/user-detail":
    show text "User: {{session.selected_user_id}}"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.pages) == 2
    users_page = app.pages[0]
    detail_page = app.pages[1]
    
    # Verify pages exist and have correct routes
    assert users_page.route == "/users"
    assert detail_page.route == "/user-detail"
    
    # Verify memory exists
    assert app.memory is not None
    assert app.memory.scope == "session"


def test_modal_navigation_pattern():
    """Test modal-based navigation pattern."""
    source = '''
app "Test"

dataset "jobs" from inline:
    fields:
        - id: int
        - name: text

memory:
    scope: session
    storage:
        selected_job: object

page "Jobs" at "/jobs":
    show data_table "All Jobs" from dataset jobs:
        row_actions:
            - label: "View Details"
              action: "show_details"
    
    modal id="job_detail":
        title: "Job Details"
        content:
            show text "Job: {{session.selected_job.name}}"
        actions:
            - label: "Close"
              action: "close_modal"
    
    action "show_details":
        set session.selected_job to row
        show modal "job_detail"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.pages) == 1
    page = app.pages[0]
    
    # Verify modal exists (would need to check page.body for modal)
    # This tests that the pattern parses without errors
    assert len(page.body) > 0


def test_breadcrumb_navigation():
    """Test breadcrumb navigation."""
    source = '''
app "Test"

page "Detail" at "/detail":
    breadcrumbs:
        - label: "Home"
          link: "/"
        - label: "Items"
          link: "/items"
        - label: "Detail"
    
    show text "Detail page"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    # Verify page parses (breadcrumbs structure would need deeper inspection)
    assert page.route == "/detail"


def test_sidebar_navigation():
    """Test sidebar navigation component."""
    source = '''
app "Test"

page "Home" at "/":
    sidebar:
        title: "My App"
        nav:
            - label: "Dashboard"
              link: "/"
            - label: "Users"
              link: "/users"
            - label: "Settings"
              link: "/settings"
    
    show text "Welcome"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    page = app.pages[0]
    assert page.route == "/"


# =============================================================================
# Integration Tests
# =============================================================================


def test_complete_navigation_app():
    """Test complete app with navigation patterns."""
    source = '''
app "Job Management"

dataset "FileProcessingJob":
    schema:
        id: uuid
        filename: text
        status: text

memory:
    scope: session
    storage:
        selected_job: object

page "Dashboard" at "/":
    sidebar:
        title: "Job Manager"
        nav:
            - label: "Dashboard"
              link: "/"
            - label: "Jobs"
              link: "/jobs"
    
    show stat_summary "Overview":
        stats:
            - label: "Total Jobs"
              value_binding: "count"

page "Jobs" at "/jobs":
    sidebar:
        title: "Job Manager"
        nav:
            - label: "Dashboard"
              link: "/"
            - label: "Jobs"
              link: "/jobs"
    
    show data_table "All Jobs" from dataset FileProcessingJob:
        columns:
            - field: "filename"
              header: "File"
            - field: "status"
              header: "Status"
        row_actions:
            - label: "View"
              action: "view_job"
    
    action "view_job":
        set session.selected_job to row
        go to page "/job-detail"

page "Job Detail" at "/job-detail":
    breadcrumbs:
        - label: "Home"
          link: "/"
        - label: "Jobs"
          link: "/jobs"
        - label: "Detail"
    
    show card "Job Info":
        header:
            title: "{{session.selected_job.filename}}"
        sections:
            - type: "info_grid"
              items:
                  - label: "Status"
                    value: "{{session.selected_job.status}}"
        actions:
            - label: "Back"
              link: "/jobs"

page "New Job" at "/jobs/new":
    show form "Create Job":
        fields:
            - name: "file"
              component: "file_input"
              accept: ".pdf,.csv"
              required: true
        on submit:
            show toast "Job created" type="success"
            go to page "/jobs"
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    assert len(app.pages) == 4
    assert app.memory is not None
    
    
    # Verify all pages parse correctly
    routes = [p.route for p in app.pages]
    assert "/" in routes
    assert "/jobs" in routes
    assert "/job-detail" in routes
    assert "/jobs/new" in routes
