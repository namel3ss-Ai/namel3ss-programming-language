"""Tests for data display components and layout primitives."""

import pytest
from namel3ss.parser import Parser
from namel3ss.ir.builder import build_frontend_ir


def test_show_data_table_basic():
    """Test basic ShowDataTable parsing and IR generation."""
    source = '''
app "Dashboard"

page "Orders" at "/orders":
  show data_table "Order List" from dataset orders:
    columns:
      - id: order_id
        label: "Order #"
        sortable: true
      - id: customer
        label: "Customer"
        sortable: true
      - id: total
        label: "Total"
        format: currency
    page_size: 25
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Verify parsing
    assert len(app.pages) == 1
    page = app.pages[0]
    assert page.name == "Orders"
    assert len(page.body) == 1
    
    statement = page.body[0]
    assert statement.__class__.__name__ == "ShowDataTable"
    assert statement.title == "Order List"
    assert statement.source == "orders"
    assert statement.page_size == 25
    assert len(statement.columns) == 3


def test_show_data_list_with_actions():
    """Test ShowDataList with row actions."""
    source = '''
app "CRM"

page "Contacts" at "/contacts":
  show data_list "Contact List" from dataset contacts:
    item_config:
      title: name
      subtitle: email
      metadata:
        phone: phone_number
        company: company_name
    row_actions:
      - label: "Edit"
        icon: edit
        action: edit_contact
      - label: "Delete"
        icon: trash
        style: danger
        action: delete_contact
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "ShowDataList"
    assert statement.title == "Contact List"
    assert statement.source == "contacts"


def test_show_stat_summary():
    """Test ShowStatSummary for metrics display."""
    source = '''
app "Analytics"

page "Dashboard" at "/":
  show stat_summary "Key Metrics" from dataset metrics:
    stats:
      - label: "Total Sales"
        value: total_sales
        format: currency
        trend: +12.5%
        trend_direction: up
      - label: "Active Users"
        value: user_count
        format: number
        icon: users
      - label: "Conversion Rate"
        value: conversion_rate
        format: percentage
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "ShowStatSummary"
    assert statement.title == "Key Metrics"
    assert statement.source == "metrics"


def test_show_timeline():
    """Test ShowTimeline for chronological data."""
    source = '''
app "Project Tracker"

page "Activity" at "/activity":
  show timeline "Project Timeline" from dataset activities:
    item_config:
      title: activity_name
      timestamp: created_at
      description: description
      icon: activity_icon
      user: user_name
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "ShowTimeline"
    assert statement.title == "Project Timeline"
    assert statement.source == "activities"


def test_show_avatar_group():
    """Test ShowAvatarGroup for user display."""
    source = '''
app "Team"

page "Team" at "/team":
  show avatar_group "Team Members" from dataset team_members:
    avatar_field: profile_picture
    name_field: full_name
    max_display: 5
    size: medium
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "ShowAvatarGroup"
    assert statement.title == "Team Members"
    assert statement.source == "team_members"


def test_show_data_chart():
    """Test ShowDataChart for data visualization."""
    source = '''
app "Analytics"

page "Charts" at "/charts":
  show data_chart "Sales Trend" from dataset sales:
    chart_type: line
    x_axis: date
    y_axis: revenue
    group_by: region
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "ShowDataChart"
    assert statement.title == "Sales Trend"
    assert statement.source == "sales"


def test_stack_layout_vertical():
    """Test vertical StackLayout."""
    source = '''
app "Layout Test"

page "Stack" at "/stack":
  layout stack:
    direction: vertical
    gap: medium
    align: center
    children:
      - show text "Header" style: heading
      - show text "Content" style: body
      - show text "Footer" style: caption
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "StackLayout"
    assert statement.direction == "vertical"
    assert statement.gap == "medium"
    assert statement.align == "center"
    assert len(statement.children) == 3


def test_stack_layout_horizontal():
    """Test horizontal StackLayout."""
    source = '''
app "Layout Test"

page "HStack" at "/hstack":
  layout stack:
    direction: horizontal
    gap: large
    justify: space_between
    children:
      - show text "Left" style: body
      - show text "Center" style: body
      - show text "Right" style: body
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "StackLayout"
    assert statement.direction == "horizontal"
    assert statement.justify == "space_between"


def test_grid_layout():
    """Test GridLayout with multiple children."""
    source = '''
app "Dashboard"

page "Grid" at "/grid":
  layout grid:
    columns: 3
    gap: large
    responsive: true
    min_column_width: 300px
    children:
      - show card "Card 1" from dataset data1
      - show card "Card 2" from dataset data2
      - show card "Card 3" from dataset data3
      - show card "Card 4" from dataset data4
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "GridLayout"
    assert statement.columns == 3
    assert statement.gap == "large"
    assert statement.responsive == True
    assert statement.min_column_width == "300px"
    assert len(statement.children) == 4


def test_split_layout():
    """Test SplitLayout with two panes."""
    source = '''
app "Editor"

page "IDE" at "/ide":
  layout split:
    orientation: horizontal
    ratio: 0.3
    resizable: true
    left:
      - show text "Sidebar" style: body
    right:
      - show text "Main Content" style: body
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "SplitLayout"
    assert statement.orientation == "horizontal"
    assert statement.ratio == 0.3
    assert statement.resizable == True


def test_tabs_layout():
    """Test TabsLayout with multiple tabs."""
    source = '''
app "Dashboard"

page "Tabs" at "/tabs":
  layout tabs:
    tabs:
      - id: overview
        label: "Overview"
        icon: home
        children:
          - show text "Overview Content" style: body
      - id: details
        label: "Details"
        icon: info
        children:
          - show text "Details Content" style: body
      - id: settings
        label: "Settings"
        icon: settings
        children:
          - show text "Settings Content" style: body
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "TabsLayout"
    assert len(statement.tabs) == 3


def test_accordion_layout():
    """Test AccordionLayout with collapsible sections."""
    source = '''
app "Help"

page "FAQ" at "/faq":
  layout accordion:
    allow_multiple: false
    sections:
      - id: getting_started
        title: "Getting Started"
        children:
          - show text "Getting started content" style: body
      - id: advanced
        title: "Advanced Topics"
        children:
          - show text "Advanced content" style: body
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "AccordionLayout"
    assert statement.allow_multiple == False
    assert len(statement.sections) == 2


def test_nested_layouts():
    """Test nested layout compositions."""
    source = '''
app "Complex Dashboard"

page "Dashboard" at "/":
  layout grid:
    columns: 2
    gap: medium
    children:
      - layout stack:
          direction: vertical
          gap: small
          children:
            - show text "Section 1" style: heading
            - show data_table "Data" from dataset items:
                columns:
                  - id: name
                    label: "Name"
      - layout stack:
          direction: vertical
          gap: small
          children:
            - show text "Section 2" style: heading
            - show data_chart "Chart" from dataset metrics:
                chart_type: bar
                x_axis: month
                y_axis: value
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    grid = app.pages[0].body[0]
    assert grid.__class__.__name__ == "GridLayout"
    assert len(grid.children) == 2
    
    # Check nested stacks
    first_stack = grid.children[0]
    assert first_stack.__class__.__name__ == "StackLayout"
    assert len(first_stack.children) == 2


def test_data_table_with_toolbar():
    """Test ShowDataTable with search and filters."""
    source = '''
app "Orders"

page "Orders" at "/orders":
  show data_table "All Orders" from dataset orders:
    columns:
      - id: order_id
        label: "Order #"
      - id: status
        label: "Status"
    toolbar:
      search:
        field: customer
        placeholder: "Search customers..."
      filters:
        - field: status
          label: "Status"
          options: ["pending", "completed", "cancelled"]
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    statement = app.pages[0].body[0]
    assert statement.__class__.__name__ == "ShowDataTable"
    assert statement.toolbar is not None


def test_complex_dashboard_integration():
    """Test complex dashboard with multiple components and layouts."""
    source = '''
app "Business Dashboard"

page "Dashboard" at "/":
  layout stack:
    direction: vertical
    gap: large
    children:
      - show text "Dashboard" style: heading
      
      - show stat_summary "KPIs" from dataset kpis:
          stats:
            - label: "Revenue"
              value: total_revenue
              format: currency
            - label: "Users"
              value: user_count
              format: number
      
      - layout grid:
          columns: 2
          gap: medium
          children:
            - show data_chart "Sales Trend" from dataset sales:
                chart_type: line
                x_axis: date
                y_axis: amount
            
            - show data_chart "Category Breakdown" from dataset categories:
                chart_type: pie
                x_axis: category
                y_axis: count
      
      - show data_table "Recent Orders" from dataset orders:
          columns:
            - id: order_id
              label: "Order #"
            - id: customer
              label: "Customer"
            - id: total
              label: "Total"
              format: currency
          page_size: 10
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Verify structure
    assert len(app.pages) == 1
    page = app.pages[0]
    assert len(page.body) == 1
    
    main_stack = page.body[0]
    assert main_stack.__class__.__name__ == "StackLayout"
    assert len(main_stack.children) == 5
    
    # Verify component types
    assert main_stack.children[0].__class__.__name__ == "ShowText"
    assert main_stack.children[1].__class__.__name__ == "ShowStatSummary"
    assert main_stack.children[2].__class__.__name__ == "GridLayout"
    assert main_stack.children[3].__class__.__name__ == "ShowDataTable"


def test_data_display_ir_generation():
    """Test IR generation for data display components."""
    source = '''
app "Analytics"

page "Reports" at "/reports":
  show data_table "Sales Report" from dataset sales:
    columns:
      - id: date
        label: "Date"
      - id: amount
        label: "Amount"
        format: currency
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Build IR
    frontend_ir = build_frontend_ir(app)
    
    # Verify IR structure
    assert len(frontend_ir.pages) == 1
    page = frontend_ir.pages[0]
    assert page.name == "Reports"
    assert len(page.components) == 1
    
    component = page.components[0]
    assert component.type == "table"  # data_table maps to table component
    assert component.data_source == "sales"


def test_layout_ir_generation():
    """Test IR generation for layout primitives."""
    source = '''
app "Layout Test"

page "Test" at "/test":
  layout grid:
    columns: 2
    gap: medium
    children:
      - show text "Column 1" style: body
      - show text "Column 2" style: body
'''
    
    parser = Parser(source)
    module = parser.parse()
    app = module.body[0]
    
    # Build IR
    frontend_ir = build_frontend_ir(app)
    
    # Verify IR structure
    assert len(frontend_ir.pages) == 1
    page = frontend_ir.pages[0]
    # Layout components should be converted to appropriate IR representation
    assert len(page.components) >= 1
