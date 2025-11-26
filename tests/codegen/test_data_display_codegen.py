"""
Test frontend codegen for data display components.
"""

import pytest
from namel3ss.parser import Parser
from namel3ss.ir.builder import IRBuilder
from namel3ss.codegen.frontend.react.pages import generate_page_component


def parse_build_and_codegen(source: str) -> str:
    """Helper to parse, build IR, and generate React code."""
    parser = Parser()
    app = parser.parse(source)
    builder = IRBuilder()
    ir = builder.build(app)
    
    page = ir["pages"][0]
    return generate_page_component(page, ir)


# =============================================================================
# DATA TABLE CODEGEN TESTS
# =============================================================================


def test_data_table_codegen_basic():
    """Test basic data table codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int
    - name: text

page test:
  path: "/test"
  title: "Test"
  
  show data_table "Users" from dataset test_data:
    columns:
      - field: id
        header: "ID"
      - field: name
        header: "Name"
'''
    code = parse_build_and_codegen(source)
    
    assert "DataTableWidget" in code
    assert '"type": "data_table"' in code
    assert '"title": "Users"' in code
    assert '"field": "id"' in code
    assert '"header": "ID"' in code


def test_data_table_codegen_with_toolbar():
    """Test data table codegen with toolbar."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_table "Items" from dataset test_data:
    columns:
      - field: id
        header: "ID"
    
    toolbar:
      searchable: true
      search_fields: ["name"]
      filters:
        - field: status
          label: "Status"
          type: select
'''
    code = parse_build_and_codegen(source)
    
    assert "DataTableWidget" in code
    assert '"searchable": true' in code
    assert '"filters"' in code


def test_data_table_codegen_with_actions():
    """Test data table codegen with row actions."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_table "Items" from dataset test_data:
    columns:
      - field: id
        header: "ID"
    
    row_actions:
      - label: "Edit"
        action: edit_item
        icon: edit
'''
    code = parse_build_and_codegen(source)
    
    assert "DataTableWidget" in code
    assert '"row_actions"' in code
    assert '"label": "Edit"' in code


# =============================================================================
# DATA LIST CODEGEN TESTS
# =============================================================================


def test_data_list_codegen_basic():
    """Test basic data list codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_list "Activities" from dataset test_data:
    item:
      title:
        field: event_name
'''
    code = parse_build_and_codegen(source)
    
    assert "DataListWidget" in code
    assert '"type": "data_list"' in code
    assert '"title": "Activities"' in code
    assert '"item_config"' in code


def test_data_list_codegen_with_avatar():
    """Test data list codegen with avatar."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_list "Users" from dataset test_data:
    item:
      avatar:
        field: profile_image
        size: lg
      
      title:
        field: username
'''
    code = parse_build_and_codegen(source)
    
    assert "DataListWidget" in code
    assert '"avatar"' in code
    assert '"size": "lg"' in code


def test_data_list_codegen_with_metadata():
    """Test data list codegen with metadata."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_list "Events" from dataset test_data:
    item:
      title:
        field: event_name
      
      metadata:
        - field: timestamp
          icon: clock
          format: relative
'''
    code = parse_build_and_codegen(source)
    
    assert "DataListWidget" in code
    assert '"metadata"' in code
    assert '"icon": "clock"' in code


# =============================================================================
# STAT SUMMARY CODEGEN TESTS
# =============================================================================


def test_stat_summary_codegen_basic():
    """Test basic stat summary codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show stat_summary from dataset test_data:
    label: "Total Revenue"
    value:
      field: revenue
      format: currency
'''
    code = parse_build_and_codegen(source)
    
    assert "StatSummaryWidget" in code
    assert '"type": "stat_summary"' in code
    assert '"label": "Total Revenue"' in code
    assert '"format": "currency"' in code


def test_stat_summary_codegen_with_delta():
    """Test stat summary codegen with delta."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show stat_summary from dataset test_data:
    label: "Sales"
    value:
      field: sales
      format: number
    
    delta:
      field: change
      format: percentage
'''
    code = parse_build_and_codegen(source)
    
    assert "StatSummaryWidget" in code
    assert '"delta"' in code
    assert '"format": "percentage"' in code


def test_stat_summary_codegen_with_sparkline():
    """Test stat summary codegen with sparkline."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show stat_summary from dataset test_data:
    label: "Views"
    value:
      field: views
      format: compact
    
    sparkline:
      data: history
      color: "#3b82f6"
'''
    code = parse_build_and_codegen(source)
    
    assert "StatSummaryWidget" in code
    assert '"sparkline"' in code
    assert '"color": "#3b82f6"' in code


# =============================================================================
# TIMELINE CODEGEN TESTS
# =============================================================================


def test_timeline_codegen_basic():
    """Test basic timeline codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show timeline "Events" from dataset test_data:
    items:
      - timestamp: event_time
        title:
          field: event_name
'''
    code = parse_build_and_codegen(source)
    
    assert "TimelineWidget" in code
    assert '"type": "timeline"' in code
    assert '"title": "Events"' in code
    assert '"items"' in code


def test_timeline_codegen_with_icons():
    """Test timeline codegen with icons and status."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show timeline "Activity" from dataset test_data:
    items:
      - timestamp: created_at
        icon: check
        status: success
        title:
          field: action
'''
    code = parse_build_and_codegen(source)
    
    assert "TimelineWidget" in code
    assert '"icon": "check"' in code
    assert '"status": "success"' in code


def test_timeline_codegen_with_grouping():
    """Test timeline codegen with date grouping."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show timeline "Log" from dataset test_data:
    group_by_date: true
    
    items:
      - timestamp: logged_at
        title:
          field: message
'''
    code = parse_build_and_codegen(source)
    
    assert "TimelineWidget" in code
    assert '"group_by_date": true' in code


# =============================================================================
# AVATAR GROUP CODEGEN TESTS
# =============================================================================


def test_avatar_group_codegen_basic():
    """Test basic avatar group codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show avatar_group "Team" from dataset test_data:
    items:
      - name_field: full_name
'''
    code = parse_build_and_codegen(source)
    
    assert "AvatarGroupWidget" in code
    assert '"type": "avatar_group"' in code
    assert '"title": "Team"' in code
    assert '"name_field": "full_name"' in code


def test_avatar_group_codegen_with_status():
    """Test avatar group codegen with status."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show avatar_group "Online" from dataset test_data:
    items:
      - image_field: avatar
        name_field: username
        status_field: status
    
    show_status: true
    size: lg
    max_visible: 5
'''
    code = parse_build_and_codegen(source)
    
    assert "AvatarGroupWidget" in code
    assert '"show_status": true' in code
    assert '"size": "lg"' in code
    assert '"max_visible": 5' in code


# =============================================================================
# DATA CHART CODEGEN TESTS
# =============================================================================


def test_data_chart_codegen_line():
    """Test line chart codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_chart "Sales" from dataset test_data:
    chart:
      type: line
      x_axis: month
      series:
        - data_key: sales
          label: "Sales"
'''
    code = parse_build_and_codegen(source)
    
    assert "DataChartWidget" in code
    assert '"type": "data_chart"' in code
    assert '"chart_config"' in code
    assert '"type": "line"' in code


def test_data_chart_codegen_bar():
    """Test bar chart codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_chart "Revenue" from dataset test_data:
    chart:
      type: bar
      x_axis: product
      legend: true
      grid: true
      
      series:
        - data_key: revenue
          label: "Revenue"
          color: "#10b981"
'''
    code = parse_build_and_codegen(source)
    
    assert "DataChartWidget" in code
    assert '"type": "bar"' in code
    assert '"legend": true' in code
    assert '"grid": true' in code


def test_data_chart_codegen_multi_series():
    """Test multi-series chart codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_chart "Comparison" from dataset test_data:
    chart:
      type: area
      x_axis: date
      
      series:
        - data_key: metric1
          label: "Metric 1"
          color: "#3b82f6"
        - data_key: metric2
          label: "Metric 2"
          color: "#10b981"
'''
    code = parse_build_and_codegen(source)
    
    assert "DataChartWidget" in code
    assert '"series"' in code
    assert '"metric1"' in code
    assert '"metric2"' in code


# =============================================================================
# EMPTY STATE CODEGEN TESTS
# =============================================================================


def test_empty_state_codegen():
    """Test empty state codegen."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_table "Items" from dataset test_data:
    columns:
      - field: id
        header: "ID"
    
    empty_state:
      icon: inbox
      title: "No items"
      message: "No items found"
      action_label: "Add Item"
      action_link: "/add"
'''
    code = parse_build_and_codegen(source)
    
    assert "DataTableWidget" in code
    assert '"empty_state"' in code
    assert '"icon": "inbox"' in code
    assert '"title": "No items"' in code
    assert '"action_label": "Add Item"' in code


# =============================================================================
# WIDGET IMPORT TESTS
# =============================================================================


def test_widget_imports_in_generated_code():
    """Test that all widget imports are included."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_table "T1" from dataset test_data:
    columns:
      - field: id
        header: "ID"
  
  show data_list "T2" from dataset test_data:
    item:
      title:
        field: name
  
  show stat_summary from dataset test_data:
    label: "Total"
    value:
      field: count
      format: number
'''
    code = parse_build_and_codegen(source)
    
    # Check that imports are present
    assert "import" in code
    assert "DataTableWidget" in code
    assert "DataListWidget" in code
    assert "StatSummaryWidget" in code


# =============================================================================
# DATA FETCHING TESTS
# =============================================================================


def test_data_fetching_in_codegen():
    """Test that data fetching is properly generated."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_table "Users" from dataset test_data:
    columns:
      - field: id
        header: "ID"
'''
    code = parse_build_and_codegen(source)
    
    # Check for data fetching patterns
    assert "test_data" in code or "dataset" in code.lower()
    assert "widget" in code.lower()


# =============================================================================
# WIDGET RENDERING TESTS
# =============================================================================


def test_widget_rendering_structure():
    """Test that widgets are properly rendered in page structure."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_chart "Sales" from dataset test_data:
    chart:
      type: line
      x_axis: month
      series:
        - data_key: sales
          label: "Sales"
'''
    code = parse_build_and_codegen(source)
    
    # Check rendering structure
    assert "DataChartWidget" in code
    assert "widget=" in code or "widget:" in code
    assert "data=" in code or "data:" in code


def test_multiple_widgets_rendering():
    """Test rendering multiple different widgets."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_table "Table" from dataset test_data:
    columns:
      - field: id
        header: "ID"
  
  show timeline "Timeline" from dataset test_data:
    items:
      - timestamp: created_at
        title:
          field: event
  
  show avatar_group "Avatars" from dataset test_data:
    items:
      - name_field: name
'''
    code = parse_build_and_codegen(source)
    
    # Check all widgets are rendered
    assert "DataTableWidget" in code
    assert "TimelineWidget" in code
    assert "AvatarGroupWidget" in code


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_full_page_with_all_components():
    """Test codegen for page with all data display components."""
    source = '''
dataset test_data:
  fields:
    - id: int

page dashboard:
  path: "/dashboard"
  title: "Dashboard"
  
  show stat_summary from dataset test_data:
    label: "Total"
    value:
      field: total
      format: number
  
  show data_chart "Chart" from dataset test_data:
    chart:
      type: line
      x_axis: date
      series:
        - data_key: value
          label: "Value"
  
  show data_table "Table" from dataset test_data:
    columns:
      - field: id
        header: "ID"
  
  show data_list "List" from dataset test_data:
    item:
      title:
        field: name
  
  show timeline "Timeline" from dataset test_data:
    items:
      - timestamp: time
        title:
          field: event
  
  show avatar_group "Team" from dataset test_data:
    items:
      - name_field: member
'''
    code = parse_build_and_codegen(source)
    
    # Verify all components are present
    assert "StatSummaryWidget" in code
    assert "DataChartWidget" in code
    assert "DataTableWidget" in code
    assert "DataListWidget" in code
    assert "TimelineWidget" in code
    assert "AvatarGroupWidget" in code
    
    # Verify page structure
    assert "dashboard" in code.lower() or "Dashboard" in code
