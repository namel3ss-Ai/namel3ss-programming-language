"""
Test IR transformation for data display components.
"""

import pytest
from namel3ss.parser import Parser
from namel3ss.ir.builder import IRBuilder
from namel3ss.ir.spec import (
    IRDataTable,
    IRDataList,
    IRStatSummary,
    IRTimeline,
    IRAvatarGroup,
    IRDataChart,
    IRColumnConfig,
    IRToolbarConfig,
    IRListItemConfig,
    IRSparklineConfig,
    IRTimelineItem,
    IRAvatarItem,
    IRChartConfig,
)


def parse_and_build_ir(source: str):
    """Helper to parse source and build IR."""
    parser = Parser()
    app = parser.parse(source)
    builder = IRBuilder()
    return builder.build(app)


# =============================================================================
# DATA TABLE IR TESTS
# =============================================================================


def test_data_table_ir_basic():
    """Test basic data table IR transformation."""
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
        width: "100px"
        sortable: true
      - field: name
        header: "Name"
        align: left
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    
    assert len(components) > 0
    
    table_component = next((c for c in components if c.get("type") == "data_table"), None)
    assert table_component is not None
    assert table_component["title"] == "Users"
    assert table_component["source"]["kind"] == "dataset"
    assert table_component["source"]["name"] == "test_data"
    assert len(table_component["columns"]) == 2
    assert table_component["columns"][0]["field"] == "id"
    assert table_component["columns"][0]["sortable"] is True


def test_data_table_ir_with_toolbar():
    """Test data table IR with toolbar configuration."""
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
      search_fields: ["name", "description"]
      filters:
        - field: status
          label: "Status"
          type: select
      bulk_actions:
        - label: "Delete"
          action: bulk_delete
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    table_component = next((c for c in components if c.get("type") == "data_table"), None)
    
    assert table_component is not None
    assert table_component["toolbar"] is not None
    assert table_component["toolbar"]["searchable"] is True
    assert len(table_component["toolbar"]["filters"]) == 1
    assert len(table_component["toolbar"]["bulk_actions"]) == 1


def test_data_table_ir_with_row_actions():
    """Test data table IR with row actions."""
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
    
    rows_per_page: 20
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    table_component = next((c for c in components if c.get("type") == "data_table"), None)
    
    assert table_component is not None
    assert len(table_component["row_actions"]) == 1
    assert table_component["rows_per_page"] == 20


# =============================================================================
# DATA LIST IR TESTS
# =============================================================================


def test_data_list_ir_basic():
    """Test basic data list IR transformation."""
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
      subtitle:
        text: "{{ description }}"
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    list_component = next((c for c in components if c.get("type") == "data_list"), None)
    
    assert list_component is not None
    assert list_component["title"] == "Activities"
    assert list_component["item_config"] is not None
    assert list_component["item_config"]["title"]["field"] == "event_name"
    assert list_component["item_config"]["subtitle"]["text"] == "{{ description }}"


def test_data_list_ir_with_avatar():
    """Test data list IR with avatar configuration."""
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
        field: profile_pic
        fallback: "U"
        size: md
      
      title:
        field: username
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    list_component = next((c for c in components if c.get("type") == "data_list"), None)
    
    assert list_component is not None
    assert list_component["item_config"]["avatar"] is not None
    assert list_component["item_config"]["avatar"]["field"] == "profile_pic"
    assert list_component["item_config"]["avatar"]["size"] == "md"


def test_data_list_ir_with_metadata_and_badge():
    """Test data list IR with metadata and badge."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_list "Tasks" from dataset test_data:
    item:
      title:
        field: task_name
      
      metadata:
        - field: due_date
          icon: clock
          format: date
        - field: assignee
          icon: user
      
      badge:
        field: priority
        style: priority_badge
      
      actions:
        - label: "Complete"
          action: complete_task
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    list_component = next((c for c in components if c.get("type") == "data_list"), None)
    
    assert list_component is not None
    assert len(list_component["item_config"]["metadata"]) == 2
    assert list_component["item_config"]["badge"] is not None
    assert len(list_component["item_config"]["actions"]) == 1


# =============================================================================
# STAT SUMMARY IR TESTS
# =============================================================================


def test_stat_summary_ir_basic():
    """Test basic stat summary IR transformation."""
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
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    stat_component = next((c for c in components if c.get("type") == "stat_summary"), None)
    
    assert stat_component is not None
    assert stat_component["label"] == "Total Revenue"
    assert stat_component["value"]["field"] == "revenue"
    assert stat_component["value"]["format"] == "currency"


def test_stat_summary_ir_with_delta_and_trend():
    """Test stat summary IR with delta and trend."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show stat_summary from dataset test_data:
    title: "Sales"
    label: "Monthly Sales"
    value:
      field: sales
      format: number
    
    delta:
      field: sales_change
      format: percentage
      show_sign: true
    
    trend:
      field: trend_direction
      up_is_good: true
    
    comparison: "vs last month"
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    stat_component = next((c for c in components if c.get("type") == "stat_summary"), None)
    
    assert stat_component is not None
    assert stat_component["title"] == "Sales"
    assert stat_component["delta"] is not None
    assert stat_component["delta"]["show_sign"] is True
    assert stat_component["trend"] is not None
    assert stat_component["comparison"] == "vs last month"


def test_stat_summary_ir_with_sparkline():
    """Test stat summary IR with sparkline."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show stat_summary from dataset test_data:
    label: "Page Views"
    value:
      field: views
      format: compact
    
    sparkline:
      data: view_history
      color: "#3b82f6"
      height: "40px"
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    stat_component = next((c for c in components if c.get("type") == "stat_summary"), None)
    
    assert stat_component is not None
    assert stat_component["sparkline"] is not None
    assert stat_component["sparkline"]["data"] == "view_history"
    assert stat_component["sparkline"]["color"] == "#3b82f6"


# =============================================================================
# TIMELINE IR TESTS
# =============================================================================


def test_timeline_ir_basic():
    """Test basic timeline IR transformation."""
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
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    timeline_component = next((c for c in components if c.get("type") == "timeline"), None)
    
    assert timeline_component is not None
    assert timeline_component["title"] == "Events"
    assert len(timeline_component["items"]) == 1
    assert timeline_component["items"][0]["timestamp"] == "event_time"


def test_timeline_ir_with_icons_and_status():
    """Test timeline IR with icons and status."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show timeline "Activity" from dataset test_data:
    group_by_date: true
    
    items:
      - timestamp: created_at
        icon: check
        status: success
        title:
          text: "{{ action }} completed"
        description:
          field: details
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    timeline_component = next((c for c in components if c.get("type") == "timeline"), None)
    
    assert timeline_component is not None
    assert timeline_component["group_by_date"] is True
    assert timeline_component["items"][0]["icon"] == "check"
    assert timeline_component["items"][0]["status"] == "success"


# =============================================================================
# AVATAR GROUP IR TESTS
# =============================================================================


def test_avatar_group_ir_basic():
    """Test basic avatar group IR transformation."""
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
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    avatar_component = next((c for c in components if c.get("type") == "avatar_group"), None)
    
    assert avatar_component is not None
    assert avatar_component["title"] == "Team"
    assert len(avatar_component["items"]) == 1
    assert avatar_component["items"][0]["name_field"] == "full_name"


def test_avatar_group_ir_with_status():
    """Test avatar group IR with status and configuration."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show avatar_group "Online Users" from dataset test_data:
    items:
      - image_field: avatar
        name_field: username
        status_field: status
        tooltip_template: "{{ username }}"
    
    max_visible: 5
    size: lg
    show_status: true
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    avatar_component = next((c for c in components if c.get("type") == "avatar_group"), None)
    
    assert avatar_component is not None
    assert avatar_component["max_visible"] == 5
    assert avatar_component["size"] == "lg"
    assert avatar_component["show_status"] is True
    assert avatar_component["items"][0]["status_field"] == "status"


# =============================================================================
# DATA CHART IR TESTS
# =============================================================================


def test_data_chart_ir_line():
    """Test line chart IR transformation."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_chart "Sales Trend" from dataset test_data:
    chart:
      type: line
      x_axis: month
      y_axis: sales
      legend: true
      grid: true
      
      series:
        - data_key: sales
          label: "Sales"
          color: "#3b82f6"
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    chart_component = next((c for c in components if c.get("type") == "data_chart"), None)
    
    assert chart_component is not None
    assert chart_component["title"] == "Sales Trend"
    assert chart_component["chart_config"]["type"] == "line"
    assert chart_component["chart_config"]["x_axis"] == "month"
    assert chart_component["chart_config"]["legend"] is True
    assert len(chart_component["chart_config"]["series"]) == 1


def test_data_chart_ir_multi_series():
    """Test multi-series chart IR transformation."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_chart "Revenue Comparison" from dataset test_data:
    chart:
      type: bar
      x_axis: product
      height: "400px"
      
      series:
        - data_key: q1
          label: "Q1"
          color: "#10b981"
        - data_key: q2
          label: "Q2"
          color: "#3b82f6"
        - data_key: q3
          label: "Q3"
          color: "#f59e0b"
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    chart_component = next((c for c in components if c.get("type") == "data_chart"), None)
    
    assert chart_component is not None
    assert chart_component["chart_config"]["type"] == "bar"
    assert chart_component["chart_config"]["height"] == "400px"
    assert len(chart_component["chart_config"]["series"]) == 3


def test_data_chart_ir_pie():
    """Test pie chart IR transformation."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_chart "Market Share" from dataset test_data:
    chart:
      type: pie
      series:
        - data_key: share
          label: "Share"
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    chart_component = next((c for c in components if c.get("type") == "data_chart"), None)
    
    assert chart_component is not None
    assert chart_component["chart_config"]["type"] == "pie"


# =============================================================================
# EMPTY STATE IR TESTS
# =============================================================================


def test_empty_state_ir():
    """Test empty state IR transformation."""
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
      title: "No items found"
      message: "Try adjusting your filters"
      action_label: "Add Item"
      action_link: "/add"
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    table_component = next((c for c in components if c.get("type") == "data_table"), None)
    
    assert table_component is not None
    assert table_component["empty_state"] is not None
    assert table_component["empty_state"]["icon"] == "inbox"
    assert table_component["empty_state"]["title"] == "No items found"
    assert table_component["empty_state"]["action_label"] == "Add Item"


# =============================================================================
# CONDITIONAL ACTION IR TESTS
# =============================================================================


def test_conditional_actions_ir():
    """Test conditional actions IR transformation."""
    source = '''
dataset test_data:
  fields:
    - id: int

page test:
  path: "/test"
  title: "Test"
  
  show data_list "Tasks" from dataset test_data:
    item:
      title:
        field: task_name
      
      actions:
        - label: "Complete"
          action: complete
          condition: "status == 'pending'"
        - label: "Reopen"
          action: reopen
          condition: "status == 'completed'"
'''
    ir = parse_and_build_ir(source)
    
    page = ir["pages"][0]
    components = page.get("components", [])
    list_component = next((c for c in components if c.get("type") == "data_list"), None)
    
    assert list_component is not None
    actions = list_component["item_config"]["actions"]
    assert len(actions) == 2
    assert actions[0]["condition"] == "status == 'pending'"
    assert actions[1]["condition"] == "status == 'completed'"
